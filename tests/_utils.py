# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Callable, List, Tuple
from unittest.mock import patch

import numpy as np
import pyarrow as pa
from scipy.sparse import coo_matrix, spmatrix
from tiledbsoma._collection import CollectionBase
from torch.utils.data._utils.worker import WorkerInfo

assert_array_equal = partial(np.testing.assert_array_equal, strict=True)


def assert_array_almost_equal(
    actual: np.ndarray, expected: List[List[float]] | List[float]
):
    np.testing.assert_array_almost_equal(actual, np.array(expected, dtype=np.float32))


XValueGen = Callable[[range, range], spmatrix]


def coords_to_float(r: int, c: int) -> float:
    return float(f"{r}.{str(c)[::-1]}")


def pytorch_x_value_gen(obs_range: range, var_range: range) -> spmatrix:
    occupied_shape = (
        obs_range.stop - obs_range.start,
        var_range.stop - var_range.start,
    )
    floats = np.array(
        [
            [coords_to_float(r, c) for c in range(var_range.start, var_range.stop)]
            for r in range(obs_range.start, obs_range.stop)
        ]
    )
    checkerboard_of_ones = coo_matrix(
        (np.indices(occupied_shape).sum(axis=0) % 2) * floats
    )
    checkerboard_of_ones.row += obs_range.start
    checkerboard_of_ones.col += var_range.start
    return checkerboard_of_ones


def pytorch_seq_x_value_gen(obs_range: range, var_range: range) -> spmatrix:
    """A sparse matrix where the values of each col are the obs_range values. Useful for checking the
    X values are being returned in the correct order."""
    data = np.vstack([list(obs_range)] * len(var_range)).flatten()
    rows = np.vstack([list(obs_range)] * len(var_range)).flatten()
    cols = np.column_stack([list(var_range)] * len(obs_range)).flatten()
    return coo_matrix((data, (rows, cols)))


def add_dataframe(coll: CollectionBase, key: str, value_range: range) -> None:
    df = coll.add_new_dataframe(
        key,
        schema=pa.schema(
            [
                ("soma_joinid", pa.int64()),
                ("label", pa.large_string()),
                ("label2", pa.large_string()),
            ]
        ),
        index_column_names=["soma_joinid"],
        domain=((value_range.start, value_range.stop),),
    )
    df.write(
        pa.Table.from_pydict(
            {
                "soma_joinid": list(value_range),
                "label": [str(i) for i in value_range],
                "label2": ["c" for i in value_range],
            }
        )
    )


def add_sparse_array(
    coll: CollectionBase,
    key: str,
    obs_range: range,
    var_range: range,
    value_gen: XValueGen,
) -> None:
    a = coll.add_new_sparse_ndarray(
        key, type=pa.float32(), shape=(obs_range.stop, var_range.stop)
    )
    tensor = pa.SparseCOOTensor.from_scipy(value_gen(obs_range, var_range))
    a.write(tensor)


@contextmanager
def mock_dist_is_initialized():
    with patch("torch.distributed.is_initialized") as mock_dist_is_initialized:
        mock_dist_is_initialized.return_value = True
        yield


@contextmanager
def patch_worker_info(worker_id: int, num_workers: int, seed: int):
    with patch("torch.utils.data.get_worker_info") as mock_get_worker_info:
        mock_get_worker_info.return_value = WorkerInfo(
            id=worker_id, num_workers=num_workers, seed=seed
        )
        yield


@contextmanager
def mock_distributed(
    rank: int = 0,
    world_size: int = 1,
    worker: Tuple[int, int, int] | None = None,
):
    worker_ctx = patch_worker_info(*worker) if worker else nullcontext()
    with (
        mock_dist_is_initialized(),
        patch("torch.distributed.get_rank") as mock_dist_get_rank,
        patch("torch.distributed.get_world_size") as mock_dist_get_world_size,
        worker_ctx,
    ):
        mock_dist_get_rank.return_value = rank
        mock_dist_get_world_size.return_value = world_size
        yield
