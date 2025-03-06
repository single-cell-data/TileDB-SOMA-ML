# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Callable, List, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from _pytest.fixtures import FixtureFunction
from pandas._testing import assert_frame_equal
from pytest import fixture
from scipy.sparse import coo_matrix, csr_matrix
from tiledbsoma._collection import CollectionBase
from torch.utils.data._utils.worker import WorkerInfo

from tiledbsoma_ml._common import MiniBatch, NDArrayJoinId

XValueGen = Callable[[range, range], coo_matrix]
ExpectedBatch = Tuple[List[List[int]], pd.DataFrame]


def assert_array_equal(
    actual: np.ndarray,
    expected: NDArrayJoinId | List[List[float]] | List[float],
    strict: bool = True,
):
    """Wrap :obj:`np.testing.assert_array_equal`, set some defaults and convert |List|'s to |np.ndarray|'s.

    - ``X`` batches are 1- or 2-D arrays of ``float32``s; this helper allows tests to pass |List| or |List| of
      |List|'s, for convenience.
    - Set ``strict=True`` by default (ensuring e.g. dtypes match).
    """
    if isinstance(expected, list):
        expected = np.array(expected, dtype=np.float32)
    np.testing.assert_array_equal(actual, expected, strict=strict)


parametrize = pytest.mark.parametrize


def default(value: Any) -> FixtureFunction:
    """Create a ``fixture`` that returns a default value.

    When using nested ``fixture``'s, this helps simulate "optional" parameters; test cases can explicitly set values for
    fixtures, but also omit them (which PyTest normally errors over).
    """
    return fixture(lambda: value)


def param(**kwargs):
    """``kwargs``-based wrapper around ``parametrize``, for fixtures that are constant in a given test case.

    Analogous to declaring "constant" variables in the fixture DAG. This is useful/common when some ``fixture``'s depend
    on other ``fixture``'s, and one or more of the dependencies have a fixed value in all "parametrizations" of a given
    test case (e.g. when other parameters are being "swept" over).

    For example:

    >>> @param(name='Alice', phone='111-111-1111')
    ... @sweep(dob=['1/1/2000', '1/1/3000'])  # Test multiple DOBs with this user name/phone
    ... def test_case(user):  # Fixture `user` is constructed from `name`, `phone`, and `dob`
    ...     ...

    Is equivalent to:

    >>> @parametrize("name,phone", [('Alice', '111-111-1111')])
    ... @parametrize("dob", [['1/1/2000', '1/1/3000']])  # Test multiple DOBs with this user name/phone
    ... def test_case(user):  # Fixture `user` is constructed from `name`, `phone`, and `dob`
    ...     ...
    """

    def rv(fn):
        for k, v in reversed(kwargs.items()):
            fn = parametrize(f"{k}", [v])(fn)

        return fn

    return rv


def sweep(**kwargs):
    """``kwargs``-based wrapper around ``parametrize``, for fixtures that are not constant in a given test case, e.g.:

    >>> @sweep(n=[1, 2, 3])
    ... def test_case(n):
    ...     ...

    Is equivalent to:

    >>> @parametrize("n", [1, 2, 3])
    ... def test_case(n):
    ...     ...
    """

    def rv(fn):
        for k, v in reversed(kwargs.items()):
            fn = parametrize(f"{k}", v)(fn)

        return fn

    return rv


def coords_to_float(r: int, c: int) -> float:
    """Encode a (row,col) position as a float.

    ``r`` is placed to the left of the decimal, while ``c`` is placed to the right, shifted 4 places.
    """
    assert c < 1e4
    return r + c * 1e-4


def pytorch_x_value_gen(obs_range: range, var_range: range) -> coo_matrix:
    """Create a sample sparse matrix for use in tests.

    The matrix has every other element nonzero, in a "checkerboard" pattern, and the nonzero elements encode their row-
    and column-indices, so tests can confidently assert that the contents of a given row are what's expected, e.g.:

    ```python
    assert_array_equal(
        pytorch_x_value_gen(range(5), range(5)),
        [
            [ 0 , 0.0001 , 0   ,    0.0003 , 0      ],
            [ 1 , 0      , 1.0002 , 0      , 1.0004 ],
            [ 0 , 2.0001 , 0   ,    2.0003 , 0      ],
            [ 3 , 0      , 3.0002 , 0      , 3.0004 ],
            [ 0 , 4.0001 , 0   ,    4.0003 , 0      ],
        ]
    )
    ```
    """
    obs_len = obs_range.stop - obs_range.start
    var_len = var_range.stop - var_range.start
    rows, cols, data = list(
        zip(
            *[
                (r + obs_range.start, c + var_range.start, coords_to_float(r, c))
                for r in range(obs_len)
                for c in range(var_len)
                if (r + c) % 2
            ]
        )
    )
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


def assert_batches_equal(
    batches: List[MiniBatch],
    expected_batches: List[ExpectedBatch],
    batch_size: int,
    return_sparse_X: bool = False,
):
    try:
        assert len(batches) == len(expected_batches)
        for (a_X, a_obs), (e_X, e_obs) in zip(
            map(lambda b: b.tpl, batches), expected_batches
        ):
            if return_sparse_X:
                assert isinstance(a_X, csr_matrix)
                a_X = a_X.toarray()
            if batch_size == 1 and not return_sparse_X:
                # Dense single-row batches are "squeezed" down to 1-D
                assert len(e_X) == 1
                e_X = e_X[0]
            assert_array_equal(a_X, e_X)
            assert_frame_equal(a_obs, e_obs)
    except AssertionError as e:
        # Raise an additional exception with a string representation of the "actual" row indices; useful for generating
        # "expected" data for test cases.
        raise AssertionError(f"Actual: {batch_idxs_str(batches)}") from e


def batch_idxs_str(batches: List[MiniBatch]) -> str:
    return "[%s]" % ", ".join(
        [
            "[%s]"
            % ", ".join(
                (
                    obs.soma_joinid.astype(str) if "soma_joinid" in obs else obs.label
                ).tolist()
            )
            for batch in batches
            for _, obs in [batch.tpl]
        ]
    )
