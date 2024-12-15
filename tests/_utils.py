# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from functools import partial
from typing import Callable, Type, Union

import numpy as np
import pyarrow as pa
from scipy.sparse import coo_matrix, spmatrix
from tiledbsoma._collection import CollectionBase

from tiledbsoma_ml import (
    ExperimentAxisQueryIterableDataset,
    ExperimentAxisQueryIterDataPipe,
)
from tiledbsoma_ml.batch_iterable import BatchIterable

assert_array_equal = partial(np.testing.assert_array_equal, strict=True)

# These control which classes are tested (for most, but not all tests).
# Centralized to allow easy add/delete of specific test parameters.
IterableWrapperType = Union[
    Type[ExperimentAxisQueryIterDataPipe],
    Type[ExperimentAxisQueryIterableDataset],
]
IterableWrappers = (
    ExperimentAxisQueryIterDataPipe,
    ExperimentAxisQueryIterableDataset,
)
PipeClassType = Union[
    Type[BatchIterable],
    IterableWrapperType,
]
PipeClasses = (
    BatchIterable,
    *IterableWrappers,
)
XValueGen = Callable[[range, range], spmatrix]


def pytorch_x_value_gen(obs_range: range, var_range: range) -> spmatrix:
    occupied_shape = (
        obs_range.stop - obs_range.start,
        var_range.stop - var_range.start,
    )
    checkerboard_of_ones = coo_matrix(np.indices(occupied_shape).sum(axis=0) % 2)
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
