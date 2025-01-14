# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from typing import Iterator, List

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
from pytest import fixture, raises
from scipy import sparse
from somacore import AxisQuery
from tiledbsoma import Experiment

from tests._utils import (
    assert_array_equal,
    param,
    parametrize,
    pytorch_seq_x_value_gen,
    sweep,
)
from tests.conftest import rank_sweep, worker_sweep
from tiledbsoma_ml import ExperimentDataset
from tiledbsoma_ml.common import Batch


@fixture
def batch_iter(ds: ExperimentDataset) -> Iterator[Batch]:
    return iter(ds)


@fixture
def batches(batch_iter: Iterator[Batch]) -> List[Batch]:
    return list(batch_iter)


@param(obs_range=6, obs_column_names=["label"], shuffle=False)
@sweep(return_sparse_X=[True, False])
@sweep(use_eager_fetch=[True, False])
def test_non_batched(
    batches: List[Batch],
    return_sparse_X: bool,
):
    """Check batches of size 1 (the default)"""
    for idx, (X_batch, obs_batch) in enumerate(batches):
        expected_X = [0, idx + 0.1, 0] if idx % 2 == 0 else [idx, 0, idx + 0.2]
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            # Sparse batches are always 2D
            assert X_batch.shape == (1, 3)
            assert_array_equal(X_batch.todense(), [expected_X])
        else:
            assert isinstance(X_batch, np.ndarray)
            # Dense single-row batches are "squeezed" down to 1-D
            assert X_batch.shape == (3,)
            assert_array_equal(X_batch, expected_X)

        assert_frame_equal(obs_batch, pd.DataFrame({"label": [str(idx)]}))


@param(
    obs_range=6,
    obs_column_names=["label"],
    shuffle=False,
    batch_size=3,
    io_batch_size=2,
    expected_shape=(2, 3),
)
@sweep(return_sparse_X=[True, False])
@sweep(use_eager_fetch=[True, False])
def test_uneven_soma_and_result_batches(
    batch_iter: Iterator[Batch],
    return_sparse_X: bool,
):
    """Check that batches are correctly created when they require fetching multiple chunks."""
    X_batch, obs_batch = next(batch_iter)
    assert X_batch.shape == (3, 3)
    if return_sparse_X:
        assert isinstance(X_batch, sparse.csr_matrix)
        X_batch = X_batch.todense()
    else:
        assert isinstance(X_batch, np.ndarray)
    assert_array_equal(X_batch, [[0, 0.1, 0], [1, 0, 1.2], [0, 2.1, 0]])
    assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

    X_batch, obs_batch = next(batch_iter)
    assert X_batch.shape == (3, 3)
    if return_sparse_X:
        assert isinstance(X_batch, sparse.csr_matrix)
        X_batch = X_batch.todense()
    else:
        assert isinstance(X_batch, np.ndarray)
    assert_array_equal(X_batch, [[3, 0, 3.2], [0, 4.1, 0], [5, 0, 5.2]])
    assert_frame_equal(obs_batch, pd.DataFrame({"label": ["3", "4", "5"]}))


@param(
    obs_range=6,
    obs_column_names=["label"],
    shuffle=False,
    batch_size=3,
    expected_shape=(2, 3),
)
@sweep(return_sparse_X=[True, False])
@sweep(use_eager_fetch=[True, False])
def test_batching__all_batches_full_size(
    batch_iter: Iterator[Batch],
    return_sparse_X: bool,
):
    X_batch, obs_batch = next(batch_iter)
    if return_sparse_X:
        assert isinstance(X_batch, sparse.csr_matrix)
        X_batch = X_batch.todense()
    assert_array_equal(X_batch, [[0, 0.1, 0], [1, 0, 1.2], [0, 2.1, 0]])
    assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

    X_batch, obs_batch = next(batch_iter)
    if return_sparse_X:
        assert isinstance(X_batch, sparse.csr_matrix)
        X_batch = X_batch.todense()
    assert_array_equal(X_batch, [[3, 0, 3.2], [0, 4.1, 0], [5, 0, 5.2]])
    assert_frame_equal(obs_batch, pd.DataFrame({"label": ["3", "4", "5"]}))

    with raises(StopIteration):
        next(batch_iter)


@param(
    obs_range=range(100_000_000, 100_000_003),
    obs_column_names=["soma_joinid", "label"],
    shuffle=False,
    batch_size=3,
    expected_shape=(1, 3),
)
@sweep(use_eager_fetch=[True, False])
def test_soma_joinids(batches: List[Batch]):
    soma_joinids = np.concatenate([obs["soma_joinid"].to_numpy() for X, obs in batches])
    assert_array_equal(soma_joinids, np.arange(100_000_000, 100_000_003))


@param(
    obs_range=5,
    obs_column_names=["label"],
    shuffle=False,
    batch_size=3,
    expected_shape=(2, 3),
)
@sweep(return_sparse_X=[True, False])
@sweep(use_eager_fetch=[True, False])
def test_batching__partial_final_batch_size(
    batch_iter: Iterator[Batch],
    return_sparse_X: bool,
):
    next(batch_iter)
    X_batch, obs_batch = next(batch_iter)
    if return_sparse_X:
        assert isinstance(X_batch, sparse.csr_matrix)
        X_batch = X_batch.todense()
    assert_array_equal(X_batch, [[3, 0, 3.2], [0, 4.1, 0]])
    assert_frame_equal(obs_batch, pd.DataFrame({"label": ["3", "4"]}))

    with raises(StopIteration):
        next(batch_iter)


@param(
    obs_range=3,
    obs_column_names=["label"],
    shuffle=False,
    batch_size=3,
    expected_shape=(1, 3),
)
@sweep(use_eager_fetch=[True, False])
def test_batching__exactly_one_batch(batch_iter: Iterator[Batch]):
    X_batch, obs_batch = next(batch_iter)
    assert_array_equal(X_batch, [[0, 0.1, 0], [1, 0, 1.2], [0, 2.1, 0]])
    assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

    with raises(StopIteration):
        next(batch_iter)


@param(
    obs_range=6, obs_query=AxisQuery(coords=([],)), batch_size=3, expected_shape=(0, 3)
)
@sweep(use_eager_fetch=[True, False])
def test_batching__empty_query_result(batch_iter: Iterator[Batch]):
    with raises(StopIteration):
        next(batch_iter)


@param(obs_range=10, var_range=1, batch_size=3, io_batch_size=4, shuffle_chunk_size=4)
@sweep(use_eager_fetch=[True, False])
def test_batching__partial_soma_batches_are_concatenated(batches: List[Batch]):
    assert [len(X) for X, obs in batches] == [3, 3, 3, 1]


@sweep(obs_range=[6, 7])
@param(shuffle=False, io_batch_size=2)
@rank_sweep(2, 3)
def test_distributed__returns_data_partition_for_rank(
    batches: List[Batch],
    obs_range: int,
    rank: int,
    world_size: int,
):
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode, using mocks
    to avoid having to do real PyTorch distributed setup."""
    soma_joinids = np.concatenate([obs["soma_joinid"].to_numpy() for X, obs in batches])

    expected_joinids = np.array_split(np.arange(obs_range), world_size)[rank][
        0 : obs_range // world_size
    ].tolist()
    assert sorted(soma_joinids) == expected_joinids


# fmt: off
@param(shuffle=False, io_batch_size=2, seed=1234)
@rank_sweep(3)
@worker_sweep(2)
@parametrize(
    "obs_range,splits",
    [
        (12, [[0, 2, 4], [4,  6,  8], [ 8, 10, 12]]),
        (13, [[0, 2, 4], [5,  7,  9], [ 9, 11, 13]]),
        (15, [[0, 3, 5], [5,  8, 10], [10, 13, 15]]),
        (16, [[0, 3, 5], [6,  9, 11], [11, 14, 16]]),
        (18, [[0, 3, 6], [6,  9, 12], [12, 15, 18]]),
        (19, [[0, 3, 6], [7, 10, 13], [13, 16, 19]]),
        (20, [[0, 3, 6], [7, 10, 13], [14, 17, 20]]),
        (21, [[0, 4, 7], [7, 11, 14], [14, 18, 21]]),
        (25, [[0, 4, 8], [9, 13, 17], [17, 21, 25]]),
        (27, [[0, 5, 9], [9, 14, 18], [18, 23, 27]]),
    ],
)
# fmt: on
def test_distributed_and_multiprocessing__returns_data_partition_for_rank(
    batches: List[Batch],
    soma_experiment: Experiment,
    rank: int,
    worker_id: int,
    splits: list[list[int]],
) -> None:
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode and DataLoader
    multiprocessing mode, using mocks to avoid having to do distributed pytorch setup or real DataLoader
    multiprocessing."""
    proc_splits = splits[rank]
    expected_joinids = list(range(proc_splits[worker_id], proc_splits[worker_id + 1]))
    soma_joinids = np.concatenate(
        [obs["soma_joinid"].to_numpy() for X, obs in batches]
    ).tolist()
    assert soma_joinids == expected_joinids


@param(obs_range=16, var_range=1, X_value_gen=pytorch_seq_x_value_gen)
def test__shuffle(batches: List[Batch]):
    assert all(X.shape == (1,) for X, _ in batches)
    soma_joinids = [obs["soma_joinid"].iloc[0] for _, obs in batches]
    X_values = [X[0].item() for X, _ in batches]

    # same elements
    assert set(soma_joinids) == set(range(16))
    # not ordered! (...with a `1/16!` probability of being ordered)
    assert soma_joinids != list(range(16))
    # randomizes X in same order as obs
    # note: X values were explicitly set to match obs_joinids to allow for this simple assertion
    assert X_values == soma_joinids


@param(obs_range=6)
def test_experiment_axis_query_iterable_error_checks(
    soma_experiment: Experiment,
    ds: ExperimentDataset,
):
    with raises(NotImplementedError):
        ds[0]

    with soma_experiment.axis_query(measurement_name="RNA") as query:
        with raises(ValueError):
            ExperimentDataset(
                query,
                obs_column_names=(),
                layer_name="raw",
                shuffle=True,
            )
