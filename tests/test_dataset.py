# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
from __future__ import annotations

from typing import Iterator, List, Sequence

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
from pytest import fixture, raises
from scipy import sparse
from somacore import AxisQuery
from tiledbsoma import Experiment

from tests._utils import (
    XValueGen,
    assert_array_equal,
    param,
    parametrize,
    sweep,
)
from tiledbsoma_ml import ExperimentDataset
from tiledbsoma_ml.common import Batch


@fixture
def batch_iter(ds: ExperimentDataset) -> Iterator[Batch]:
    """Iterator over an |ExperimentDataset|'s |Batch|'s."""
    return iter(ds)


@fixture
def batches(batch_iter: Iterator[Batch]) -> List[Batch]:
    """List of an |ExperimentDataset|'s |Batch|'s."""
    return list(batch_iter)


@fixture
def check(
    expected: List[List[int]],  # Batches of row idxs
    ds: ExperimentDataset,
    batches: List[Batch],
    soma_experiment: Experiment,
    obs_range: int | range,
    var_range: int | range,
    X_value_gen: XValueGen,
    obs_column_names: Sequence[str],
    batch_size: int,
    return_sparse_X: bool,
):
    """Verify the ``batches`` produced by ``ds`` match the ``expected`` values.

    ``expected`` contains row indices, which are mapped to ``X`` and ``obs`` objects using the same ``X_value_gen`` /
    ``obs_column_names`` that were used to create the ``soma_experiment``.

    Test cases invoke this function by declaring a ``check`` argument. All arguments are themselves ``fixture``s, which
    pytest will evaluate/reuse.
    """
    try:
        assert ds.shape == (len(expected), var_range)
        obs = soma_experiment.obs.read().concat().to_pandas()
        if isinstance(obs_range, int):
            obs_range = range(obs_range)
        if isinstance(var_range, int):
            var_range = range(var_range)

        X = X_value_gen(obs_range, var_range)
        X = X.tocsr()
        expected_batches = [
            (
                [
                    X[row_idx + obs_range.start].toarray()[0].tolist()
                    for row_idx in row_idx_batch
                ],
                pd.concat(
                    [obs.loc[[row_idx], obs_column_names] for row_idx in row_idx_batch]
                ).reset_index(drop=True),
            )
            for row_idx_batch in expected
        ]
        assert len(batches) == len(expected_batches)
        for (a_X, a_obs), (e_X, e_obs) in zip(batches, expected_batches):
            if return_sparse_X:
                assert isinstance(a_X, sparse.csr_matrix)
                a_X = a_X.toarray()
            if batch_size == 1 and not return_sparse_X:
                # Dense single-row batches are "squeezed" down to 1-D
                assert len(e_X) == 1
                e_X = e_X[0]
            assert_array_equal(a_X, e_X)
            assert_frame_equal(a_obs, e_obs)
    except AssertionError as e:
        # Raise an additional exception with a string representation of the actual values; useful for generating the
        # "expected" data for test cases.
        actual_ids = ", ".join(
            [
                "[%s]"
                % ", ".join(
                    (
                        obs.soma_joinid.astype(str)
                        if "soma_joinid" in obs
                        else obs.label
                    ).tolist()
                )
                for _, obs in batches
            ]
        )
        raise AssertionError(f"Actual: {actual_ids}") from e


# Turn off formatting from this point on; many test-case "expected" values are aligned across lines deliberately, in
# ways black/ruff would clobber.
# fmt: off

@sweep(io_batch_size=[ 2, 6, 10 ])
@param(obs_range=10, shuffle=False, batch_size=3, expected=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
def test_io_batch_sizes_unshuffled(check):
    """When ``shuffle=False``, ``io_batch_size`` doesn't affect the emitted batches."""
    pass


@param(obs_range=10, shuffle_chunk_size=2, io_batch_size=2, batch_size=3)
@parametrize("seed,expected", [
    # `shuffle_chunk_size=2, io_batch_size=2` means consecutive integers come in pairs (even across `batch_size=3` divisions)
    #      |     |       |      |      |
    (111, [[2, 3, 1], [0, 9, 8], [5, 4, 6], [7]]),
    (222, [[0, 1, 7], [6, 3, 2], [8, 9, 4], [5]]),
])
def test_obs10_shuf2_io2_batch3(check):
    pass


@param(obs_range=30, shuffle_chunk_size=3, io_batch_size=6, batch_size=4)
@parametrize("seed,expected", [
    # Batches are size 4, but every 6 consecutive idxs are an "IO batch" consisting of two "shuffled chunks" of size 3:
    #      |                         |                          |                          |                          |
    (111, [[26,  4,  3, 25], [ 5, 24,  1,  2], [ 0, 29, 27, 28], [17, 22, 15, 21], [23, 16, 14, 18], [20, 12, 13, 19], [ 8,  6,  9, 11], [ 7, 10]]),
    (222, [[25, 26, 27, 24], [28, 29, 17, 15], [21, 23, 22, 16], [ 0, 14,  1,  2], [12, 13, 10,  5], [11,  3,  9,  4], [ 7, 19,  6, 18], [20,  8]]),
])
def test_obs30_shuf3_io6_batch4(check):
    pass


@param(obs_range=10, shuffle_chunk_size=2, io_batch_size=4, batch_size=1)
@parametrize("seed,expected", [
    # Each IO batch (4 elems) consists of two shuffle chunks (2 elems each), shuffled:
    #      |                  |                   |
    (111, [[3], [2], [0], [1], [5], [4], [9], [8], [7], [6]]),
    (444, [[7], [6], [8], [9], [1], [5], [4], [0], [3], [2]]),
])
def test_obs10_shuf2_io4_batch1(check):
    pass


@param(obs_range=3, shuffle_chunk_size=3, io_batch_size=3, batch_size=3)
@parametrize("seed,expected", [
    (False, [[0, 1, 2]]),
    (  111, [[1, 0, 2]]),
])
def test_obs3_shuf3_io3_batch3(check):
    """Emit exactly one batch."""
    pass


@param(
    obs_range=range(100_000_000, 100_000_003),
    shuffle=False,
    batch_size=3,
    expected=[[0, 1, 2]]
)
@sweep(use_eager_fetch=[True, False])
def test_large_obs_ids(batches: List[Batch], check):
    soma_joinids = np.concatenate([obs["soma_joinid"].to_numpy() for X, obs in batches])
    assert_array_equal(soma_joinids, np.arange(100_000_000, 100_000_003))


@param(obs_range=6, obs_query=AxisQuery(coords=([],)), batch_size=3)
@sweep(use_eager_fetch=[True, False])
def test_batching__empty_query_result(ds: ExperimentDataset, batch_iter: Iterator[Batch]):
    with raises(StopIteration):
        next(batch_iter)
    assert ds.shape == (0, 3)


@param(obs_range=40, world_size=2, num_workers=2, shuffle_chunk_size=2, io_batch_size=4, batch_size=3)
@parametrize("seed,rank,worker_id,expected", [
    (False, 0, 0, [[ 0,  1,  2], [ 3,  4,  5], [ 6,  7,  8], [ 9]]),
    (False, 0, 1, [[10, 11, 12], [13, 14, 15], [16, 17, 18], [19]]),
    (False, 1, 0, [[20, 21, 22], [23, 24, 25], [26, 27, 28], [29]]),
    (False, 1, 1, [[30, 31, 32], [33, 34, 35], [36, 37, 38], [39]]),
    (  111, 0, 0, [[ 3,  2,  0], [ 1,  5,  4], [ 9,  8,  7], [ 6]]),
    (  111, 0, 1, [[13, 12, 10], [11, 15, 14], [19, 18, 17], [16]]),
    (  111, 1, 0, [[23, 22, 20], [21, 25, 24], [29, 28, 27], [26]]),
    (  111, 1, 1, [[33, 32, 30], [31, 35, 34], [39, 38, 37], [36]]),
])
def test_gpu_worker_partitioning__even(check):
    """40 rows / 2 GPUs / [2 workers per GPU] = 10 rows per worker.

    Those 10 are then shuffled in chunks of 2, concatenated/shuffled/fetched in IO batches of 4, and re-batched for GPU
    in 3's.

    Note that each worker's row idxs are a constant offset of the others'; the same shuffle-seed is used on each
    worker's row-idx range.
    """
    pass


@param(obs_range=41, world_size=2, num_workers=2, shuffle_chunk_size=2, io_batch_size=4, batch_size=3)
@parametrize("seed,rank,worker_id,expected", [
    (False, 0, 0, [[ 0,  1,  2], [ 3,  4,  5], [ 6,  7,  8], [ 9]]),
    (False, 0, 1, [[10, 11, 12], [13, 14, 15], [16, 17, 18], [19]]),
    (False, 1, 0, [[21, 22, 23], [24, 25, 26], [27, 28, 29], [30]]),  # 1 element dropped before 2nd GPU's range begins
    (False, 1, 1, [[31, 32, 33], [34, 35, 36], [37, 38, 39], [40]]),
    (  111, 0, 0, [[ 3,  2,  0], [ 1,  5,  4], [ 9,  8,  7], [ 6]]),
    (  111, 0, 1, [[13, 12, 10], [11, 15, 14], [19, 18, 17], [16]]),
    (  111, 1, 0, [[24, 23, 21], [22, 26, 25], [30, 29, 28], [27]]),  # 1 element dropped before 2nd GPU's range begins
    (  111, 1, 1, [[34, 33, 31], [32, 36, 35], [40, 39, 38], [37]]),
])
def test_gpu_worker_partitioning__drop1(check):
    """Same as above, but with 41 obs_joinids, 1 gets dropped (index 20), so that each GPU is assigned 20 rows."""
    pass


@param(obs_range=42, world_size=2, num_workers=2, shuffle_chunk_size=2, io_batch_size=4, batch_size=3)
@parametrize("seed,rank,worker_id,expected", [
    (False, 0, 0, [[ 0,  1,  2], [ 3,  4,  5], [ 6,  7,  8], [ 9, 10]]),
    (False, 0, 1, [[11, 12, 13], [14, 15, 16], [17, 18, 19], [20,   ]]),
    (False, 1, 0, [[21, 22, 23], [24, 25, 26], [27, 28, 29], [30, 31]]),
    (False, 1, 1, [[32, 33, 34], [35, 36, 37], [38, 39, 40], [41,   ]]),
    (  111, 0, 0, [[ 2, 10,  3], [ 0,  4,  9], [ 8,  1,  6], [ 5,  7]]),
    (  111, 0, 1, [[14, 13, 11], [12, 16, 15], [20, 19, 18], [17,   ]]),
    (  111, 1, 0, [[23, 31, 24], [21, 25, 30], [29, 22, 27], [26, 28]]),
    (  111, 1, 1, [[35, 34, 32], [33, 37, 36], [41, 40, 39], [38,   ]]),
])
def test_gpu_worker_partitioning__uneven_workers(check):
    """21 rows and 2 workers for each of 2 GPUs, workers get 11 and 10 rows, resp.

    Shuffled in chunks of 2, fetched in IO batches of 4, finally re-batched in 3's for GPUs.
    """
    pass


@param(obs_range=6)
def test_experiment_dataset_error_checks(
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
