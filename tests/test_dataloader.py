# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
from __future__ import annotations

from typing import Iterator, List
from unittest.mock import patch

from pytest import fixture, raises
from torch.utils.data import DataLoader

from tests._utils import (
    default,
    param,
    parametrize,
)
from tiledbsoma_ml.common import MiniBatch
from tiledbsoma_ml.dataloader import experiment_dataloader
from tiledbsoma_ml.dataset import ExperimentDataset

verify_dataset_shape = default(False)


@fixture
def dataloader(ds: ExperimentDataset, num_workers: int):
    """Wrap an |ExperimentDataset| fixture in a |DataLoader|, for use in tests."""
    yield experiment_dataloader(ds, num_workers=num_workers)


@fixture
def batch_iter(dataloader: DataLoader) -> Iterator[MiniBatch]:
    """Iterator over a |DataLoader|'s |MiniBatch|'s."""
    return iter(dataloader)


@fixture
def batches(batch_iter: Iterator[MiniBatch]) -> List[MiniBatch]:
    """List of a |DataLoader|'s |MiniBatch|'s."""
    return list(batch_iter)


# Turn off formatting from this point on; many test-case "expected" values are aligned across lines deliberately, in
# ways black/ruff would clobber.
# fmt: off


@param(obs_range=40, shuffle_chunk_size=2, io_batch_size=4, batch_size=3)
@parametrize("seed,rank,world_size,num_workers,expected", [
    (False, 0, 2, 2, [[ 0,  1,  2], [10, 11, 12], [ 3,  4,  5], [13, 14, 15], [ 6,  7,  8], [16, 17, 18], [ 9], [19]]),
    (False, 1, 2, 2, [[20, 21, 22], [30, 31, 32], [23, 24, 25], [33, 34, 35], [26, 27, 28], [36, 37, 38], [29], [39]]),
    (  111, 0, 2, 2, [[ 3,  2,  0], [13, 12, 10], [ 1,  5,  4], [11, 15, 14], [ 9,  8,  7], [19, 18, 17], [ 6], [16]]),
    (  111, 1, 2, 2, [[23, 22, 20], [33, 32, 30], [21, 25, 24], [31, 35, 34], [29, 28, 27], [39, 38, 37], [26], [36]]),
])
def test_gpu_worker_partitioning__even(check):
    """40 rows / 2 GPUs / [2 workers per GPU] = 10 rows per worker.

    Those 10 are then shuffled in chunks of 2, concatenated/shuffled/fetched in IO batches of 4, and re-batched for GPU
    in 3's. Each GPU rank then interleaves batches from its 2 workers.

    Note that each worker's row idxs are a constant offset of the others'; the same shuffle-seed is used on each
    worker's row-idx range.
    """
    pass


@param(obs_range=41, shuffle_chunk_size=2, io_batch_size=4, batch_size=3)
@parametrize("seed,rank,world_size,num_workers,expected", [
    (False, 0, 2, 2, [[ 0,  1,  2], [10, 11, 12], [ 3,  4,  5], [13, 14, 15], [ 6,  7,  8], [16, 17, 18], [ 9], [19]]),
    (False, 1, 2, 2, [[21, 22, 23], [31, 32, 33], [24, 25, 26], [34, 35, 36], [27, 28, 29], [37, 38, 39], [30], [40]]),  # 1 element dropped before 2nd GPU's range begins
    (  111, 0, 2, 2, [[ 3,  2,  0], [13, 12, 10], [ 1,  5,  4], [11, 15, 14], [ 9,  8,  7], [19, 18, 17], [ 6], [16]]),
    (  111, 1, 2, 2, [[24, 23, 21], [34, 33, 31], [22, 26, 25], [32, 36, 35], [30, 29, 28], [40, 39, 38], [27], [37]]),  # 1 element dropped before 2nd GPU's range begins
])
def test_gpu_worker_partitioning__drop1(check):
    pass


@param(obs_range=42, shuffle_chunk_size=2, io_batch_size=4, batch_size=3)
@parametrize("seed,rank,world_size,num_workers,expected", [
    (False, 0, 2, 2, [[ 0,  1,  2], [11, 12, 13], [ 3,  4,  5], [14, 15, 16], [ 6,  7,  8], [17, 18, 19], [ 9, 10], [20]]),
    (False, 1, 2, 2, [[21, 22, 23], [32, 33, 34], [24, 25, 26], [35, 36, 37], [27, 28, 29], [38, 39, 40], [30, 31], [41]]),
    (  111, 0, 2, 2, [[ 2, 10,  3], [14, 13, 11], [ 0,  4,  9], [12, 16, 15], [ 8,  1,  6], [20, 19, 18], [ 5,  7], [17]]),
    (  111, 1, 2, 2, [[23, 31, 24], [35, 34, 32], [21, 25, 30], [33, 37, 36], [29, 22, 27], [41, 40, 39], [26, 28], [38]]),
])
def test_gpu_worker_partitioning__uneven_workers(check):
    """21 rows and 2 workers for each of 2 GPUs, workers get 11 and 10 rows, resp.

    Shuffled in chunks of 2, fetched in IO batches of 4, finally re-batched in 3's for GPUs.
    """
    pass


def test_unsupported_params_fail():
    with patch("tiledbsoma_ml.dataset.ExperimentDataset") as mock_dataset:
        with raises(ValueError):
            experiment_dataloader(mock_dataset, shuffle=True)
        with raises(ValueError):
            experiment_dataloader(mock_dataset, batch_size=3)
        with raises(ValueError):
            experiment_dataloader(mock_dataset, batch_sampler=[])
        with raises(ValueError):
            experiment_dataloader(mock_dataset, sampler=[])
