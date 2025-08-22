from __future__ import annotations

import pytest
import torch
from typing import Iterator, List
from pytest import fixture
from torch.utils.data import DataLoader

from tests._utils import param, parametrize
from tiledbsoma_ml._common import MiniBatch
from tiledbsoma_ml.dataloader import experiment_dataloader
from tiledbsoma_ml.dataset import ExperimentDataset

def _cuda_ready() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        # Prove driver+runtime actually work (avoids late AcceleratorError in workers)
        torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        return True
    except Exception:
        return False

# Skip the entire module before any fixtures run if CUDA isn't usable
pytestmark = pytest.mark.skipif(
    not _cuda_ready(),
    reason="CUDA not usable (unavailable or driver/runtime mismatch)",
)

@fixture
def dataloader(ds: ExperimentDataset, num_workers: int):
    """Wrap an ExperimentDataset fixture in a DataLoader, for use in tests."""
    yield experiment_dataloader(ds, num_workers=num_workers)

@fixture
def batch_iter(dataloader: DataLoader) -> Iterator[MiniBatch]:
    """Iterator over a DataLoader's MiniBatches."""
    return iter(dataloader)

@fixture
def batches(batch_iter: Iterator[MiniBatch]) -> List[MiniBatch]:
    """List of a DataLoader's MiniBatches."""
    return list(batch_iter)

# Reuse the GPU checker under the familiar name `check`
@fixture
def check(check_gpu):
    return check_gpu

@param(
    obs_range=40,
    shuffle_chunk_size=2,
    io_batch_size=4,
    batch_size=3,
    shuffle=True,
    shuffle_mode="gpu_iobatch",
    device=torch.device("cuda", 0),
)
@parametrize(
    "seed,rank,world_size,num_workers",
    [
        (False, 0, 2, 2),
        (False, 1, 2, 2),
        (111, 0, 2, 2),
        (111, 1, 2, 2),
    ],
)
def test_gpu_iobatch_worker_partitioning_even(check):
    pass

@param(
    obs_range=41,
    shuffle_chunk_size=2,
    io_batch_size=4,
    batch_size=3,
    shuffle=True,
    shuffle_mode="gpu_iobatch",
    device=torch.device("cuda", 0),
)
@parametrize("seed,rank,world_size,num_workers", [(False, 0, 2, 2), (False, 1, 2, 2)])
def test_gpu_iobatch_worker_partitioning_drop1(check):
    pass

@param(
    obs_range=42,
    shuffle_chunk_size=2,
    io_batch_size=4,
    batch_size=3,
    shuffle=True,
    shuffle_mode="gpu_iobatch",
    device=torch.device("cuda", 0),
)
@parametrize("seed,rank,world_size,num_workers", [(False, 0, 2, 2), (False, 1, 2, 2)])
def test_gpu_iobatch_worker_partitioning_uneven_workers(check):
    pass
