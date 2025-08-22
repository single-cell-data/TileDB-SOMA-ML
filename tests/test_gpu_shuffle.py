import torch
from pytest import fixture

from tests._utils import param, parametrize

from tiledbsoma_ml.dataloader import experiment_dataloader

@fixture
def dataloader(ds, num_workers):
    yield experiment_dataloader(ds, num_workers=num_workers)

@fixture
def batch_iter(dataloader):
    return iter(dataloader)

@fixture
def batches(batch_iter):
    return list(batch_iter)


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
@parametrize(
    "seed,rank,world_size,num_workers",
    [
        (False, 0, 2, 2),
        (False, 1, 2, 2),
    ],
)
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
@parametrize(
    "seed,rank,world_size,num_workers",
    [
        (False, 0, 2, 2),
        (False, 1, 2, 2),
    ],
)
def test_gpu_iobatch_worker_partitioning_uneven_workers(check):
    pass
