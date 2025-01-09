# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger("tiledbsoma_ml.pytorch")


def get_distributed_world_rank() -> Tuple[int, int]:
    """Return tuple containing equivalent of ``torch.distributed`` world size and rank."""
    world_size, rank = 1, 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Lightning doesn't use RANK! LOCAL_RANK is only for the local node. There
        # is a NODE_RANK for the node's rank, but no way to tell the local node's
        # world. So computing a global rank is impossible(?).  Using LOCAL_RANK as a
        # proxy, which works fine on a single-CPU box. TODO: could throw/error
        # if NODE_RANK != 0.
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["LOCAL_RANK"])
    elif torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    return world_size, rank


def get_worker_world_rank() -> Tuple[int, int]:
    """Return number of DataLoader workers and our worker rank/id."""
    num_workers, worker = 1, 0
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        num_workers = int(os.environ["NUM_WORKERS"])
        worker = int(os.environ["WORKER"])
    else:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker = worker_info.id
    return num_workers, worker


def init_multiprocessing() -> None:
    """Ensures use of "spawn" for starting child processes with multiprocessing.

    Forked processes are known to be problematic:
      https://pytorch.org/docs/stable/notes/multiprocessing.html#avoiding-and-fighting-deadlocks
    Also, CUDA does not support forked child processes:
      https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing

    Private.
    """
    orig_start_method = torch.multiprocessing.get_start_method()
    if orig_start_method != "spawn":
        if orig_start_method:
            logger.warning(
                "switching torch multiprocessing start method from "
                f'"{torch.multiprocessing.get_start_method()}" to "spawn"'
            )
        torch.multiprocessing.set_start_method("spawn", force=True)
