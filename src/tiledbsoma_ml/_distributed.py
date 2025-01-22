# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
"""Utilities for multiprocess training: determine GPU "rank" / "world_size" and DataLoader worker ID / count."""

import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger("tiledbsoma_ml.pytorch")


def get_distributed_world_rank() -> Tuple[int, int]:
    """Return tuple containing equivalent of |torch.distributed| rank and world size."""
    rank, world_size = 0, 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Lightning doesn't use RANK! LOCAL_RANK is only for the local node. There
        # is a NODE_RANK for the node's rank, but no way to tell the local node's
        # world. So computing a global rank is impossible(?).  Using LOCAL_RANK as a
        # proxy, which works fine on a single-CPU box. TODO: could throw/error
        # if NODE_RANK != 0.
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    return rank, world_size


def get_worker_world_rank() -> Tuple[int, int]:
    """Return |DataLoader| ID, and the total number of |DataLoader| workers."""
    worker, num_workers = 0, 1
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker = worker_info.id
            num_workers = worker_info.num_workers
    return worker, num_workers


def init_multiprocessing() -> None:
    """Ensures use of "spawn" for starting child processes with multiprocessing.

    Note:
      - Forked processes are known to be problematic: `Avoiding and fighting deadlocks <https://pytorch.org/docs/stable/notes/multiprocessing.html#avoiding-and-fighting-deadlocks>`_.
      - CUDA does not support forked child processes: `CUDA in multiprocessing <https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing>`_.

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
