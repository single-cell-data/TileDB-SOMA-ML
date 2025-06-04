# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, TypeVar
import os
import numpy as np
import torch

from torch.utils.data import DataLoader

from tiledbsoma_ml._distributed import init_multiprocessing
from tiledbsoma_ml.dataset import ExperimentDataset

_T = TypeVar("_T")


def experiment_dataloader(
    ds: ExperimentDataset,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """|DataLoader| factory method for safely wrapping an |ExperimentDataset|.

    Several |DataLoader| constructor parameters are not applicable, or are non-performant when using loaders from this
    module, including ``shuffle``, ``batch_size``, ``sampler``, and ``batch_sampler``. Specifying any of these
    parameters will result in an error.

    Refer to `the DataLoader docs <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ for more
    information on |DataLoader| parameters, and |ExperimentDataset| for info on corresponding parameters.

    Args:
        ds:
            A |IterableDataset|. May include chained data pipes.
        **dataloader_kwargs:
            Additional keyword arguments to pass to the |DataLoader| constructor,
            except for ``shuffle``, ``batch_size``, ``sampler``, and ``batch_sampler``, which are not
            supported when using data loaders in this module.

    Returns:
        |DataLoader|

    Raises:
        ValueError: if any of the ``shuffle``, ``batch_size``, ``sampler``, or ``batch_sampler`` params
            are passed as keyword arguments.

    Lifecycle:
        experimental
    """
    unsupported_dataloader_args = [
        "shuffle",
        "batch_size",
        "sampler",
        "batch_sampler",
    ]
    if set(unsupported_dataloader_args).intersection(dataloader_kwargs.keys()):
        raise ValueError(
            f"The {','.join(unsupported_dataloader_args)} DataLoader parameters are not supported"
        )

    if dataloader_kwargs.get("num_workers", 0) > 0:
        init_multiprocessing()

    # Set default optimized parameters if not specified
    default_kwargs = {
        "num_workers": min(8, os.cpu_count() or 4),  # Optimize worker count
        "pin_memory": True,  # Enable pinned memory for faster GPU transfer
        "prefetch_factor": 3,  # Increase prefetch factor
        "persistent_workers": True,  # Keep workers alive between epochs
        "multiprocessing_context": "spawn",  # More stable than fork
    }

    # Update with user provided kwargs, preserving defaults if not specified
    dataloader_kwargs = {**default_kwargs, **dataloader_kwargs}

    if "collate_fn" not in dataloader_kwargs:
        dataloader_kwargs["collate_fn"] = _optimized_collate

    return DataLoader(
        ds,
        batch_size=None,  # batching is handled by upstream iterator
        shuffle=False,  # shuffling is handled by upstream iterator
        **dataloader_kwargs,
    )


def _collate_noop(datum: _T) -> _T:
    """Noop collation used by |experiment_dataloader|.

    Private.
    """
    return datum


def _optimized_collate(batch: list[_T]) -> _T:
    """Optimized collate function that pre-allocates memory and uses vectorized operations.
    
    This function is designed to efficiently collate batches of data, minimizing memory
    allocations and using vectorized operations where possible.
    
    Args:
        batch: List of data items to collate
        
    Returns:
        Collated batch
    """
    if not batch:
        return None
        
    # If the batch contains numpy arrays, convert them to tensors efficiently
    if isinstance(batch[0], np.ndarray):
        # Pre-allocate memory for the batch
        batch_size = len(batch)
        sample_shape = batch[0].shape
        dtype = batch[0].dtype
        
        # Create a contiguous tensor for the entire batch
        result = torch.empty((batch_size, *sample_shape), dtype=torch.from_numpy(np.array(0, dtype=dtype)).dtype)
        
        # Copy data efficiently using vectorized operations
        for i, item in enumerate(batch):
            result[i] = torch.from_numpy(item)
            
        return result
        
    # For other types, use the default collate behavior
    return _collate_noop(batch)
