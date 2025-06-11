# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, TypeVar

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

    if "collate_fn" not in dataloader_kwargs:
        dataloader_kwargs["collate_fn"] = _collate_noop

    return DataLoader(
        ds,
        batch_size=None,  # batching is handled by upstream iterator
        shuffle=False,  # shuffling is handled by upstream iterator
        **dataloader_kwargs,
    )


def optimized_experiment_dataloader(
    ds: ExperimentDataset,
    num_workers: int = 2,
    pin_memory: bool = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Optimized |DataLoader| factory for maximum performance with GPU training.
    
    This function provides optimized defaults for high-performance GPU training,
    including automatic pin_memory detection, worker optimization, and memory management.
    
    Args:
        ds: A |ExperimentDataset| to wrap.
        num_workers: Number of worker processes. Defaults to 2 for optimal S3 performance.
        pin_memory: Whether to pin memory. Auto-detected if None.
        persistent_workers: Keep workers alive between epochs. Defaults to True.
        prefetch_factor: Samples loaded ahead by each worker. Defaults to 2.
        **dataloader_kwargs: Additional DataLoader arguments.
        
    Returns:
        Optimized |DataLoader| instance.
        
    Lifecycle:
        experimental
    """
    import torch
    
    # Auto-detect optimal pin_memory setting
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # Set optimized defaults
    optimized_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "prefetch_factor": prefetch_factor,
        "drop_last": False,  # Don't drop last incomplete batch
        **dataloader_kwargs
    }
    
    return experiment_dataloader(ds, **optimized_kwargs)


def _collate_noop(datum: _T) -> _T:
    """Noop collation used by |experiment_dataloader|.

    Private.
    """
    return datum
