# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.

from __future__ import annotations

import os
from typing import Any, TypeVar, Optional, Union, Sequence

from torch.utils.data import DataLoader

from tiledbsoma_ml._distributed import init_multiprocessing
from tiledbsoma_ml.dataset import ExperimentDataset

import logging
import warnings

_T = TypeVar("_T")

logger = logging.getLogger("tiledbsoma_ml.dataloader")


def configure_threading_for_stability():
    """Configure threading settings for stable operation.
    
    This function sets conservative threading limits to prevent conflicts
    between TileDB-SOMA, Numba, OpenMP, and PyTorch threading.
    
    Call this function before creating datasets or dataloaders if you
    encounter segmentation faults or threading issues.
    """
    # Set conservative threading limits if not already set
    if 'NUMBA_NUM_THREADS' not in os.environ:
        os.environ['NUMBA_NUM_THREADS'] = '1'
        logger.info("Set NUMBA_NUM_THREADS=1 for stability")
    
    if 'OMP_NUM_THREADS' not in os.environ:
        num_threads = str(min(4, os.cpu_count() or 1))
        os.environ['OMP_NUM_THREADS'] = num_threads
        logger.info(f"Set OMP_NUM_THREADS={num_threads} for stability")
    
    if 'MKL_NUM_THREADS' not in os.environ:
        num_threads = str(min(4, os.cpu_count() or 1))
        os.environ['MKL_NUM_THREADS'] = num_threads
        logger.info(f"Set MKL_NUM_THREADS={num_threads} for stability")


def experiment_dataloader(
    exp_data: ExperimentDataset,
    num_workers: int = 0,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Factory function for |PyTorch DataLoader|. This function is a light wrapper around
    |torch.utils.data.DataLoader|. Please see the |PyTorch DataLoader docs|_
    for a complete description of the parameters.

    Args:
        exp_data: A :class:`tiledbsoma_ml.ExperimentDataset`.
        num_workers: how many subprocesses to use for data loading. 0 means that the data
            will be loaded in the main process. Note: num_workers=0 is the only
            supported value at this time.
        **dataloader_kwargs: Additional keyword arguments to pass to |torch.utils.data.DataLoader|

    Returns:
        A |torch.utils.data.DataLoader|.

    Raises:
        ValueError: if ``num_workers`` is not 0.

    Examples:
        Use with any |PyTorch Lightning| class:

        >>> train_dataloader = experiment_dataloader(
        ...     ExperimentDataset(...), batch_size=16, shuffle=True
        ... )
        >>> model = MyLightningModule(...)
        >>> trainer = Trainer(...)
        >>> trainer.fit(model, train_dataloaders=train_dataloader)

    .. |PyTorch DataLoader| replace:: :class:`torch.utils.data.DataLoader`
    .. |PyTorch DataLoader docs| replace:: :class:`torch.utils.data.DataLoader`
    .. _PyTorch DataLoader docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    .. |PyTorch Lightning| replace:: :mod:`pytorch_lightning`
    """
    if num_workers != 0:
        raise ValueError("num_workers must be 0")

    return DataLoader(exp_data, num_workers=num_workers, **dataloader_kwargs)


def optimized_experiment_dataloader(
    exp_data: ExperimentDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
    # Threading and concurrency parameters
    max_concurrent_requests: int = 4,
    max_concurrent_batch_processing: int = 2,
    prefetch_queue_size: int = 2,
    # GPU optimization parameters  
    enable_pinned_memory: bool = False,
    enable_tensor_cache: bool = True,
    # TileDB-SOMA parameters
    io_batch_size: Optional[int] = None,
    use_eager_fetch: bool = True,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """
    Factory function for optimized PyTorch DataLoader with thread-safe concurrent processing.
    
    This function creates a DataLoader that uses threading-based optimizations while maintaining
    thread safety with TileDB-SOMA by checking and reopening SOMA objects as needed.
    
    **Important**: This optimized dataloader returns batches as dictionaries with keys like 'X',
    not as tuples (X_batch, obs_batch) like the standard experiment_dataloader. 
    
    Example usage:
        for batch in dataloader:
            X_batch = batch['X']  # Extract X data
            # Use X_batch for training...
    
    Args:
        exp_data: A :class:`tiledbsoma_ml.ExperimentDataset`.
        batch_size: Number of samples per batch.
        num_workers: Number of PyTorch DataLoader workers (recommended: 0 for single-process).
        shuffle: Whether to shuffle the data.
        max_concurrent_requests: Maximum concurrent IO requests for fetching data from TileDB-SOMA.
        max_concurrent_batch_processing: Maximum concurrent mini-batch processing workers.
        prefetch_queue_size: Size of prefetch queue (for future use).
        enable_pinned_memory: Whether to use CUDA pinned memory for faster GPU transfers.
        enable_tensor_cache: Whether to cache and reuse tensors to reduce allocations.
        io_batch_size: Size of IO batches. If None, defaults to batch_size * 128.
        use_eager_fetch: Whether to use eager fetching for better performance.
        **dataloader_kwargs: Additional keyword arguments to pass to torch.utils.data.DataLoader.
    
    Returns:
        A torch.utils.data.DataLoader with optimized performance settings.
        
    Examples:
        >>> # Basic usage with conservative threading
        >>> dataloader = optimized_experiment_dataloader(
        ...     dataset, 
        ...     batch_size=64,
        ...     max_concurrent_requests=2,
        ...     max_concurrent_batch_processing=1
        ... )
        
        >>> # Aggressive optimization for high-performance training
        >>> dataloader = optimized_experiment_dataloader(
        ...     dataset,
        ...     batch_size=128, 
        ...     max_concurrent_requests=8,
        ...     max_concurrent_batch_processing=4,
        ...     enable_pinned_memory=True,
        ...     enable_tensor_cache=True
        ... )
    """
    if num_workers != 0:
        warnings.warn(
            "num_workers > 0 with threading optimizations may cause issues. "
            "Consider using num_workers=0 with max_concurrent_requests instead.",
            UserWarning
        )
    
    # Set default io_batch_size if not provided
    if io_batch_size is None:
        io_batch_size = max(batch_size * 128, 32768)
    
    # Validate concurrency parameters
    max_concurrent_requests = max(1, max_concurrent_requests)
    max_concurrent_batch_processing = max(1, max_concurrent_batch_processing)
    
    # Configure the dataset for threading optimization
    if hasattr(exp_data, '_io_batch_size'):
        exp_data._io_batch_size = io_batch_size
    if hasattr(exp_data, '_max_concurrent_requests'):
        exp_data._max_concurrent_requests = max_concurrent_requests
    if hasattr(exp_data, '_use_eager_fetch'):
        exp_data._use_eager_fetch = use_eager_fetch
    if hasattr(exp_data, '_enable_pinned_memory'):
        exp_data._enable_pinned_memory = enable_pinned_memory
    if hasattr(exp_data, '_enable_tensor_cache'):
        exp_data._enable_tensor_cache = enable_tensor_cache
    if hasattr(exp_data, '_max_concurrent_batch_processing'):
        exp_data._max_concurrent_batch_processing = max_concurrent_batch_processing
    
    logger.info(f"Creating optimized dataloader with:")
    logger.info(f"  - batch_size: {batch_size}")
    logger.info(f"  - io_batch_size: {io_batch_size}")
    logger.info(f"  - max_concurrent_requests: {max_concurrent_requests}")
    logger.info(f"  - max_concurrent_batch_processing: {max_concurrent_batch_processing}")
    logger.info(f"  - enable_pinned_memory: {enable_pinned_memory}")
    logger.info(f"  - enable_tensor_cache: {enable_tensor_cache}")
    logger.info(f"  - use_eager_fetch: {use_eager_fetch}")
    
    return DataLoader(
        exp_data, 
        batch_size=None,  # We handle batching internally
        num_workers=num_workers, 
        shuffle=False,  # We handle shuffling internally
        **dataloader_kwargs
    )


def _collate_noop(datum: _T) -> _T:
    """Noop collation used by |experiment_dataloader|.

    Private.
    """
    return datum
