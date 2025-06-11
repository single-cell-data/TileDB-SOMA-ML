# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import Iterable, Iterator, Optional
from queue import Queue
import threading

import attrs
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from tiledbsoma_ml._common import MiniBatch, NDArrayNumber
from tiledbsoma_ml._csr import CSR_IO_Buffer
from tiledbsoma_ml._eager_iter import EagerIterator
from tiledbsoma_ml._io_batch_iterable import IOBatch, IOBatchIterable

logger = logging.getLogger("tiledbsoma_ml._mini_batch_iterable")

# Thread-local tensor cache for reusing memory allocations
_thread_local_cache = threading.local()

def _get_tensor_cache():
    """Get thread-local tensor cache for reusing tensor allocations."""
    if not hasattr(_thread_local_cache, 'tensor_cache'):
        _thread_local_cache.tensor_cache = {}
    return _thread_local_cache.tensor_cache


@attrs.define(frozen=True)
class MiniBatchIterable:
    """Iterate over mini-batches from IO batches, with GPU optimization support."""

    io_batch_iter: IOBatchIterable
    batch_size: int
    use_eager_fetch: bool = True
    return_sparse_X: bool = False
    
    # GPU optimization parameters
    pin_memory: bool = True
    prefetch_factor: int = 2
    use_cuda_streams: bool = True
    tensor_cache_size: int = 4

    def __iter__(self) -> Iterator[MiniBatch]:
        """Emit mini-batches with GPU optimizations."""
        if self.use_eager_fetch:
            io_batch_iter = EagerIterator(iter(self.io_batch_iter))
        else:
            io_batch_iter = iter(self.io_batch_iter)

        # Set up CUDA streams if available and requested
        cuda_available = torch.cuda.is_available()
        stream = None
        if cuda_available and self.use_cuda_streams:
            try:
                stream = torch.cuda.Stream()
            except Exception as e:
                logger.warning(f"Could not create CUDA stream: {e}")
                stream = None

        # Prefetching queue for GPU optimization
        prefetch_queue = Queue(maxsize=self.prefetch_factor) if self.prefetch_factor > 1 else None
        prefetch_thread = None

        if prefetch_queue:
            prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                args=(io_batch_iter, prefetch_queue, stream),
                daemon=True
            )
            prefetch_thread.start()
            batch_source = self._get_from_queue(prefetch_queue)
        else:
            batch_source = io_batch_iter

        try:
            for X_io_batch, obs_io_batch in batch_source:
                yield from self._process_io_batch_optimized(
                    X_io_batch, obs_io_batch, stream
                )
        finally:
            if prefetch_thread:
                prefetch_queue.put(None)  # Signal thread to stop
                prefetch_thread.join(timeout=1.0)

    def _prefetch_worker(self, io_batch_iter: Iterator[IOBatch], queue: Queue, stream):
        """Worker thread for prefetching batches."""
        try:
            for batch in io_batch_iter:
                queue.put(batch)
            queue.put(None)  # Signal end of batches
        except Exception as e:
            logger.error(f"Error in prefetch worker: {e}")
            queue.put(None)

    def _get_from_queue(self, queue: Queue) -> Iterator[IOBatch]:
        """Get batches from prefetch queue."""
        while True:
            batch = queue.get()
            if batch is None:
                break
            yield batch

    def _process_io_batch_optimized(
        self, 
        X_io_batch: CSR_IO_Buffer, 
        obs_io_batch, 
        stream
    ) -> Iterator[MiniBatch]:
        """Process IO batch with GPU optimizations."""
        total_rows = X_io_batch.shape[0]
        
        for start_row in range(0, total_rows, self.batch_size):
            end_row = min(start_row + self.batch_size, total_rows)
            batch_slice = slice(start_row, end_row)
            
            # Extract mini-batch data with optimized memory management
            if self.return_sparse_X:
                X_batch = self._extract_sparse_batch_optimized(X_io_batch, batch_slice, stream)
            else:
                X_batch = self._extract_dense_batch_optimized(X_io_batch, batch_slice, stream)
            
            obs_batch = obs_io_batch.iloc[batch_slice].copy()
            
            yield X_batch, obs_batch

    def _extract_dense_batch_optimized(
        self, 
        X_io_batch: CSR_IO_Buffer, 
        batch_slice: slice,
        stream
    ) -> NDArrayNumber:
        """Extract dense batch with GPU optimizations and tensor caching."""
        # Get tensor cache
        tensor_cache = _get_tensor_cache()
        
        # Calculate batch dimensions
        n_rows = batch_slice.stop - batch_slice.start
        n_cols = X_io_batch.shape[1]
        batch_shape = (n_rows, n_cols)
        dtype = X_io_batch.dtype
        
        # Try to reuse cached tensor
        cache_key = f"dense_{batch_shape}_{dtype}"
        
        if cache_key in tensor_cache and len(tensor_cache) <= self.tensor_cache_size:
            cached_array = tensor_cache[cache_key]
            if cached_array.shape == batch_shape and cached_array.dtype == dtype:
                # Reuse cached array
                X_batch = cached_array
                X_batch.fill(0)  # Reset values
            else:
                # Create new array
                X_batch = self._create_optimized_array(batch_shape, dtype)
                tensor_cache[cache_key] = X_batch.copy()
        else:
            # Create new array and cache it
            X_batch = self._create_optimized_array(batch_shape, dtype)
            if len(tensor_cache) < self.tensor_cache_size:
                tensor_cache[cache_key] = X_batch.copy()

        # Fill the array using optimized CSR extraction
        try:
            if stream and torch.cuda.is_available():
                with torch.cuda.stream(stream):
                    X_io_batch.slice_tonumpy_inplace(batch_slice, X_batch)
            else:
                # Use the optimized in-place slice method
                result = X_io_batch.slice_tonumpy(batch_slice)
                X_batch[:] = result
        except AttributeError:
            # Fallback to original method
            X_batch = X_io_batch.slice_tonumpy(batch_slice)

        return X_batch

    def _extract_sparse_batch_optimized(
        self, 
        X_io_batch: CSR_IO_Buffer, 
        batch_slice: slice,
        stream
    ) -> sparse.csr_matrix:
        """Extract sparse batch with optimizations."""
        return X_io_batch.slice_toscipy(batch_slice)

    def _create_optimized_array(self, shape, dtype) -> NDArrayNumber:
        """Create optimized numpy array with optional pinned memory."""
        if self.pin_memory and torch.cuda.is_available():
            try:
                # Create pinned memory tensor and convert to numpy
                tensor = torch.zeros(shape, dtype=self._numpy_to_torch_dtype(dtype), pin_memory=True)
                return tensor.numpy()
            except Exception as e:
                logger.warning(f"Could not create pinned memory array: {e}")
                
        # Fallback to regular numpy array
        return np.zeros(shape, dtype=dtype)

    def _numpy_to_torch_dtype(self, numpy_dtype):
        """Convert numpy dtype to torch dtype."""
        dtype_mapping = {
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.bool_: torch.bool,
        }
        return dtype_mapping.get(numpy_dtype, torch.float32)
