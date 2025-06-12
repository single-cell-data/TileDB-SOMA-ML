# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.

import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional, Sequence, Tuple

import attrs
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from ._csr import CSR_IO_Buffer
from ._io_batch_iterable import IOBatch

logger = logging.getLogger("tiledbsoma_ml._mini_batch_iterable")


def _check_and_reopen_soma_object(soma_obj, original_uri: str):
    """Check if a SOMA object is open and reopen it if necessary (same as in IOBatchIterable)."""
    try:
        # Try to access a basic property to check if it's open
        _ = soma_obj.schema
        return soma_obj  # Object is open and accessible
    except Exception:
        # Object is closed, need to reopen it
        logger.debug(f"Reopening closed SOMA object: {original_uri}")
        if hasattr(soma_obj, 'open'):
            # For SparseNDArray and DataFrame
            return type(soma_obj).open(original_uri, mode="r", context=soma_obj.context)
        else:
            raise RuntimeError(f"Don't know how to reopen SOMA object of type {type(soma_obj)}")


# Thread-local storage for caching tensors to avoid repeated allocations
_thread_local = threading.local()


def _get_thread_local_cache():
    """Get or create thread-local tensor cache."""
    if not hasattr(_thread_local, 'tensor_cache'):
        _thread_local.tensor_cache = {}
        _thread_local.pinned_memory_pool = {}
    return _thread_local.tensor_cache, _thread_local.pinned_memory_pool


@attrs.define(frozen=True)
class MiniBatchIterable:
    """An ``Iterable`` of mini-batches, each mini-batch is a ``Dict[str, Union[np.ndarray, torch.Tensor]]``
    containing PyTorch Tensors or NumPy arrays (depending on ``return_sparse_X``).

    The ``X`` key refers to a dense |np.ndarray| containing X data. The remainder of keys refer to
    |pandas.DataFrame| columns in ``obs``, and contain 1D |np.ndarray|'s.

    Example::

        mini_batch_iterable = MiniBatchIterable(...)
        mini_batch = next(mini_batch_iterable)
        print(mini_batch)
        # {'X': np.ndarray(...), 'label': np.ndarray(...), ...}

    """

    io_batch_iter: Iterator[IOBatch]
    encoders: Sequence[any]
    mini_batch_size: int
    return_sparse_X: bool = False
    use_eager_fetch: bool = True
    shuffle: bool = True
    seed: Optional[int] = None
    soma_joinid: bool = False
    # GPU optimization settings
    enable_pinned_memory: bool = False
    enable_tensor_cache: bool = True
    max_concurrent_batch_processing: int = 2

    def __iter__(self) -> Iterator:
        """Iterate over mini-batches with optimized concurrent processing."""
        mini_batch_rng = np.random.default_rng(self.seed)
        
        # Process IOBatches and yield mini-batches
        for io_batch in self.io_batch_iter:
            # Process each IO batch into mini-batches
            yield from self._process_io_batch_into_mini_batches(io_batch, mini_batch_rng)

    def _process_io_batch_into_mini_batches(self, io_batch: IOBatch, mini_batch_rng: np.random.Generator) -> Iterator:
        """Process a single IO batch into multiple mini-batches with optimizations."""
        X_io_batch, obs_io_batch = io_batch
        
        if self.max_concurrent_batch_processing > 1 and X_io_batch.shape[0] > self.mini_batch_size * 2:
            # Use concurrent processing for large IO batches
            yield from self._process_io_batch_concurrent(X_io_batch, obs_io_batch, mini_batch_rng)
        else:
            # Use sequential processing for smaller batches
            yield from self._process_io_batch_sequential(X_io_batch, obs_io_batch, mini_batch_rng)

    def _process_io_batch_concurrent(
        self, X_io_batch: CSR_IO_Buffer, obs_io_batch: pd.DataFrame, mini_batch_rng: np.random.Generator
    ) -> Iterator:
        """Process IO batch into mini-batches using concurrent processing."""
        
        # Convert to dense for easier processing
        X_dense = X_io_batch.to_scipy_sparse_matrix().toarray()
        
        # Calculate mini-batch indices
        num_samples = X_dense.shape[0]
        indices = np.arange(num_samples)
        if self.shuffle:
            mini_batch_rng.shuffle(indices)
        
        # Split into mini-batch chunks
        mini_batch_indices_list = [
            indices[i:i + self.mini_batch_size] 
            for i in range(0, len(indices), self.mini_batch_size)
        ]
        
        if len(mini_batch_indices_list) <= 1:
            # Fall back to sequential for single mini-batch
            yield from self._process_io_batch_sequential(X_io_batch, obs_io_batch, mini_batch_rng)
            return
        
        def process_mini_batch_chunk(mini_batch_indices: np.ndarray) -> dict:
            """Process a chunk of data into a mini-batch in a thread-safe way."""
            cache, pinned_pool = _get_thread_local_cache()
            
            # Extract data for this mini-batch
            X_mini = X_dense[mini_batch_indices]
            obs_mini = obs_io_batch.iloc[mini_batch_indices]
            
            # Create mini-batch dictionary
            mini_batch = self._create_mini_batch_dict(X_mini, obs_mini, cache, pinned_pool)
            
            return mini_batch
        
        # Use ThreadPoolExecutor for concurrent mini-batch processing
        with ThreadPoolExecutor(max_workers=self.max_concurrent_batch_processing) as executor:
            # Submit all mini-batch processing jobs
            futures = [
                executor.submit(process_mini_batch_chunk, mini_batch_indices)
                for mini_batch_indices in mini_batch_indices_list
            ]
            
            # Yield results as they complete
            for future in futures:  # Maintain order
                try:
                    yield future.result()
                except Exception as e:
                    logger.error(f"Error processing mini-batch: {e}")
                    raise

    def _process_io_batch_sequential(
        self, X_io_batch: CSR_IO_Buffer, obs_io_batch: pd.DataFrame, mini_batch_rng: np.random.Generator
    ) -> Iterator:
        """Process IO batch into mini-batches sequentially (fallback method)."""
        # Convert to dense
        X_dense = X_io_batch.to_scipy_sparse_matrix().toarray()
        
        # Shuffle if requested
        num_samples = X_dense.shape[0]
        indices = np.arange(num_samples)
        if self.shuffle:
            mini_batch_rng.shuffle(indices)
        
        cache, pinned_pool = _get_thread_local_cache()
        
        # Generate mini-batches
        for i in range(0, len(indices), self.mini_batch_size):
            mini_batch_indices = indices[i:i + self.mini_batch_size]
            
            X_mini = X_dense[mini_batch_indices]
            obs_mini = obs_io_batch.iloc[mini_batch_indices]
            
            yield self._create_mini_batch_dict(X_mini, obs_mini, cache, pinned_pool)

    def _create_mini_batch_dict(
        self, X_mini: np.ndarray, obs_mini: pd.DataFrame, cache: dict, pinned_pool: dict
    ) -> dict:
        """Create a mini-batch dictionary with optimizations."""
        mini_batch = {}
        
        # Handle X data
        if self.return_sparse_X:
            # Convert to sparse if requested
            X_sparse = sparse.csr_matrix(X_mini)
            mini_batch["X"] = X_sparse
        else:
            # Convert to PyTorch tensor with optimizations
            if self.enable_tensor_cache:
                # Try to reuse tensor if same shape
                tensor_key = f"X_{X_mini.shape}"
                if tensor_key in cache and cache[tensor_key].shape == X_mini.shape:
                    X_tensor = cache[tensor_key]
                    X_tensor.copy_(torch.from_numpy(X_mini))
                else:
                    X_tensor = torch.from_numpy(X_mini.copy()).float()
                    if self.enable_pinned_memory and torch.cuda.is_available():
                        X_tensor = X_tensor.pin_memory()
                    cache[tensor_key] = X_tensor
            else:
                X_tensor = torch.from_numpy(X_mini.copy()).float()
                if self.enable_pinned_memory and torch.cuda.is_available():
                    X_tensor = X_tensor.pin_memory()
            
            mini_batch["X"] = X_tensor

        # Handle obs data with encoders
        for encoder in self.encoders:
            column_name = encoder.name
            if column_name in obs_mini.columns:
                encoded_values = encoder.transform(obs_mini[column_name])
                
                # Convert to tensor if needed
                if not self.return_sparse_X and not isinstance(encoded_values, (sparse.spmatrix, sparse._base.spbase)):
                    if self.enable_tensor_cache:
                        tensor_key = f"{column_name}_{encoded_values.shape}"
                        if tensor_key in cache and cache[tensor_key].shape == encoded_values.shape:
                            values_tensor = cache[tensor_key]
                            if encoded_values.dtype == np.int64:
                                values_tensor.copy_(torch.from_numpy(encoded_values))
                            else:
                                values_tensor.copy_(torch.from_numpy(encoded_values.astype(np.float32)))
                        else:
                            if encoded_values.dtype == np.int64:
                                values_tensor = torch.from_numpy(encoded_values.copy())
                            else:
                                values_tensor = torch.from_numpy(encoded_values.astype(np.float32))
                            
                            if self.enable_pinned_memory and torch.cuda.is_available():
                                values_tensor = values_tensor.pin_memory()
                            cache[tensor_key] = values_tensor
                    else:
                        if encoded_values.dtype == np.int64:
                            values_tensor = torch.from_numpy(encoded_values.copy())
                        else:
                            values_tensor = torch.from_numpy(encoded_values.astype(np.float32))
                        
                        if self.enable_pinned_memory and torch.cuda.is_available():
                            values_tensor = values_tensor.pin_memory()
                    
                    mini_batch[column_name] = values_tensor
                else:
                    mini_batch[column_name] = encoded_values

        # Add soma_joinid if requested
        if self.soma_joinid:
            soma_joinids = obs_mini["soma_joinid"].values
            if not self.return_sparse_X:
                if self.enable_tensor_cache:
                    tensor_key = f"soma_joinid_{soma_joinids.shape}"
                    if tensor_key in cache and cache[tensor_key].shape == soma_joinids.shape:
                        joinid_tensor = cache[tensor_key]
                        joinid_tensor.copy_(torch.from_numpy(soma_joinids))
                    else:
                        joinid_tensor = torch.from_numpy(soma_joinids.copy())
                        if self.enable_pinned_memory and torch.cuda.is_available():
                            joinid_tensor = joinid_tensor.pin_memory()
                        cache[tensor_key] = joinid_tensor
                else:
                    joinid_tensor = torch.from_numpy(soma_joinids.copy())
                    if self.enable_pinned_memory and torch.cuda.is_available():
                        joinid_tensor = joinid_tensor.pin_memory()
                
                mini_batch["soma_joinid"] = joinid_tensor
            else:
                mini_batch["soma_joinid"] = soma_joinids
        
        # Cleanup
        gc.collect(generation=0)
        
        return mini_batch
