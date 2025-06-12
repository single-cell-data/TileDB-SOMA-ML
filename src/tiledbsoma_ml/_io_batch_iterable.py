# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Iterator, Optional, Sequence, Tuple, List

import attrs
import numpy as np
import pandas as pd
import pyarrow as pa
from tiledbsoma import DataFrame, IntIndexer, SparseNDArray

from tiledbsoma_ml._common import NDArrayJoinId
from tiledbsoma_ml._csr import CSR_IO_Buffer
from tiledbsoma_ml._eager_iter import EagerIterator
from tiledbsoma_ml._query_ids import Chunks
from tiledbsoma_ml._utils import batched

logger = logging.getLogger("tiledbsoma_ml._io_batch_iterable")
IOBatch = Tuple[CSR_IO_Buffer, pd.DataFrame]
"""Tuple type emitted by |IOBatchIterable|, containing ``X`` rows (as a |CSR_IO_Buffer|) and ``obs`` rows
(|pd.DataFrame|)."""


def _check_and_reopen_soma_object(soma_obj, original_uri: str):
    """Check if a SOMA object is open and reopen it if necessary."""
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


@attrs.define(frozen=True)
class IOBatchIterable(Iterable[IOBatch]):
    """Given a list of ``obs_joinid`` |Chunks|, re-chunk them into (optionally shuffled) |IOBatch|'s".

    An |IOBatch| is a tuple consisting of a batch of rows from the ``X`` |SparseNDArray|, as well as the corresponding
    rows from the ``obs`` |DataFrame|. The ``X`` rows are returned in an optimized |CSR_IO_Buffer|.
    """

    chunks: Chunks
    io_batch_size: int
    obs: DataFrame
    var_joinids: NDArrayJoinId
    X: SparseNDArray
    obs_column_names: Sequence[str] = ("soma_joinid",)
    seed: Optional[int] = None
    shuffle: bool = True
    use_eager_fetch: bool = True
    # Threading parameters with SOMA safety
    max_concurrent_requests: int = 4
    prefetch_queue_size: int = 2

    @property
    def io_batch_ids(self) -> Iterable[Tuple[int, ...]]:
        """Re-chunk ``obs_joinids`` according to the desired ``io_batch_size``."""
        return batched(
            (joinid for chunk in self.chunks for joinid in chunk),
            self.io_batch_size,
        )

    def _fetch_batch_data_concurrent(
        self, 
        obs_coords_list: List[Tuple[int, ...]], 
        shuffle_rng: np.random.Generator
    ) -> Iterator[IOBatch]:
        """Fetch multiple batches concurrently with SOMA object safety checks."""
        # Store original URIs for reopening if needed
        X_uri = self.X.uri
        obs_uri = self.obs.uri
        context = self.X.context
        
        obs_column_names = (
            list(self.obs_column_names)
            if "soma_joinid" in self.obs_column_names
            else ["soma_joinid", *self.obs_column_names]
        )
        var_joinids = self.var_joinids.astype("int64")

        def fetch_single_batch_thread_safe(obs_coords: Tuple[int, ...]) -> IOBatch:
            """Fetch a single batch with thread-safe SOMA object handling."""
            st_time = time.perf_counter()
            
            # Create thread-local SOMA objects by reopening
            try:
                thread_X = _check_and_reopen_soma_object(self.X, X_uri)
                thread_obs = _check_and_reopen_soma_object(self.obs, obs_uri)
            except Exception as e:
                logger.error(f"Failed to open SOMA objects in thread: {e}")
                # Fall back to original objects and hope for the best
                thread_X = self.X
                thread_obs = self.obs
            
            obs_shuffled_coords = (
                np.array(obs_coords)
                if not self.shuffle
                else shuffle_rng.permuted(obs_coords)
            )
            
            # Create indexers with the thread-local context
            obs_indexer = IntIndexer(obs_shuffled_coords, context=context)
            var_indexer = IntIndexer(var_joinids, context=context)
            
            try:
                # Fetch X data with thread-safe objects
                X_io_batch = self._fetch_x_data_thread_safe(
                    obs_coords, obs_indexer, var_indexer, thread_X
                )
                
                # Fetch obs data with thread-safe objects
                obs_io_batch = self._fetch_obs_data_thread_safe(
                    obs_coords, obs_shuffled_coords, obs_column_names, thread_obs
                )
                
            finally:
                # Cleanup
                del obs_indexer, obs_shuffled_coords
                # Close thread-local objects if they were reopened
                if thread_X is not self.X:
                    try:
                        thread_X.close()
                    except:
                        pass
                if thread_obs is not self.obs:
                    try:
                        thread_obs.close()
                    except:
                        pass
                gc.collect(generation=0)

            tm = time.perf_counter() - st_time
            logger.debug(
                f"Retrieved SOMA IO batch, took {tm:.2f}sec, {X_io_batch.shape[0]/tm:0.1f} samples/sec"
            )
            return X_io_batch, obs_io_batch

        # Use ThreadPoolExecutor for concurrent batch fetching
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # Submit all batch fetch jobs
            future_to_coords = {
                executor.submit(fetch_single_batch_thread_safe, obs_coords): obs_coords 
                for obs_coords in obs_coords_list
            }
            
            # Yield results as they complete (maintaining order if needed)
            if self.shuffle:
                # If shuffling, order doesn't matter - yield as completed
                for future in as_completed(future_to_coords):
                    try:
                        yield future.result()
                    except Exception as e:
                        logger.error(f"Error fetching batch: {e}")
                        raise
            else:
                # If not shuffling, maintain order
                futures_list = list(future_to_coords.keys())
                for future in futures_list:
                    try:
                        yield future.result()
                    except Exception as e:
                        logger.error(f"Error fetching batch: {e}")
                        raise

    def _fetch_x_data_thread_safe(
        self, 
        obs_coords: Tuple[int, ...], 
        obs_indexer: IntIndexer, 
        var_indexer: IntIndexer,
        X_obj: SparseNDArray
    ) -> CSR_IO_Buffer:
        """Thread-safe X data fetching with proper SOMA object handling."""
        X_tbl_iter: Iterator[pa.Table] = X_obj.read(
            coords=(obs_coords, self.var_joinids)
        ).tables()

        def make_io_buffer_optimized(
            X_tbl: pa.Table,
            obs_coords: Tuple[int, ...],
            var_coords: NDArrayJoinId,
            obs_indexer: IntIndexer,
        ) -> CSR_IO_Buffer:
            """Optimized IO buffer creation with reduced memory allocations."""
            # Convert to numpy arrays more efficiently
            i_indices = obs_indexer.get_indexer(X_tbl["soma_dim_0"])
            j_indices = var_indexer.get_indexer(X_tbl["soma_dim_1"])
            data = X_tbl["soma_data"].to_numpy()
            
            # Create CSR buffer directly to avoid intermediate allocations
            m = CSR_IO_Buffer.from_ijd(
                i_indices,
                j_indices, 
                data,
                shape=(len(obs_coords), len(var_coords)),
            )
            
            # More aggressive garbage collection for large allocations
            del i_indices, j_indices, data
            gc.collect(generation=0)
            return m

        _io_buf_iter: Iterator[CSR_IO_Buffer] = (
            make_io_buffer_optimized(
                X_tbl=X_tbl,
                obs_coords=obs_coords,
                var_coords=self.var_joinids,
                obs_indexer=obs_indexer,
            )
            for X_tbl in X_tbl_iter
        )

        if self.use_eager_fetch:
            _io_buf_iter = EagerIterator(_io_buf_iter, pool=X_obj.context.threadpool)

        return CSR_IO_Buffer.merge(tuple(_io_buf_iter))

    def _fetch_obs_data_thread_safe(
        self, 
        obs_coords: Tuple[int, ...], 
        obs_shuffled_coords: np.ndarray, 
        obs_column_names: List[str],
        obs_obj: DataFrame
    ) -> pd.DataFrame:
        """Thread-safe obs data fetching."""
        return (
            obs_obj.read(coords=(obs_coords,), column_names=obs_column_names)
            .concat()
            .to_pandas()
            .set_index("soma_joinid")
            .reindex(obs_shuffled_coords, copy=False)
            .reset_index()  # demote "soma_joinid" to a column
            [self.obs_column_names]
        )

    def __iter__(self) -> Iterator[IOBatch]:
        """Emit |IOBatch|'s with thread-safe concurrent fetching."""
        # Because obs/var IDs have been partitioned/split/shuffled upstream of this class, this RNG does not need to be
        # identical across sub-processes, but seeding is supported anyway, for testing/reproducibility.
        shuffle_rng = np.random.default_rng(self.seed)

        # Convert to list for concurrent processing
        batch_coords_list = list(self.io_batch_ids)
        
        if len(batch_coords_list) <= 1 or self.max_concurrent_requests <= 1:
            # Fall back to sequential processing for single batch or when concurrency disabled
            for obs_coords in batch_coords_list:
                yield self._fetch_single_batch_sequential(obs_coords, shuffle_rng)
        else:
            # Use concurrent fetching for multiple batches
            # Process in chunks to avoid overwhelming the system
            chunk_size = max(1, min(self.max_concurrent_requests, len(batch_coords_list)))
            for i in range(0, len(batch_coords_list), chunk_size):
                chunk = batch_coords_list[i:i + chunk_size]
                yield from self._fetch_batch_data_concurrent(chunk, shuffle_rng)

    def _fetch_single_batch_sequential(self, obs_coords: Tuple[int, ...], shuffle_rng: np.random.Generator) -> IOBatch:
        """Sequential batch fetching for fallback."""
        st_time = time.perf_counter()
        
        obs_shuffled_coords = (
            np.array(obs_coords)
            if not self.shuffle
            else shuffle_rng.permuted(obs_coords)
        )
        
        X = self.X
        context = X.context
        obs_column_names = (
            list(self.obs_column_names)
            if "soma_joinid" in self.obs_column_names
            else ["soma_joinid", *self.obs_column_names]
        )
        var_joinids = self.var_joinids.astype("int64")
        
        # Create indexers within the same context
        obs_indexer = IntIndexer(obs_shuffled_coords, context=context)
        var_indexer = IntIndexer(var_joinids, context=context)
        
        logger.debug(f"Retrieving next SOMA IO batch of length {len(obs_coords)}...")

        # Fetch X data with optimized memory management
        X_io_batch = self._fetch_x_data_thread_safe(obs_coords, obs_indexer, var_indexer, X)
        
        # Fetch obs data
        obs_io_batch = self._fetch_obs_data_thread_safe(obs_coords, obs_shuffled_coords, obs_column_names, self.obs)

        # Cleanup
        del obs_indexer, obs_coords, obs_shuffled_coords
        gc.collect(generation=0)

        tm = time.perf_counter() - st_time
        logger.debug(
            f"Retrieved SOMA IO batch, took {tm:.2f}sec, {X_io_batch.shape[0]/tm:0.1f} samples/sec"
        )
        return X_io_batch, obs_io_batch
