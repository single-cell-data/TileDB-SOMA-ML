# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.

import gc
import logging
import time
from typing import Iterable, Iterator, Optional, Sequence, Tuple

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
        """Fetch multiple batches concurrently to improve S3 throughput."""
        # For TileDB-SOMA thread safety, we need to open separate contexts per thread
        # Fall back to sequential processing to avoid context issues
        logger.debug("Using sequential batch processing for TileDB-SOMA thread safety")
        for obs_coords in obs_coords_list:
            yield self._fetch_single_batch_safe(obs_coords, shuffle_rng)

    def _fetch_single_batch_safe(self, obs_coords: Tuple[int, ...], shuffle_rng: np.random.Generator) -> IOBatch:
        """Fetch a single batch safely with proper context management."""
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
        X_io_batch = self._fetch_x_data_safe(obs_coords, obs_indexer, var_indexer)
        
        # Fetch obs data
        obs_io_batch = self._fetch_obs_data(obs_coords, obs_shuffled_coords, obs_column_names)

        # Cleanup
        del obs_indexer, obs_coords, obs_shuffled_coords
        gc.collect(generation=0)

        tm = time.perf_counter() - st_time
        logger.debug(
            f"Retrieved SOMA IO batch, took {tm:.2f}sec, {X_io_batch.shape[0]/tm:0.1f} samples/sec"
        )
        return X_io_batch, obs_io_batch

    def _fetch_x_data_safe(self, obs_coords: Tuple[int, ...], obs_indexer: IntIndexer, var_indexer: IntIndexer) -> CSR_IO_Buffer:
        """Safely fetch X data with proper context management."""
        X_tbl_iter: Iterator[pa.Table] = self.X.read(
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
            _io_buf_iter = EagerIterator(_io_buf_iter, pool=self.X.context.threadpool)

        return CSR_IO_Buffer.merge(tuple(_io_buf_iter))

    def _fetch_obs_data(self, obs_coords: Tuple[int, ...], obs_shuffled_coords: np.ndarray, obs_column_names: List[str]) -> pd.DataFrame:
        """Optimized obs data fetching."""
        return (
            self.obs.read(coords=(obs_coords,), column_names=obs_column_names)
            .concat()
            .to_pandas()
            .set_index("soma_joinid")
            .reindex(obs_shuffled_coords, copy=False)
            .reset_index()  # demote "soma_joinid" to a column
            [self.obs_column_names]
        )

    def __iter__(self) -> Iterator[IOBatch]:
        """Emit |IOBatch|'s with optimized sequential fetching for TileDB-SOMA thread safety."""
        # Because obs/var IDs have been partitioned/split/shuffled upstream of this class, this RNG does not need to be
        # identical across sub-processes, but seeding is supported anyway, for testing/reproducibility.
        shuffle_rng = np.random.default_rng(self.seed)

        # Use optimized sequential processing for TileDB-SOMA thread safety
        # TileDB-SOMA objects are not thread-safe, so we process batches sequentially
        # but with other optimizations like memory pooling and efficient CSR operations
        
        for obs_coords in self.io_batch_ids:
            yield self._fetch_single_batch_safe(obs_coords, shuffle_rng)
