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
from tiledbsoma import Experiment

from tiledbsoma_ml._common import NDArrayJoinId
from tiledbsoma_ml._csr import CSR_IO_Buffer
from tiledbsoma_ml._eager_iter import EagerIterator
from tiledbsoma_ml._query_ids import Chunks
from tiledbsoma_ml._utils import batched
import os
import psutil
import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

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
    use_eager_fetch: int = 1

    @property
    def io_batch_ids(self) -> Iterable[Tuple[int, ...]]:
        """Re-chunk ``obs_joinids`` according to the desired ``io_batch_size``."""
        return batched(
            (joinid for chunk in self.chunks for joinid in chunk),
            self.io_batch_size,
        )

    def __iter__(self) -> Iterator[IOBatch]:
        """Emit |IOBatch|'s."""
        # Because obs/var IDs have been partitioned/split/shuffled upstream of this class, this RNG does not need to be
        # identical across sub-processes, but seeding is supported anyway, for testing/reproducibility.
        shuffle_rng = np.random.default_rng(self.seed)
        X = self.X
        context = X.context
        obs_column_names = (
            list(self.obs_column_names)
            if "soma_joinid" in self.obs_column_names
            else ["soma_joinid", *self.obs_column_names]
        )
        # NOTE: `.astype("int64")` works around the `np.int64` singleton failing reference-equality after cross-process
        # SerDes.
        var_joinids = self.var_joinids.astype("int64")
        var_indexer = IntIndexer(var_joinids, context=context)

        for obs_coords in self.io_batch_ids:
            st_time = time.perf_counter()
            obs_shuffled_coords = (
                np.array(obs_coords)
                if not self.shuffle
                else shuffle_rng.permuted(obs_coords)
            )
            obs_indexer = IntIndexer(obs_shuffled_coords, context=context)
            logger.debug(
                f"Retrieving next SOMA IO batch of length {len(obs_coords)}..."
            )

            # To maximize opportunities for concurrency, when in eager_fetch mode,
            # create the X read iterator first, as the eager iterator will begin
            # the read-ahead immediately. Then proceed to fetch obs DataFrame.
            # This matters most on latent backing stores, e.g., S3.
            X_tbl_iter: Iterator[pa.Table] = X.read(
                coords=(obs_coords, self.var_joinids)
            ).tables()

            def make_io_buffer(
                X_tbl: pa.Table,
                obs_coords: NDArrayJoinId,
                var_coords: NDArrayJoinId,
                obs_indexer: IntIndexer,
            ) -> CSR_IO_Buffer:
                """This function provides a GC after we throw off (large) garbage."""
                m = CSR_IO_Buffer.from_ijd(
                    obs_indexer.get_indexer(X_tbl["soma_dim_0"]),
                    var_indexer.get_indexer(X_tbl["soma_dim_1"]),
                    X_tbl["soma_data"].to_numpy(),
                    shape=(len(obs_coords), len(var_coords)),
                )
                gc.collect(generation=0)
                return m

            _io_buf_iter: Iterator[CSR_IO_Buffer] = (
                make_io_buffer(
                    X_tbl=X_tbl,
                    obs_coords=np.array(obs_coords),
                    var_coords=self.var_joinids,
                    obs_indexer=obs_indexer,
                )
                for X_tbl in X_tbl_iter
            )
            if self.use_eager_fetch:
                _io_buf_iter = EagerIterator(_io_buf_iter, pool=X.context.threadpool)

            # Now that X read is potentially in progress (in eager mode), go fetch obs data
            # fmt: off
            obs_io_batch = (
                self.obs.read(coords=(obs_coords,), column_names=obs_column_names)
                .concat()
                .to_pandas()
                .set_index("soma_joinid")
                .reindex(obs_shuffled_coords, copy=False)
                .reset_index()  # demote "soma_joinid" to a column
                [self.obs_column_names]
            )  # fmt: on

            X_io_batch = CSR_IO_Buffer.merge(tuple(_io_buf_iter))

            del obs_indexer, obs_coords, obs_shuffled_coords, _io_buf_iter
            gc.collect()

            tm = time.perf_counter() - st_time
            logger.debug(
                f"Retrieved SOMA IO batch, took {tm:.2f}sec, {X_io_batch.shape[0]/tm:0.1f} samples/sec"
            )
            yield X_io_batch, obs_io_batch


class TrainingMetrics:
    def __init__(self):
        self.batch_times = []
        self.data_load_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.losses = []
    
    def update(self, batch_time, data_load_time, loss):
        self.batch_times.append(batch_time)
        self.data_load_times.append(data_load_time)
        self.losses.append(loss)
        
        if torch.cuda.is_available():
            self.gpu_memory_usage.append(torch.cuda.memory_allocated())
        self.cpu_memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
    
    def get_stats(self):
        return {
            'avg_batch_time': np.mean(self.batch_times),
            'avg_data_load_time': np.mean(self.data_load_times),
            'max_gpu_memory': max(self.gpu_memory_usage) / 1e9 if self.gpu_memory_usage else 0,  # GB
            'max_cpu_memory': max(self.cpu_memory_usage),  # MB
            'avg_loss': np.mean(self.losses)
        }

def _io_batch_iterable(
    X: Experiment,
    batch_size: int,
    shuffle: bool,
    seed: int,
    use_eager_fetch: int,
    layer_name: str,
    dataloader_kwargs: dict,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Create an iterable that yields batches of data from a SOMAExperiment.

    Args:
        X: The SOMAExperiment to iterate over.
        batch_size: The number of cells to include in each batch.
        shuffle: Whether to shuffle the data before iterating.
        seed: The random seed to use for shuffling.
        use_eager_fetch: Number of batches to prefetch.
        layer_name: The name of the layer to use.
        dataloader_kwargs: Additional keyword arguments to pass to the DataLoader.

    Returns:
        An iterator that yields tuples of (data, labels) tensors.
    """
    # Get the measurement and layer from the experiment
    measurement = X.ms[layer_name]
    X_data = measurement.X["data"]
    obs = X.obs

    # Create an IOBatchIterable to handle the data loading
    io_iterable = IOBatchIterable(
        chunks=[range(len(obs))],  # Single chunk for all data
        io_batch_size=batch_size,
        obs=obs,
        var_joinids=np.arange(X_data.shape[1]),
        X=X_data,
        seed=seed,
        shuffle=shuffle,
        use_eager_fetch=use_eager_fetch
    )

    # Create an iterator from the IOBatchIterable
    _io_buf_iter = iter(io_iterable)

    # Use EagerIterator with optimized parameters for prefetching
    if use_eager_fetch:
        # Calculate optimal prefetch size based on batch size and available memory
        prefetch_size = min(use_eager_fetch * 2, 8)  # Cap at 8 to prevent memory issues
        
        # Create a dedicated thread pool for prefetching
        prefetch_pool = ThreadPoolExecutor(
            max_workers=min(prefetch_size, os.cpu_count() or 1),
            thread_name_prefix="eager_fetch"
        )
        
        _io_buf_iter = EagerIterator(
            _io_buf_iter,
            pool=prefetch_pool,
            prefetch=prefetch_size
        )

    return _io_buf_iter
