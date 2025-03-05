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
