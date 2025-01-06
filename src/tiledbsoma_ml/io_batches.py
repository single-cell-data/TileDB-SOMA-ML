# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
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

from tiledbsoma_ml._csr import CSR_IO_Buffer
from tiledbsoma_ml._utils import EagerIterator, batched
from tiledbsoma_ml.common import NDArrayJoinId
from tiledbsoma_ml.query_ids import Chunks

logger = logging.getLogger("tiledbsoma_ml.io_batches")
IOBatch = Tuple[CSR_IO_Buffer, pd.DataFrame]


@attrs.define(frozen=True)
class IOBatches(Iterable[IOBatch]):
    chunks: Chunks
    io_batch_size: int
    obs: DataFrame
    var_joinids: NDArrayJoinId
    X: SparseNDArray
    obs_column_names: Sequence[str] = ("soma_joinid",)
    seed: Optional[int] = None
    shuffle: bool = True
    use_eager_fetch: bool = True
    """Given a list of obs_joinid chunks, re-chunk them into (optionally shuffled) "IO batches".

    An "IO Batch" is a tuple consisting of a batch of rows from the ``X`` ``SparseNDArray``, as
    well as the corresponding rows from the ``obs`` ``DataFrame``. The ``X`` rows are returned in
    an optimized ``CSR_IO_Buffer``."""

    @property
    def io_batch_ids(self) -> Iterable[Tuple[int, ...]]:
        """Re-chunk the ``obs_joinids`` according to the desired ``io_batch_size``."""
        return batched(
            (joinid for chunk in self.chunks for joinid in chunk),
            self.io_batch_size,
        )

    def __iter__(self) -> Iterator[IOBatch]:
        """Emit IO batches, i.e. ``(X: CSR_IO_Buffer, obs: pd.DataFrame)`` tuples."""
        # Create RNG - does not need to be identical across processes, but use the seed anyway
        # for reproducibility.
        shuffle_rng = np.random.default_rng(self.seed)
        X = self.X
        context = X.context
        obs_column_names = (
            list(self.obs_column_names)
            if "soma_joinid" in self.obs_column_names
            else ["soma_joinid", *self.obs_column_names]
        )
        # Round-trip though tuple avoids `TypeError: IntIndexer only supports array of type int64`.
        # TODO: debug / work around that error; serde'ing the ndarray apparently results in a second np.int64 instance, that fails reference equality check vs. the version from the worker-process.
        var_joinids = np.array(tuple(self.var_joinids))
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

            _io_buf_iter = (
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
