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
import pyarrow.types as pat
from tiledbsoma import DataFrame, IntIndexer, SparseNDArray

from tiledbsoma_ml._common import NDArrayJoinId
from tiledbsoma_ml._csr import CSR_IO_Buffer, smallest_uint_dtype, coo_scatter_to_csr
from tiledbsoma_ml._eager_iter import EagerIterator
from tiledbsoma_ml._query_ids import Chunks
from tiledbsoma_ml._utils import batched

logger = logging.getLogger("tiledbsoma_ml._io_batch_iterable")
IOBatch = Tuple[CSR_IO_Buffer, pd.DataFrame]
"""Tuple type emitted by |IOBatchIterable|, containing ``X`` rows (as a |CSR_IO_Buffer|) and ``obs`` rows
(|pd.DataFrame|)."""


def _np_dtype_from_arrow(t: pa.DataType) -> np.dtype:
    # Fast, no-copy dtype mapping for common numeric types
    # Might be worth creating a preset dictionary with these as keys instead
    if pat.is_float16(t): return np.dtype(np.float16)
    if pat.is_float32(t): return np.dtype(np.float32)
    if pat.is_float64(t): return np.dtype(np.float64)
    if pat.is_int8(t):    return np.dtype(np.int8)
    if pat.is_int16(t):   return np.dtype(np.int16)
    if pat.is_int32(t):   return np.dtype(np.int32)
    if pat.is_int64(t):   return np.dtype(np.int64)
    if pat.is_uint8(t):   return np.dtype(np.uint8)
    if pat.is_uint16(t):  return np.dtype(np.uint16)
    if pat.is_uint32(t):  return np.dtype(np.uint32)
    if pat.is_uint64(t):  return np.dtype(np.uint64)
    if pat.is_boolean(t): return np.dtype(np.bool_)
    # Fallback allocate small array and return its datatype
    return np.asarray(pa.array([], type=t)).dtype

def _col_to_numpy(col: pa.ChunkedArray) -> np.ndarray:
    """
    Return a NumPy view/copy for an Arrow column:
    - zero-copy if the column has exactly one chunk and supports it,
    - otherwise a single contiguous NumPy array (copy).
    """
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 1:
            arr = col.chunk(0)
            try:
                return arr.to_numpy(zero_copy_only=True)  
            except TypeError:
                return arr.to_numpy()
        return col.to_numpy(zero_copy_only=False)
    # Not chunked -> treat as Array
    try:
        return col.to_numpy(zero_copy_only=True)
    except TypeError:
        return col.to_numpy()


def _iter_col_numpy(col: pa.ChunkedArray):
    """Yield NumPy arrays for each chunk; zero-copy when possible."""
    if isinstance(col, pa.ChunkedArray):
        for i in range(col.num_chunks):
            arr = col.chunk(i)
            try:
                yield arr.to_numpy(zero_copy_only=True)
            except TypeError:
                yield arr.to_numpy()
    else:
        try:
            yield col.to_numpy(zero_copy_only=True)
        except TypeError:
            yield col.to_numpy()


@attrs.define(frozen=True)
class IOBatchIterable(Iterable[IOBatch]):
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
        return batched((joinid for chunk in self.chunks for joinid in chunk),
                       self.io_batch_size)

    def __iter__(self) -> Iterator[IOBatch]:
        X = self.X
        context = X.context

        counts_buf = np.zeros(self.io_batch_size, dtype=np.int64)
        shuffle_rng = np.random.default_rng(self.seed) if self.shuffle else None

        obs_column_names = (
            list(self.obs_column_names)
            if "soma_joinid" in self.obs_column_names
            else ["soma_joinid", *self.obs_column_names]
        )

        var_joinids = np.asarray(self.var_joinids, dtype=np.int64)
        var_indexer = IntIndexer(var_joinids, context=context)

        for obs_coords in self.io_batch_ids:
            st_time = time.perf_counter()

            obs_order = (np.fromiter(obs_coords, dtype=np.int64, count=len(obs_coords))
                        if shuffle_rng is None
                        else shuffle_rng.permuted(obs_coords))

            obs_indexer = IntIndexer(obs_order, context=context)
            logger.debug(f"Retrieving next SOMA IO batch of length {len(obs_coords)}...")

            # First read X
            tables = X.read(coords=(obs_coords, self.var_joinids)).tables()
            if self.use_eager_fetch:
                tables = EagerIterator(tables, pool=X.context.threadpool)
            # Read/Materialize only once
            tables = list(tables)

            obs_io_batch = (
                self.obs.read(coords=(obs_coords,), column_names=obs_column_names)
                .concat().to_pandas()
                .set_index("soma_joinid").reindex(obs_order, copy=False)
                .reset_index()[self.obs_column_names]
            )

            # Count
            n_rows = len(obs_coords)
            n_cols = len(var_joinids)
            counts_buf[:n_rows].fill(0)
            total_nnz = 0
            data_dtype = None

            # Pull from the dictionary for datatype
            if tables:
                data_dtype = np.asarray(
                    pa.array([], type=tables[0].schema.field("soma_data").type)
                ).dtype

            for tbl in tables:
                # row ids per chunk -> bincount
                for Ai_chunk in _iter_col_numpy(tbl["soma_dim_0"]):
                    Ai = obs_indexer.get_indexer(Ai_chunk)
                    counts_buf[:n_rows] += np.bincount(Ai, minlength=n_rows)
                    total_nnz += Ai.shape[0]

            if total_nnz == 0:
                X_io_batch = CSR_IO_Buffer.from_pjd(
                    np.zeros((n_rows + 1,), dtype=smallest_uint_dtype(0)),
                    np.zeros((0,), dtype=smallest_uint_dtype(n_cols)),
                    np.zeros((0,), dtype=np.float32),
                    shape=(n_rows, n_cols),
                )
                del obs_indexer, obs_coords, obs_order, tables
                gc.collect()
                tm = time.perf_counter() - st_time
                logger.debug(
                    f"Retrieved SOMA IO batch, took {tm:.2f}sec, {X_io_batch.shape[0]/tm:0.1f} samples/sec"
                )
                yield X_io_batch, obs_io_batch
                continue

            # Allocate final CSR once
            indptr = np.empty((n_rows + 1,), dtype=smallest_uint_dtype(int(total_nnz)))
            indptr[0] = 0
            np.cumsum(counts_buf[:n_rows], out=indptr[1:])
            indices = np.empty((total_nnz,), dtype=smallest_uint_dtype(n_cols))
            data    = np.empty((total_nnz,), dtype=data_dtype)
            offsets = indptr[:-1].copy()

            # Scatter
            for tbl in tables:
                # Iterate chunks in lockstep. We assume SOMA returns aligned chunking across columns.
                iter_Ai = _iter_col_numpy(tbl["soma_dim_0"])
                iter_Aj = _iter_col_numpy(tbl["soma_dim_1"])
                iter_Ad = _iter_col_numpy(tbl["soma_data"])
                for Ai_chunk, Aj_chunk, Ad_chunk in zip(iter_Ai, iter_Aj, iter_Ad):
                    Ai = obs_indexer.get_indexer(Ai_chunk)
                    Aj = var_indexer.get_indexer(Aj_chunk).astype(indices.dtype, copy=False)
                    Ad = Ad_chunk  # already NumPy
                    coo_scatter_to_csr(Ai, Aj, Ad, offsets, indices, data)

            X_io_batch = CSR_IO_Buffer.from_pjd(indptr, indices, data, shape=(n_rows, n_cols))

            del obs_indexer, obs_coords, obs_order, tables
            gc.collect()

            tm = time.perf_counter() - st_time
            logger.debug(
                f"Retrieved SOMA IO batch, took {tm:.2f}sec, {X_io_batch.shape[0]/tm:0.1f} samples/sec"
            )
            yield X_io_batch, obs_io_batch
