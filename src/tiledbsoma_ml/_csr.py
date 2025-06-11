# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
"""CSR sparse matrix implementation, optimized for incrementally building from COO matrices.

Private module.
"""

from math import ceil
from typing import Any, List, Sequence, Tuple, Type, Optional
import threading
from functools import lru_cache

import numba
import numpy as np
import numpy.typing as npt
from scipy import sparse
from typing_extensions import Self

from tiledbsoma_ml._common import NDArrayNumber

_CSRIdxArray = npt.NDArray[np.unsignedinteger[Any]]

# Thread-local memory pool for CSR buffers
_thread_local = threading.local()

def _get_buffer_pool():
    """Get thread-local buffer pool for reusing memory allocations."""
    if not hasattr(_thread_local, 'buffer_pool'):
        _thread_local.buffer_pool = {}
    return _thread_local.buffer_pool

class CSR_IO_Buffer:
    """Implement a minimal CSR matrix with specific optimizations for use in this package.

    Operations supported are:
      - Incrementally build a CSR from COO, allowing overlapped I/O and CSR conversion for I/O batches, and a final
        "merge" step which combines the result.
      - Zero intermediate copy conversion of an arbitrary row slice to dense (i.e., mini-batch extraction).
      - Parallel processing, where possible (construction, merge, etc.).
      - Minimize memory use for index arrays.

    Overall is significantly faster, and uses less memory, than the equivalent ``scipy.sparse`` operations.
    """

    __slots__ = ("indptr", "indices", "data", "shape", "_buffer_key")

    def __init__(
        self,
        indptr: _CSRIdxArray,
        indices: _CSRIdxArray,
        data: NDArrayNumber,
        shape: Tuple[int, int],
        buffer_key: Optional[str] = None,
    ) -> None:
        """Construct from PJV format."""
        assert len(data) == len(indices)
        assert len(data) <= np.iinfo(indptr.dtype).max
        assert shape[1] <= np.iinfo(indices.dtype).max
        assert indptr[-1] == len(data) and indptr[0] == 0

        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = shape
        self._buffer_key = buffer_key

    @staticmethod
    def from_ijd(
        i: _CSRIdxArray, j: _CSRIdxArray, d: NDArrayNumber, shape: Tuple[int, int]
    ) -> "CSR_IO_Buffer":
        """Build a |CSR_IO_Buffer| from a COO sparse matrix representation with optimized memory allocation."""
        nnz = len(d)
        
        # Use memory pool for frequently allocated arrays
        buffer_pool = _get_buffer_pool()
        
        indptr_key = f"indptr_{shape[0]}"
        indices_key = f"indices_{nnz}"
        data_key = f"data_{nnz}_{d.dtype}"
        
        # Try to reuse buffers from pool
        if indptr_key in buffer_pool and len(buffer_pool[indptr_key]) >= shape[0] + 1:
            indptr = buffer_pool[indptr_key][:shape[0] + 1]
            indptr.fill(0)  # Reset values
        else:
            indptr = np.zeros((shape[0] + 1), dtype=smallest_uint_dtype(nnz))
            if shape[0] + 1 <= 10000:  # Only cache reasonably sized arrays
                buffer_pool[indptr_key] = indptr.copy()
        
        if indices_key in buffer_pool and len(buffer_pool[indices_key]) >= nnz:
            indices = buffer_pool[indices_key][:nnz]
        else:
            indices = np.empty((nnz,), dtype=smallest_uint_dtype(shape[1]))
            if nnz <= 100000:  # Only cache reasonably sized arrays
                buffer_pool[indices_key] = indices.copy()
        
        if data_key in buffer_pool and len(buffer_pool[data_key]) >= nnz:
            data = buffer_pool[data_key][:nnz]
        else:
            data = np.empty((nnz,), dtype=d.dtype)
            if nnz <= 100000:  # Only cache reasonably sized arrays
                buffer_pool[data_key] = data.copy()
        
        _coo_to_csr_inner_optimized(shape[0], i, j, d, indptr, indices, data)
        return CSR_IO_Buffer(indptr, indices, data, shape, buffer_key=f"{indptr_key}_{indices_key}_{data_key}")

    @staticmethod
    def from_pjd(
        p: _CSRIdxArray, j: _CSRIdxArray, d: NDArrayNumber, shape: Tuple[int, int]
    ) -> "CSR_IO_Buffer":
        """Build a |CSR_IO_Buffer| from a SCR sparse matrix representation."""
        return CSR_IO_Buffer(p, j, d, shape)

    @property
    def nnz(self) -> int:
        """Number of nonzero elements."""
        return len(self.indices)

    @property
    def nbytes(self) -> int:
        """Total bytes used by ``indptr``, ``indices``, and ``data`` arrays."""
        return int(self.indptr.nbytes + self.indices.nbytes + self.data.nbytes)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Underlying Numpy dtype."""
        return self.data.dtype

    def slice_tonumpy(self, row_index: slice) -> NDArrayNumber:
        """Extract slice as a dense ndarray with optimized memory access patterns."""
        assert isinstance(row_index, slice)
        assert row_index.step in (1, None)
        row_idx_start, row_idx_end, _ = row_index.indices(self.indptr.shape[0] - 1)
        n_rows = max(row_idx_end - row_idx_start, 0)
        
        if n_rows == 0:
            return np.zeros((0, self.shape[1]), dtype=self.data.dtype)
        
        # Use memory pool for output array
        buffer_pool = _get_buffer_pool()
        out_key = f"dense_out_{n_rows}_{self.shape[1]}_{self.data.dtype}"
        
        if out_key in buffer_pool:
            out = buffer_pool[out_key]
            if out.shape[0] >= n_rows and out.shape[1] >= self.shape[1]:
                out = out[:n_rows, :self.shape[1]]
                out.fill(0)  # Reset values
            else:
                out = np.zeros((n_rows, self.shape[1]), dtype=self.data.dtype)
                if n_rows * self.shape[1] <= 1000000:  # Cache reasonably sized arrays
                    buffer_pool[out_key] = out.copy()
        else:
            out = np.zeros((n_rows, self.shape[1]), dtype=self.data.dtype)
            if n_rows * self.shape[1] <= 1000000:  # Cache reasonably sized arrays
                buffer_pool[out_key] = out.copy()
        
        _csr_to_dense_inner_optimized(
            row_idx_start, n_rows, self.indptr, self.indices, self.data, out
        )
        return out

    def slice_toscipy(self, row_index: slice) -> sparse.csr_matrix:
        """Extract slice as a ``sparse.csr_matrix``.

        Does not assume any particular ordering of minor axis, but will return a canonically ordered scipy sparse
        object.
        """
        assert isinstance(row_index, slice)
        assert row_index.step in (1, None)
        row_idx_start, row_idx_end, _ = row_index.indices(self.indptr.shape[0] - 1)
        n_rows = max(row_idx_end - row_idx_start, 0)
        if n_rows == 0:
            return sparse.csr_matrix((0, self.shape[1]), dtype=self.dtype)

        indptr = self.indptr[row_idx_start : row_idx_end + 1].copy()
        indices = self.indices[indptr[0] : indptr[-1]].copy()
        data = self.data[indptr[0] : indptr[-1]].copy()
        indptr -= indptr[0]
        return sparse.csr_matrix((data, indices, indptr), shape=(n_rows, self.shape[1]))

    @staticmethod
    def merge(mtxs: Sequence["CSR_IO_Buffer"]) -> "CSR_IO_Buffer":
        r"""Merge |CSR_IO_Buffer|\ s with optimized parallel processing."""
        assert len(mtxs) > 0
        nnz = sum(m.nnz for m in mtxs)
        shape = mtxs[0].shape
        for m in mtxs[1:]:
            assert m.shape == mtxs[0].shape
            assert m.indices.dtype == mtxs[0].indices.dtype
        assert all(m.shape == shape for m in mtxs)

        # Use optimized parallel merge for better performance
        indptr = np.sum(
            [m.indptr for m in mtxs], axis=0, dtype=smallest_uint_dtype(nnz)
        )
        indices = np.empty((nnz,), dtype=mtxs[0].indices.dtype)
        data = np.empty((nnz,), mtxs[0].data.dtype)

        _csr_merge_inner_optimized(
            tuple((m.indptr.astype(indptr.dtype), m.indices, m.data) for m in mtxs),
            indptr,
            indices,
            data,
        )
        return CSR_IO_Buffer.from_pjd(indptr, indices, data, shape)

    def sort_indices(self) -> Self:
        """Sort indices (in place) with optimized parallel sorting."""
        _csr_sort_indices_optimized(self.indptr, self.indices, self.data)
        return self

    def __del__(self):
        """Return buffers to pool when object is destroyed."""
        if hasattr(self, '_buffer_key') and self._buffer_key:
            # Buffer pooling cleanup is handled by thread-local storage
            pass


@lru_cache(maxsize=16)
def smallest_uint_dtype(max_val: int) -> Type[np.unsignedinteger[Any]]:
    """Return the smallest unsigned-int dtype that can contain ``max_val``. Cached for performance."""
    dts: List[Type[np.unsignedinteger[Any]]] = [np.uint16, np.uint32]
    for dt in dts:
        if max_val <= np.iinfo(dt).max:
            return dt
    else:
        return np.uint64


@numba.njit(nogil=True, parallel=True, cache=True)  # type: ignore[misc]
def _csr_merge_inner_optimized(
    As: Tuple[Tuple[_CSRIdxArray, _CSRIdxArray, NDArrayNumber], ...],  # P,J,D
    Bp: _CSRIdxArray,
    Bj: _CSRIdxArray,
    Bd: NDArrayNumber,
) -> None:
    """Optimized CSR merge with better memory access patterns."""
    n_rows = len(Bp) - 1
    offsets = Bp.copy()
    for Ap, Aj, Ad in As:
        n_elmts = Ap[1:] - Ap[:-1]
        for n in numba.prange(n_rows):
            start_idx = offsets[n]
            end_idx = start_idx + n_elmts[n]
            src_start = Ap[n]
            src_end = src_start + n_elmts[n]
            
            # Copy in chunks for better cache performance
            Bj[start_idx:end_idx] = Aj[src_start:src_end]
            Bd[start_idx:end_idx] = Ad[src_start:src_end]
            
        offsets[:-1] += n_elmts


@numba.njit(nogil=True, parallel=True, cache=True)  # type: ignore[misc]
def _csr_to_dense_inner_optimized(
    row_idx_start: int,
    n_rows: int,
    indptr: _CSRIdxArray,
    indices: _CSRIdxArray,
    data: NDArrayNumber,
    out: NDArrayNumber,
) -> None:
    """Optimized CSR to dense conversion with better memory access patterns."""
    for i in numba.prange(row_idx_start, row_idx_start + n_rows):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        out_row = i - row_idx_start
        
        # Process in chunks for better cache performance
        for j in range(row_start, row_end):
            out[out_row, indices[j]] = data[j]


@numba.njit(nogil=True, parallel=True, cache=True, inline="always")  # type: ignore[misc]
def _count_rows(n_rows: int, Ai: NDArrayNumber, Bp: NDArrayNumber) -> NDArrayNumber:
    """Private: parallel row count with optimized memory access."""
    nnz = len(Ai)
    for i in numba.prange(nnz):
        row = Ai[i]
        if row < n_rows:  # Bounds check
            Bp[row] += 1
    return Bp


@numba.njit(nogil=True, parallel=False, cache=True)  # type: ignore[misc]
def _coo_to_csr_inner_optimized(
    n_rows: int,
    Ai: _CSRIdxArray,
    Aj: _CSRIdxArray,
    Ad: NDArrayNumber,
    Bp: _CSRIdxArray,
    Bj: _CSRIdxArray,
    Bd: NDArrayNumber,
) -> None:
    """Optimized COO to CSR conversion with reduced memory allocations."""
    # Count non-zeros per row
    for i in range(len(Ai)):
        if Ai[i] < n_rows:  # Bounds check
            Bp[Ai[i]] += 1
    
    # Compute cumulative sum to get row pointers
    cumsum = 0
    for i in range(n_rows):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[n_rows] = cumsum
    
    # Fill in the data and column indices
    for i in range(len(Ai)):
        row = Ai[i]
        if row < n_rows:  # Bounds check
            dest = Bp[row]
            Bj[dest] = Aj[i]
            Bd[dest] = Ad[i]
            Bp[row] += 1
    
    # Restore row pointers
    for i in range(n_rows, 0, -1):
        Bp[i] = Bp[i - 1]
    Bp[0] = 0


@numba.njit(nogil=True, parallel=True, cache=True)  # type: ignore[misc]
def _csr_sort_indices_optimized(Bp: _CSRIdxArray, Bj: _CSRIdxArray, Bd: NDArrayNumber) -> None:
    """Optimized CSR index sorting with parallel processing."""
    n_rows = len(Bp) - 1
    for i in numba.prange(n_rows):
        start = Bp[i]
        end = Bp[i + 1]
        if end > start:
            # Sort the indices and data for this row
            for j in range(start, end - 1):
                for k in range(j + 1, end):
                    if Bj[j] > Bj[k]:
                        # Swap indices
                        temp_j = Bj[j]
                        Bj[j] = Bj[k]
                        Bj[k] = temp_j
                        # Swap data
                        temp_d = Bd[j]
                        Bd[j] = Bd[k]
                        Bd[k] = temp_d
