# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
"""CSR sparse matrix implementation, optimized for incrementally building from COO matrices.

Private module.
"""

from math import ceil
from typing import Any, List, Sequence, Tuple, Type

import numba
import numpy as np
import numpy.typing as npt
import time, os, sys
from scipy import sparse
from typing_extensions import Self

from tiledbsoma_ml._common import NDArrayNumber

_CSRIdxArray = npt.NDArray[np.unsignedinteger[Any]]


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

    __slots__ = ("indptr", "indices", "data", "shape")

    def __init__(
        self,
        indptr: _CSRIdxArray,
        indices: _CSRIdxArray,
        data: NDArrayNumber,
        shape: Tuple[int, int],
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

    @staticmethod
    def from_ijd(
        i: _CSRIdxArray, j: _CSRIdxArray, d: NDArrayNumber, shape: Tuple[int, int]
    ) -> "CSR_IO_Buffer":
        """Build a |CSR_IO_Buffer| from a COO sparse matrix representation."""
        nnz = len(d)
        indptr: _CSRIdxArray = np.zeros((shape[0] + 1), dtype=smallest_uint_dtype(nnz))
        indices: _CSRIdxArray = np.empty((nnz,), dtype=smallest_uint_dtype(shape[1]))
        data = np.empty((nnz,), dtype=d.dtype)
        _coo_to_csr_inner(shape[0], i, j, d, indptr, indices, data)
        return CSR_IO_Buffer(indptr, indices, data, shape)

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
        """Extract slice as a dense ndarray.

        Does not assume any particular ordering of minor axis.
        """
        t0 = time.perf_counter()
        assert isinstance(row_index, slice)
        assert row_index.step in (1, None)
        row_idx_start, row_idx_end, _ = row_index.indices(self.indptr.shape[0] - 1)
        t1 = time.perf_counter()
        n_rows = max(row_idx_end - row_idx_start, 0)
        t2 = time.perf_counter()
        out = np.zeros((n_rows, self.shape[1]), dtype=self.data.dtype)
        print(f"We have n_rows= {n_rows} and shape be {self.shape[1]}")
        t3 = time.perf_counter()
        if n_rows >= 0:
            _csr_to_dense_inner(
                row_idx_start, n_rows, self.indptr, self.indices, self.data, out
            )
        else:
            print("0 rows here")
        t4 = time.perf_counter()
        print(
            f"[PID {os.getpid()}] slice_tonumpy called for rows "
            f"{row_index.start}:{row_index.stop}\n",   # noqa: E501
        )
        print(f"withing slice_tonumpy, we have t1 {(t1-t0)*1000:7.1f}; t2 {(t2-t1)*1000:7.1f}; t3 {(t3-t2)*1000:7.1f}; t4 {(t4-t3)*1000:7.1f};\n")

        return out

    def slice_toscipy(self, row_index: slice) -> sparse.csr_matrix:
        """Extract slice as a ``sparse.csr_matrix``.

        Does not assume any particular ordering of minor axis, but will return a canonically ordered scipy sparse
        object.
        """
        print("In scipy")
        print(
            f"[PID {os.getpid()}] slice_toscipy called for rows "
            f"{row_index.start}:{row_index.stop}",   # noqa: E501
            file=sys.stderr, flush=True
        )
        t0 = time.perf_counter()
        assert isinstance(row_index, slice)
        assert row_index.step in (1, None)
        t1 = time.perf_counter()
        row_idx_start, row_idx_end, _ = row_index.indices(self.indptr.shape[0] - 1)
        t2 = time.perf_counter()
        n_rows = max(row_idx_end - row_idx_start, 0)
        t3 = time.perf_counter()
        if n_rows == 0:
            print("Num Rows 0???")
            return sparse.csr_matrix((0, self.shape[1]), dtype=self.dtype)

        indptr = self.indptr[row_idx_start : row_idx_end + 1].copy()
        t4 = time.perf_counter()
        indices = self.indices[indptr[0] : indptr[-1]].copy()
        t5 = time.perf_counter()
        data = self.data[indptr[0] : indptr[-1]].copy()
        indptr -= indptr[0]
        t6 = time.perf_counter()
        print("TEST\n")
        print(f"withing slice_toscipy, we have t1 {(t1-t0)*1000:7.1f}; t2 {(t2-t1)*1000:7.1f}; t3 {(t3-t2)*1000:7.1f}; t4 {(t4-t3)*1000:7.1f}; t5 {(t5-t4)*1000:7.1f}; t6 {(t6-t5)*1000:7.1f};\n")
        return sparse.csr_matrix((data, indices, indptr), shape=(n_rows, self.shape[1]))

    @staticmethod
    def merge(mtxs: Sequence["CSR_IO_Buffer"]) -> "CSR_IO_Buffer":
        r"""Merge |CSR_IO_Buffer|\ s."""
        assert len(mtxs) > 0
        nnz = sum(m.nnz for m in mtxs)
        shape = mtxs[0].shape
        for m in mtxs[1:]:
            assert m.shape == mtxs[0].shape
            assert m.indices.dtype == mtxs[0].indices.dtype
        assert all(m.shape == shape for m in mtxs)

        indptr = np.sum(
            [m.indptr for m in mtxs], axis=0, dtype=smallest_uint_dtype(nnz)
        )
        indices = np.empty((nnz,), dtype=mtxs[0].indices.dtype)
        data = np.empty((nnz,), mtxs[0].data.dtype)

        _csr_merge_inner(
            tuple((m.indptr.astype(indptr.dtype), m.indices, m.data) for m in mtxs),
            indptr,
            indices,
            data,
        )
        return CSR_IO_Buffer.from_pjd(indptr, indices, data, shape)

    def sort_indices(self) -> Self:
        """Sort indices (in place)."""
        _csr_sort_indices(self.indptr, self.indices, self.data)
        return self


def smallest_uint_dtype(max_val: int) -> Type[np.unsignedinteger[Any]]:
    """Return the smallest unsigned-int dtype that can contain ``max_val``."""
    dts: List[Type[np.unsignedinteger[Any]]] = [np.uint16, np.uint32]
    for dt in dts:
        if max_val <= np.iinfo(dt).max:
            return dt
    else:
        return np.uint64


@numba.njit(nogil=True, parallel=True)  # type: ignore[misc]
def _csr_merge_inner(
    As: Tuple[Tuple[_CSRIdxArray, _CSRIdxArray, NDArrayNumber], ...],  # P,J,D
    Bp: _CSRIdxArray,
    Bj: _CSRIdxArray,
    Bd: NDArrayNumber,
) -> None:
    n_rows = len(Bp) - 1
    offsets = Bp.copy()
    for Ap, Aj, Ad in As:
        n_elmts = Ap[1:] - Ap[:-1]
        for n in numba.prange(n_rows):
            Bj[offsets[n] : offsets[n] + n_elmts[n]] = Aj[Ap[n] : Ap[n] + n_elmts[n]]
            Bd[offsets[n] : offsets[n] + n_elmts[n]] = Ad[Ap[n] : Ap[n] + n_elmts[n]]
        offsets[:-1] += n_elmts

@numba.njit(nogil=True, parallel=True)  # type: ignore[misc]
def _csr_to_dense_inner(
    row_idx_start: int,
    n_rows: int,
    indptr: _CSRIdxArray,
    indices: _CSRIdxArray,
    data: NDArrayNumber,
    out: NDArrayNumber,
) -> None:
    for i in numba.prange(row_idx_start, row_idx_start + n_rows):
        for j in range(indptr[i], indptr[i + 1]):
            out[i - row_idx_start, indices[j]] = data[j]


@numba.njit(nogil=True, parallel=True, inline="always")  # type: ignore[misc]
def _count_rows(n_rows: int, Ai: NDArrayNumber, Bp: NDArrayNumber) -> NDArrayNumber:
    """Private: parallel row count."""
    nnz = len(Ai)

    partition_size = 32 * 1024**2
    n_partitions = ceil(nnz / partition_size)
    if n_partitions > 1:
        counts = np.zeros((n_partitions, n_rows), dtype=Bp.dtype)
        for p in numba.prange(n_partitions):
            for n in range(p * partition_size, min(nnz, (p + 1) * partition_size)):
                row = Ai[n]
                counts[p, row] += 1

        Bp[:-1] = counts.sum(axis=0)
    else:
        for n in range(nnz):
            row = Ai[n]
            Bp[row] += 1

    return Bp


@numba.njit(nogil=True, parallel=True)  # type: ignore[misc]
def _coo_to_csr_inner(
    n_rows: int,
    Ai: _CSRIdxArray,
    Aj: _CSRIdxArray,
    Ad: NDArrayNumber,
    Bp: _CSRIdxArray,
    Bj: _CSRIdxArray,
    Bd: NDArrayNumber,
) -> None:
    nnz = len(Ai)

    _count_rows(n_rows, Ai, Bp)

    # cum sum to get the row index pointers (NOTE: starting with zero)
    cumsum = 0
    for n in range(n_rows):
        tmp = Bp[n]
        Bp[n] = cumsum
        cumsum += tmp
    Bp[n_rows] = nnz

    # Reorganize all the data. Side effect: pointers shifted (reversed in the
    # subsequent section).
    #
    # Method is concurrent (partitioned by rows) if number of rows is greater
    # than 2**partition_bits. This partitioning scheme leverages the fact
    # that reads are much cheaper than writes.
    #
    # The code is equivalent to:
    #   for n in range(nnz):
    #       row = Ai[n]
    #       dst_row = Bp[row]
    #       Bj[dst_row] = Aj[n]
    #       Bd[dst_row] = Ad[n]
    #       Bp[row] += 1

    partition_bits = 13
    n_partitions = (n_rows + 2**partition_bits - 1) >> partition_bits
    for p in numba.prange(n_partitions):
        for n in range(nnz):
            row = Ai[n]
            if (row >> partition_bits) != p:
                continue
            dst_row = Bp[row]
            Bj[dst_row] = Aj[n]
            Bd[dst_row] = Ad[n]
            Bp[row] += 1

    # Shift the pointers by one slot (i.e., start at zero)
    prev_ptr = 0
    for n in range(n_rows + 1):
        tmp = Bp[n]
        Bp[n] = prev_ptr
        prev_ptr = tmp


@numba.njit(nogil=True, parallel=True)  # type: ignore[misc]
def _csr_sort_indices(Bp: _CSRIdxArray, Bj: _CSRIdxArray, Bd: NDArrayNumber) -> None:
    """In-place sort of minor axis indices."""
    n_rows = len(Bp) - 1
    for r in numba.prange(n_rows):
        row_start = Bp[r]
        row_end = Bp[r + 1]
        order = np.argsort(Bj[row_start:row_end])
        Bj[row_start:row_end] = Bj[row_start:row_end][order]
        Bd[row_start:row_end] = Bd[row_start:row_end][order]
