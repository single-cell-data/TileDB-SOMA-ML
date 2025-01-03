# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

import sys
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse

from tests._utils import parametrize


@parametrize(  # keep these small as we materialize as a dense ndarray
    "shape",
    [(100, 10), (10, 100), (1, 1), (1, 100), (100, 1), (0, 0), (10, 0), (0, 10)],
)
@parametrize("dtype", [np.float32, np.float64, np.int32])
def test_construct_from_ijd(shape: Tuple[int, int], dtype: npt.DTypeLike):
    from tiledbsoma_ml._csr import CSR_IO_Buffer

    sp_coo = sparse.random(shape[0], shape[1], dtype=dtype, format="coo", density=0.05)
    sp_csr = sp_coo.tocsr()

    _ncsr = CSR_IO_Buffer.from_ijd(
        sp_coo.row, sp_coo.col, sp_coo.data, shape=sp_coo.shape
    )
    assert _ncsr.nnz == sp_coo.nnz == sp_csr.nnz
    assert _ncsr.dtype == sp_coo.dtype == sp_csr.dtype
    assert _ncsr.nbytes == (
        _ncsr.data.nbytes + _ncsr.indices.nbytes + _ncsr.indptr.nbytes
    )

    # CSR_IO_Buffer makes no guarantees about minor axis ordering (ie, "canonical" form) until
    # sort_indices is called, so use the SciPy sparse csr package to validate by round-tripping.
    assert (
        sparse.csr_matrix((_ncsr.data, _ncsr.indices, _ncsr.indptr), shape=_ncsr.shape)
        != sp_csr
    ).nnz == 0

    # Check dense slicing
    assert np.array_equal(_ncsr.slice_tonumpy(slice(0, shape[0])), sp_coo.toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(0, shape[0])), sp_csr.toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(1, -1)), sp_csr[1:-1].toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(None, -2)), sp_csr[:-2].toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(None)), sp_csr[:].toarray())

    # Check sparse slicing
    assert (_ncsr.slice_toscipy(slice(0, shape[0])) != sp_csr).nnz == 0
    assert (_ncsr.slice_toscipy(slice(1, -1)) != sp_csr[1:-1]).nnz == 0
    assert (_ncsr.slice_toscipy(slice(None, -2)) != sp_csr[:-2]).nnz == 0
    assert (_ncsr.slice_toscipy(slice(None)) != sp_csr[:]).nnz == 0


@parametrize(
    "shape",
    [(100, 10), (10, 100), (1, 1), (1, 100), (100, 1), (0, 0), (10, 0), (0, 10)],
)
@parametrize("dtype", [np.float32, np.float64, np.int32])
def test_construct_from_pjd(shape: Tuple[int, int], dtype: npt.DTypeLike):
    from tiledbsoma_ml._csr import CSR_IO_Buffer

    sp_csr = sparse.random(shape[0], shape[1], dtype=dtype, format="csr", density=0.05)

    _ncsr = CSR_IO_Buffer.from_pjd(
        sp_csr.indptr.copy(),
        sp_csr.indices.copy(),
        sp_csr.data.copy(),
        shape=sp_csr.shape,
    )

    # CSR_IO_Buffer makes no guarantees about minor axis ordering (ie, "canonical" form) until
    # sort_indices is called, so use the SciPy sparse csr package to validate by round-tripping.
    assert (
        sparse.csr_matrix((_ncsr.data, _ncsr.indices, _ncsr.indptr), shape=_ncsr.shape)
        != sp_csr
    ).nnz == 0

    # Check dense slicing
    assert np.array_equal(_ncsr.slice_tonumpy(slice(0, shape[0])), sp_csr.toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(1, -1)), sp_csr[1:-1].toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(None, -2)), sp_csr[:-2].toarray())
    assert np.array_equal(_ncsr.slice_tonumpy(slice(None)), sp_csr[:].toarray())

    # Check sparse slicing
    assert (_ncsr.slice_toscipy(slice(0, shape[0])) != sp_csr).nnz == 0
    assert (_ncsr.slice_toscipy(slice(1, -1)) != sp_csr[1:-1]).nnz == 0
    assert (_ncsr.slice_toscipy(slice(None, -2)) != sp_csr[:-2]).nnz == 0
    assert (_ncsr.slice_toscipy(slice(None)) != sp_csr[:]).nnz == 0


@parametrize(
    "shape",
    [(100, 10), (10, 100)],
)
@parametrize("dtype", [np.float32, np.float64, np.int32])
@parametrize("n_splits", [2, 3, 4])
def test_merge(shape: Tuple[int, int], dtype: npt.DTypeLike, n_splits: int):
    from tiledbsoma_ml._csr import CSR_IO_Buffer

    sp_coo = sparse.random(shape[0], shape[1], dtype=dtype, format="coo", density=0.5)
    splits = [
        t
        for t in zip(
            np.array_split(sp_coo.row, n_splits),
            np.array_split(sp_coo.col, n_splits),
            np.array_split(sp_coo.data, n_splits),
            **(dict(strict=False) if sys.version_info >= (3, 10) else {}),
        )
    ]
    _ncsr = CSR_IO_Buffer.merge(
        [CSR_IO_Buffer.from_ijd(i, j, d, shape=sp_coo.shape) for i, j, d in splits]
    )

    assert (
        sp_coo.tocsr()
        != sparse.csr_matrix(
            (_ncsr.data, _ncsr.indices, _ncsr.indptr), shape=_ncsr.shape
        )
    ).nnz == 0


@parametrize(
    "shape",
    [(100, 10), (10, 100), (1, 1), (1, 100), (100, 1), (0, 0), (10, 0), (0, 10)],
)
def test_sort_indices(shape: Tuple[int, int]):
    from tiledbsoma_ml._csr import CSR_IO_Buffer

    sp_coo = sparse.random(
        shape[0], shape[1], dtype=np.float32, format="coo", density=0.05
    )
    sp_csr = sp_coo.tocsr()

    _ncsr = CSR_IO_Buffer.from_ijd(
        sp_coo.row, sp_coo.col, sp_coo.data, shape=sp_coo.shape
    ).sort_indices()

    assert np.array_equal(sp_csr.indptr, _ncsr.indptr)
    assert np.array_equal(sp_csr.indices, _ncsr.indices)
    assert np.array_equal(sp_csr.data, _ncsr.data)
