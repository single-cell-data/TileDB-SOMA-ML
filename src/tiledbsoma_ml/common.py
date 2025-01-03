# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

NDArrayNumber = npt.NDArray[np.number[Any]]
NDArrayJoinId = npt.NDArray[np.int64]
XBatch = Union[NDArrayNumber, sparse.csr_matrix]
Batch = Tuple[XBatch, pd.DataFrame]
""""Batch" type yielded by ``ExperimentDataset``; pairs a slice of ``X`` rows with
a corresponding slice of ``obs``. In the default case, a Batch is a tuple of :class:`numpy.ndarray`
and :class:`pandas.DataFrame` (for ``X`` and ``obs``, respectively). If the iterator is created
with ``return_sparse_X`` as True, the ``X`` slice is returned as a
:class:`scipy.sparse.csr_matrix`. If the ``batch_size`` is 1, the :class:`numpy.ndarray` will be
returned with rank 1; in all other cases, objects are returned with rank 2."""
