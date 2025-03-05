# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
"""Type aliases used in |ExperimentDataset| and |experiment_dataloader|."""

from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

NDArrayNumber = npt.NDArray[np.number[Any]]
NDArrayJoinId = npt.NDArray[np.int64]
"""|ndarray| of ``int64``s representing SOMA joinids."""
XBatch = Union[NDArrayNumber, sparse.csr_matrix]
MiniBatch = Tuple[XBatch, pd.DataFrame]
"""Yielded by |ExperimentDataset|; pairs a slice of ``X`` rows with corresponding ``obs`` rows.

When |return_sparse_X| is ``False`` (the default), a |MiniBatch| is a tuple of |ndarray| and |pd.DataFrame| (for ``X``
and ``obs``, respectively). If ``batch_size=1``, the |ndarray| will have rank 1 (representing a single row), otherwise
it will have rank 2. If |return_sparse_X| is ``True``, the ``X`` slice is returned as a |csr_matrix| (which is always
rank 2).
"""
