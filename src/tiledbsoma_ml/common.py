# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
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
Batch = Tuple[XBatch, pd.DataFrame]
"""Yielded by |ExperimentDataset|; pairs a slice of ``X`` rows with corresponding ``obs`` rows.

In the default case, a Batch is a tuple of |np.ndarray| and |pd.DataFrame| (for ``X`` and ``obs``, respectively). If the
iterator is created with ``return_sparse_X`` as True, the ``X`` slice is returned as a |csr_matrix| instead. If the
``batch_size`` is 1, the |ndarray| will be returned with rank 1; in all other cases, objects are returned with rank 2.
"""
