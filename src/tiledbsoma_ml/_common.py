# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
"""Type aliases used in |ExperimentDataset| and |experiment_dataloader|."""
from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from attrs import Factory, define
from scipy import sparse
from scipy.sparse import csr_matrix
from torch import Tensor

from tiledbsoma_ml.encoders import Encoder

NDArrayNumber = npt.NDArray[np.number[Any]]
NDArrayJoinId = npt.NDArray[np.int64]
"""|ndarray| of ``int64``s representing SOMA joinids."""
XBatch = Union[NDArrayNumber, sparse.csr_matrix]
XObsTensors = Tuple[Tensor, Tensor]
"""|torch.Tensor|s corresponding to ``X`` and (encoded) ``obs``, (resp.)."""


@define(init=False)
class MiniBatch:
    """Yielded by |ExperimentDataset|; pairs a slice of ``X`` rows with corresponding ``obs`` rows.

    When |return_sparse_X| is ``False`` (the default), a |MiniBatch| is a tuple of |ndarray| and |pd.DataFrame| (for
    ``X`` and ``obs``, respectively). If ``batch_size=1``, the |ndarray| will have rank 1 (representing a single row),
    otherwise it will have rank 2. If |return_sparse_X| is ``True``, the ``X`` slice is returned as a |csr_matrix|
    (which is always rank 2).
    """

    X: XBatch
    obs: pd.DataFrame
    encoders: list[Encoder] = Factory(list)

    def __init__(
        self,
        X: XBatch,
        obs: pd.DataFrame,
        batch_size: int | None = None,
        encoders: list[Encoder] | None = None,
    ):
        if batch_size == 1 and not isinstance(X, csr_matrix):
            # "Squeeze" batches of size 1
            X = X[0]

        self.__attrs_init__(
            X=X,
            obs=obs,
            encoders=encoders or [],
        )

    @property
    def obs_encoded(self) -> Tensor:
        obs = self.obs
        obs_encoded = pd.DataFrame()

        # Add the soma_joinid to the original obs, in case that is requested by the encoders.
        obs["soma_joinid"] = obs.index

        for enc in self.encoders:
            obs_encoded[enc.name] = enc.transform(obs)

        # `to_numpy()` avoids copying the numpy array data
        return torch.from_numpy(obs_encoded.to_numpy())

    @property
    def tensors(self) -> XObsTensors:
        X = self.X
        if isinstance(X, csr_matrix):
            coo = X.tocoo()
            X = torch.sparse_coo_tensor(
                # Note: The `np.array` seems unnecessary, but PyTorch warns bare array is "extremely slow"
                indices=torch.from_numpy(np.array([coo.row, coo.col])),
                values=coo.data,
                size=coo.shape,
            )
        else:
            X = torch.from_numpy(X)

        return X, self.obs_encoded

    @property
    def tpl(self) -> tuple[XBatch, pd.DataFrame]:
        return self.X, self.obs
