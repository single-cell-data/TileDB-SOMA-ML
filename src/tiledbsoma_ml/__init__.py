# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
"""An API to support machine learning applications built on SOMA."""

from importlib.metadata import PackageNotFoundError, version

from .common import MiniBatch, NDArrayJoinId, NDArrayNumber, XBatch
from .dataloader import experiment_dataloader
from .dataset import ExperimentDataset
from .query_ids import Chunks, Partition, QueryIDs, SamplingMethod

try:
    __version__ = version("tiledbsoma-ml")
except PackageNotFoundError:
    # package is not installed
    pass


__all__ = [
    "ExperimentDataset",
    "experiment_dataloader",
    # Type aliases, from `.common`
    "MiniBatch",
    "NDArrayJoinId",
    "NDArrayNumber",
    "XBatch",
    # `.query_ids`
    "Chunks",
    "Partition",
    "QueryIDs",
    "SamplingMethod",
]
