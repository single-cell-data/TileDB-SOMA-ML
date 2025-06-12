# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
"""An API to support machine learning applications built on SOMA."""

from importlib.metadata import PackageNotFoundError, version

from ._query_ids import SamplingMethod
from .dataloader import experiment_dataloader, optimized_experiment_dataloader
from .dataset import ExperimentDataset, OptimizedExperimentDataset
from .scvi import SCVIDataModule

try:
    __version__ = version("tiledbsoma-ml")
except PackageNotFoundError:
    # package is not installed
    pass


__all__ = [
    "ExperimentDataset",
    "OptimizedExperimentDataset",
    "experiment_dataloader",
    "optimized_experiment_dataloader",
    "SamplingMethod",
    "SCVIDataModule",
]
