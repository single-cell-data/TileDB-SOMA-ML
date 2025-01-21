# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

"""An API to support machine learning applications built on SOMA."""

from importlib.metadata import PackageNotFoundError, version

from .dataloader import experiment_dataloader
from .datapipe import ExperimentAxisQueryIterDataPipe
from .dataset import ExperimentAxisQueryIterableDataset

try:
    __version__ = version("tiledbsoma-ml")
except PackageNotFoundError:
    # package is not installed
    pass


__all__ = [
    "ExperimentAxisQueryIterDataPipe",
    "ExperimentAxisQueryIterableDataset",
    "experiment_dataloader",
]
