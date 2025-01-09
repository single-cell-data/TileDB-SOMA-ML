# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from contextlib import contextmanager
from typing import Dict, Generator, Union

import attrs
from tiledbsoma import Experiment, SOMATileDBContext


@attrs.define(frozen=True, kw_only=True)
class ExperimentLocator:
    """State required to open an |Experiment|.

    Serializable across multiple processes.

    Private implementation class.
    """

    uri: str
    tiledb_timestamp_ms: int
    tiledb_config: Dict[str, Union[str, float]]

    @classmethod
    def create(cls, experiment: Experiment) -> "ExperimentLocator":
        return ExperimentLocator(
            uri=experiment.uri,
            tiledb_timestamp_ms=experiment.tiledb_timestamp_ms,
            tiledb_config=experiment.context.tiledb_config,
        )

    @contextmanager
    def open_experiment(self) -> Generator[Experiment, None, None]:
        context = SOMATileDBContext(tiledb_config=self.tiledb_config)
        yield Experiment.open(
            self.uri, tiledb_timestamp=self.tiledb_timestamp_ms, context=context
        )
