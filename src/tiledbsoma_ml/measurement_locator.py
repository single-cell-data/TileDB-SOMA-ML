# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from contextlib import contextmanager
from typing import Dict, Generator, Union

import attrs
from tiledbsoma import Experiment, SOMATileDBContext, Measurement


@attrs.define(frozen=True, kw_only=True)
class MeasurementLocator:
    """State required to open the Experiment.

    Serializable across multiple processes.

    Private implementation class.
    """

    uri: str
    measurement_name: str
    tiledb_timestamp_ms: int
    tiledb_config: Dict[str, Union[str, float]]

    @classmethod
    def create(cls, experiment: Experiment, measurement_name: str) -> "MeasurementLocator":
        return MeasurementLocator(
            uri=experiment.uri,
            measurement_name=measurement_name,
            tiledb_timestamp_ms=experiment.tiledb_timestamp_ms,
            tiledb_config=experiment.context.tiledb_config,
        )

    @contextmanager
    def open_experiment(self) -> Generator[Experiment, None, None]:
        context = SOMATileDBContext(tiledb_config=self.tiledb_config)
        yield Experiment.open(
            self.uri, tiledb_timestamp=self.tiledb_timestamp_ms, context=context
        )

    @contextmanager
    def open_measurement(self) -> Generator[Measurement, None, None]:
        with self.open_experiment() as exp:
            yield exp.ms[self.measurement_name]
