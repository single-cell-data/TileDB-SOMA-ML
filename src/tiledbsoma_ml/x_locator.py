# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Generator, Tuple

import attrs
from tiledbsoma import DataFrame, Experiment, SOMATileDBContext, SparseNDArray


@attrs.define(frozen=True, kw_only=True)
class XLocator:
    """State required to open an ``X`` |SparseNDArray| (and associated ``obs`` |DataFrame|), within an |Experiment|.

    Serializable across multiple processes.
    """

    uri: str
    measurement_name: str
    layer_name: str
    tiledb_timestamp_ms: int
    tiledb_config: Dict[str, str | float]

    @classmethod
    def create(
        cls,
        experiment: Experiment,
        measurement_name: str,
        layer_name: str,
    ) -> "XLocator":
        """Initialize an |XLocator| object from an |Experiment|, ``measurement_name``, and ``layer_name``.

        The arguments provide sufficient info to identify a specific ``X`` "layer" in the provided |Experiment|.
        """
        return XLocator(
            uri=experiment.uri,
            measurement_name=measurement_name,
            layer_name=layer_name,
            tiledb_timestamp_ms=experiment.tiledb_timestamp_ms,
            tiledb_config=experiment.context.tiledb_config,
        )

    @contextmanager
    def open(self) -> Generator[Tuple[SparseNDArray, DataFrame], None, None]:
        context = SOMATileDBContext(tiledb_config=self.tiledb_config)
        with Experiment.open(
            self.uri, tiledb_timestamp=self.tiledb_timestamp_ms, context=context
        ) as exp:
            print(exp.ms)
            print(f"first: {exp.ms[self.measurement_name]}")
            print(f"name:{self.measurement_name}", f"layer:{self.layer_name}")
            X = exp.ms[self.measurement_name].X[self.layer_name]
            obs = exp.obs
            if not isinstance(X, SparseNDArray):
                raise NotImplementedError(
                    "Experiment only supports X layers which are of type SparseNDArray"
                )

            yield X, obs
