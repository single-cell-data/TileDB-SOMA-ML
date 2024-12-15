# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
#
# conftest.py defines pytest fixtures that are available to all test files.

from pathlib import Path
from typing import Union

import pytest
from tiledbsoma import Experiment, Measurement
from tiledbsoma._collection import Collection

from ._utils import XValueGen, add_dataframe, add_sparse_array


@pytest.fixture
def X_layer_names() -> list[str]:
    return ["raw"]


@pytest.fixture(scope="function")
def soma_experiment(
    tmp_path: Path,
    obs_range: Union[int, range],
    var_range: Union[int, range],
    X_value_gen: XValueGen,
) -> Experiment:
    with Experiment.create((tmp_path / "exp").as_posix()) as exp:
        if isinstance(obs_range, int):
            obs_range = range(obs_range)
        if isinstance(var_range, int):
            var_range = range(var_range)

        add_dataframe(exp, "obs", obs_range)
        ms = exp.add_new_collection("ms")
        rna = ms.add_new_collection("RNA", Measurement)
        add_dataframe(rna, "var", var_range)
        rna_x = rna.add_new_collection("X", Collection)
        add_sparse_array(rna_x, "raw", obs_range, var_range, X_value_gen)

    return Experiment.open((tmp_path / "exp").as_posix())
