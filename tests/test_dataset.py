# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tiledbsoma as soma
from pandas._testing import assert_frame_equal
from scipy import sparse
from tiledbsoma import Experiment

from tests._utils import (
    assert_array_equal,
    mock_distributed,
    parametrize,
    pytorch_seq_x_value_gen,
    pytorch_x_value_gen,
)
from tiledbsoma_ml import ExperimentDataset


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@parametrize("return_sparse_X", [True, False])
@parametrize("use_eager_fetch", [True, False])
def test_non_batched(
    soma_experiment: Experiment,
    return_sparse_X: bool,
    use_eager_fetch: bool,
) -> None:
    """Check batches of size 1 (the default)"""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            shuffle=False,
            return_sparse_X=return_sparse_X,
            use_eager_fetch=use_eager_fetch,
        )
        assert ds.shape == (6, 3)
        batch_iter = iter(ds)
        for idx, (X_batch, obs_batch) in enumerate(batch_iter):
            expected_X = [0, idx + 0.1, 0] if idx % 2 == 0 else [idx, 0, idx + 0.2]
            if return_sparse_X:
                assert isinstance(X_batch, sparse.csr_matrix)
                # Sparse batches are always 2D
                assert X_batch.shape == (1, 3)
                assert_array_equal(X_batch.todense(), [expected_X])
            else:
                assert isinstance(X_batch, np.ndarray)
                # Dense single-row batches are "squeezed" down to 1-D
                assert X_batch.shape == (3,)
                assert_array_equal(X_batch, expected_X)

            assert_frame_equal(obs_batch, pd.DataFrame({"label": [str(idx)]}))


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@parametrize("return_sparse_X", [True, False])
@parametrize("use_eager_fetch", [True, False])
def test_uneven_soma_and_result_batches(
    soma_experiment: Experiment,
    return_sparse_X: bool,
    use_eager_fetch: bool,
) -> None:
    """Check that batches are correctly created when they require fetching multiple chunks."""
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            shuffle=False,
            batch_size=3,
            io_batch_size=2,
            return_sparse_X=return_sparse_X,
            use_eager_fetch=use_eager_fetch,
        )
        assert ds.shape == (2, 3)
        batch_iter = iter(ds)

        X_batch, obs_batch = next(batch_iter)
        assert X_batch.shape == (3, 3)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        else:
            assert isinstance(X_batch, np.ndarray)
        assert_array_equal(X_batch, [[0, 0.1, 0], [1, 0, 1.2], [0, 2.1, 0]])
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

        X_batch, obs_batch = next(batch_iter)
        assert X_batch.shape == (3, 3)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        else:
            assert isinstance(X_batch, np.ndarray)
        assert_array_equal(X_batch, [[3, 0, 3.2], [0, 4.1, 0], [5, 0, 5.2]])
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["3", "4", "5"]}))


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@parametrize("return_sparse_X", [True, False])
@parametrize("use_eager_fetch", [True, False])
def test_batching__all_batches_full_size(
    soma_experiment: Experiment,
    return_sparse_X: bool,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            return_sparse_X=return_sparse_X,
            use_eager_fetch=use_eager_fetch,
        )
        batch_iter = iter(ds)
        assert ds.shape == (2, 3)

        X_batch, obs_batch = next(batch_iter)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        assert_array_equal(X_batch, [[0, 0.1, 0], [1, 0, 1.2], [0, 2.1, 0]])
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

        X_batch, obs_batch = next(batch_iter)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        assert_array_equal(X_batch, [[3, 0, 3.2], [0, 4.1, 0], [5, 0, 5.2]])
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["3", "4", "5"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(range(100_000_000, 100_000_003), 3, pytorch_x_value_gen)],
)
@parametrize("use_eager_fetch", [True, False])
def test_soma_joinids(
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["soma_joinid", "label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
        )
        assert ds.shape == (1, 3)

        soma_joinids = np.concatenate(
            [batch[1]["soma_joinid"].to_numpy() for batch in ds]
        )
        assert_array_equal(soma_joinids, np.arange(100_000_000, 100_000_003))


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(5, 3, pytorch_x_value_gen)],
)
@parametrize("return_sparse_X", [True, False])
@parametrize("use_eager_fetch", [True, False])
def test_batching__partial_final_batch_size(
    soma_experiment: Experiment,
    return_sparse_X: bool,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            return_sparse_X=return_sparse_X,
            use_eager_fetch=use_eager_fetch,
        )
        assert ds.shape == (2, 3)
        batch_iter = iter(ds)

        next(batch_iter)
        X_batch, obs_batch = next(batch_iter)
        if return_sparse_X:
            assert isinstance(X_batch, sparse.csr_matrix)
            X_batch = X_batch.todense()
        assert_array_equal(X_batch, [[3, 0, 3.2], [0, 4.1, 0]])
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["3", "4"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(3, 3, pytorch_x_value_gen)],
)
@parametrize("use_eager_fetch", [True, False])
def test_batching__exactly_one_batch(
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            shuffle=False,
            use_eager_fetch=use_eager_fetch,
        )
        assert ds.shape == (1, 3)
        batch_iter = iter(ds)
        X_batch, obs_batch = next(batch_iter)
        assert_array_equal(X_batch, [[0, 0.1, 0], [1, 0, 1.2], [0, 2.1, 0]])
        assert_frame_equal(obs_batch, pd.DataFrame({"label": ["0", "1", "2"]}))

        with pytest.raises(StopIteration):
            next(batch_iter)


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen)],
)
@parametrize("use_eager_fetch", [True, False])
def test_batching__empty_query_result(
    soma_experiment: Experiment,
    use_eager_fetch: bool,
) -> None:
    with soma_experiment.axis_query(
        measurement_name="RNA", obs_query=soma.AxisQuery(coords=([],))
    ) as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            use_eager_fetch=use_eager_fetch,
        )
        assert ds.shape == (0, 3)
        batch_iter = iter(ds)

        with pytest.raises(StopIteration):
            next(batch_iter)


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(10, 1, pytorch_x_value_gen)],
)
@parametrize("use_eager_fetch", [True, False])
def test_batching__partial_soma_batches_are_concatenated(
    soma_experiment: Experiment, use_eager_fetch: bool
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            batch_size=3,
            # Set SOMA batch read size such that PyTorch batches will span the tail and head of two SOMA batches
            io_batch_size=4,
            shuffle_chunk_size=4,
            use_eager_fetch=use_eager_fetch,
        )
        batches = list(ds)
        assert [len(batch[0]) for batch in batches] == [3, 3, 3, 1]


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(6, 3, pytorch_x_value_gen), (7, 3, pytorch_x_value_gen)],
)
@parametrize(
    "world_size,rank",
    [(3, 0), (3, 1), (3, 2), (2, 0), (2, 1)],
)
def test_distributed__returns_data_partition_for_rank(
    soma_experiment: Experiment,
    obs_range: int,
    world_size: int,
    rank: int,
) -> None:
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode, using mocks
    to avoid having to do real PyTorch distributed setup."""

    with (
        mock_distributed(rank=rank, world_size=world_size),
        soma_experiment.axis_query(measurement_name="RNA") as query,
    ):
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["soma_joinid"],
            io_batch_size=2,
            shuffle=False,
        )
        batches = list(iter(ds))
        soma_joinids = np.concatenate(
            [batch[1]["soma_joinid"].to_numpy() for batch in batches]
        )

        expected_joinids = np.array_split(np.arange(obs_range), world_size)[rank][
            0 : obs_range // world_size
        ].tolist()
        assert sorted(soma_joinids) == expected_joinids


# fmt: off
@parametrize(
    "obs_range,var_range,X_value_gen,world_size,num_workers,splits",
    [
        (12, 3, pytorch_x_value_gen, 3, 2, [[0, 2, 4], [4,  6,  8], [ 8, 10, 12]]),
        (13, 3, pytorch_x_value_gen, 3, 2, [[0, 2, 4], [5,  7,  9], [ 9, 11, 13]]),
        (15, 3, pytorch_x_value_gen, 3, 2, [[0, 3, 5], [5,  8, 10], [10, 13, 15]]),
        (16, 3, pytorch_x_value_gen, 3, 2, [[0, 3, 5], [6,  9, 11], [11, 14, 16]]),
        (18, 3, pytorch_x_value_gen, 3, 2, [[0, 3, 6], [6,  9, 12], [12, 15, 18]]),
        (19, 3, pytorch_x_value_gen, 3, 2, [[0, 3, 6], [7, 10, 13], [13, 16, 19]]),
        (20, 3, pytorch_x_value_gen, 3, 2, [[0, 3, 6], [7, 10, 13], [14, 17, 20]]),
        (21, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 7], [7, 11, 14], [14, 18, 21]]),
        (25, 3, pytorch_x_value_gen, 3, 2, [[0, 4, 8], [9, 13, 17], [17, 21, 25]]),
        (27, 3, pytorch_x_value_gen, 3, 2, [[0, 5, 9], [9, 14, 18], [18, 23, 27]]),
    ],
)
# fmt: on
def test_distributed_and_multiprocessing__returns_data_partition_for_rank(
    soma_experiment: Experiment,
    world_size: int,
    num_workers: int,
    splits: list[list[int]],
) -> None:
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode and DataLoader
    multiprocessing mode, using mocks to avoid having to do distributed pytorch setup or real DataLoader
    multiprocessing."""

    for rank in range(world_size):
        proc_splits = splits[rank]
        for worker_id in range(num_workers):
            expected_joinids = list(
                range(proc_splits[worker_id], proc_splits[worker_id + 1])
            )
            with (
                mock_distributed(rank, world_size, (worker_id, num_workers, 1234)),
                soma_experiment.axis_query(measurement_name="RNA") as query,
            ):
                ds = ExperimentDataset(
                    query,
                    layer_name="raw",
                    obs_column_names=["soma_joinid"],
                    io_batch_size=2,
                    shuffle=False,
                )

                batches = list(iter(ds))

                soma_joinids = np.concatenate(
                    [batch[1]["soma_joinid"].to_numpy() for batch in batches]
                ).tolist()

                assert soma_joinids == expected_joinids


@parametrize("obs_range,var_range,X_value_gen", [(16, 1, pytorch_seq_x_value_gen)])
def test__shuffle(soma_experiment: Experiment) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            shuffle=True,
        )

        batches = list(iter(ds))
        assert all(X.shape == (1,) for X, _ in batches)
        soma_joinids = [obs["soma_joinid"].iloc[0] for _, obs in batches]
        X_values = [X[0].item() for X, _ in batches]

        # same elements
        assert set(soma_joinids) == set(range(16))
        # not ordered! (...with a `1/16!` probability of being ordered)
        assert soma_joinids != list(range(16))
        # randomizes X in same order as obs
        # note: X values were explicitly set to match obs_joinids to allow for this simple assertion
        assert X_values == soma_joinids


@parametrize("obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)])
def test_experiment_axis_query_iterable_error_checks(
    soma_experiment: Experiment,
) -> None:
    with soma_experiment.axis_query(measurement_name="RNA") as query:
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            shuffle=True,
        )
        with pytest.raises(NotImplementedError):
            ds[0]

        with pytest.raises(ValueError):
            ExperimentDataset(
                query,
                obs_column_names=(),
                layer_name="raw",
                shuffle=True,
            )
