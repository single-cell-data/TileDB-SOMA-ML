# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from sys import stdout
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from scipy import sparse
from somacore import AxisQuery
from tiledbsoma import Experiment

from tests._utils import (
    XValueGen,
    assert_array_equal,
    mock_distributed,
    parametrize,
    pytorch_x_value_gen,
)
from tiledbsoma_ml._utils import batched
from tiledbsoma_ml.common import Batch
from tiledbsoma_ml.dataset import ExperimentDataset


ExpectedBatch = Tuple[List[int], pd.DataFrame]
ExpectedBatches = List[ExpectedBatch]
Batches = Tuple[List[Batch], ExpectedBatches | None]


@dataclass
class Case:
    obs_range: int
    batch_size: int
    io_batch_size: int
    expected: List[List[int]] | None = None
    var_range: int = 3
    shuffle_chunk_size: int | None = None
    seed: int | None = None
    X_value_gen: XValueGen = field(default=pytorch_x_value_gen)
    obs_query: AxisQuery | None = None
    gpu: Tuple[int, int] | None = None
    worker: Tuple[int, int] | None = None
    return_sparse_X: bool = False
    use_eager_fetch: bool = True
    # Print observed batches' row idxs and exit; useful for generating "expected" batch values
    debug: bool = False

    def batches(self, experiment: Experiment) -> Batches:
        return batches(experiment=experiment, **asdict(self))


def cases(
    cases: Iterable[Case] | Case,
    sweep_sparse: bool = False,
    sweep_eager: bool = False,
):
    if isinstance(cases, Case):
        cases = [cases]
    if sweep_sparse:
        cases = [
            case
            for case0 in cases
            for case in [
                case0,
                replace(case0, return_sparse_X=not case0.return_sparse_X),
            ]
        ]
    if sweep_eager:
        cases = [
            case
            for case0 in cases
            for case in [
                case0,
                replace(case0, use_eager_fetch=not case0.use_eager_fetch),
            ]
        ]
    return parametrize(
        "case,obs_range,var_range,X_value_gen",
        [(case, case.obs_range, case.var_range, case.X_value_gen) for case in cases],
    )


sweep_cases = partial(cases, sweep_sparse=True, sweep_eager=True)


def batches(
    experiment: Experiment,
    obs_range: int,
    var_range: int,
    io_batch_size: int,
    shuffle_chunk_size: int | None,
    batch_size: int,
    seed: int | None,
    X_value_gen: XValueGen,
    obs_query: AxisQuery | None,
    expected: List[List[int]] | None,
    gpu: Tuple[int, int] | None,
    worker: Tuple[int, int] | None,
    debug: bool,
    use_eager_fetch: bool,
    return_sparse_X: bool,
) -> Batches:
    if gpu or worker:
        distributed_ctx = mock_distributed(*(gpu or (0, 1)), (*worker, seed))
    else:
        distributed_ctx = nullcontext()
    with (
        distributed_ctx,
        experiment.axis_query(
            measurement_name="RNA", obs_query=obs_query
        ) as query,
    ):
        ds = ExperimentDataset(
            query,
            layer_name="raw",
            obs_column_names=["label"],
            batch_size=batch_size,
            io_batch_size=io_batch_size,
            shuffle_chunk_size=shuffle_chunk_size,
            shuffle=shuffle_chunk_size is not None,
            seed=seed,
            use_eager_fetch=use_eager_fetch,
            return_sparse_X=return_sparse_X,
        )

        batches: List[Batch] = list(iter(ds))
        if debug:
            # Print observed batch indices and exit; useful for generating "expected" values for test cases
            out = stdout.write
            out("\n[")
            for idx, (_, obs) in enumerate(batches):
                if idx > 0:
                    out(", ")
                out(f"[{', '.join([ f'{label:>2}' for label in obs.label ])}]")
            out("],\n")
            return [], []

        if expected is not None:
            assert ds.shape == (len(expected), var_range)
            obs = experiment.obs.read().concat().to_pandas()
            X = X_value_gen(range(obs_range), range(var_range)).toarray()
            expected_batches: List[Tuple[List[int], pd.DataFrame]] | None = [
                (
                    [X[row_idx].tolist() for row_idx in row_idx_batch],
                    pd.concat(
                        [obs.loc[[row_idx], ["label"]] for row_idx in row_idx_batch]
                    ).reset_index(drop=True),
                )
                for row_idx_batch in expected
            ]
        else:
            expected_batches = None

    return batches, expected_batches


def check(experiment: Experiment, case: Case):
    """Wrapper around ``check_case`` that "spreads" the ``Case``'s fields."""
    check_case(experiment=experiment, **asdict(case))


def check_case(
    shuffle_chunk_size: int,
    io_batch_size: int,
    batch_size: int,
    return_sparse_X: bool,
    **kwargs,
):
    """Given fields from a ``Case``, and an ``Experiment``, create an ``ExperimentBatchDataset`` and verify that its
    emitted batches match ``Case.expected``."""
    actual_batches, expected_batches = batches(
        shuffle_chunk_size=shuffle_chunk_size,
        io_batch_size=io_batch_size,
        batch_size=batch_size,
        return_sparse_X=return_sparse_X,
        **kwargs,
    )
    assert len(actual_batches) == len(expected_batches)
    for idx, ((a_X, a_obs), (e_X, e_obs)) in enumerate(zip(actual_batches, expected_batches)):
        if return_sparse_X:
            assert isinstance(a_X, sparse.csr_matrix)
            a_X = a_X.toarray()
        if batch_size == 1 and not return_sparse_X:
            assert len(e_X) == 1
            e_X = e_X[0]
        assert_array_equal(a_X, e_X)
        assert_frame_equal(a_obs, e_obs)
        # try:
        # except AssertionError:
    # expected_labels = ", ".join([ ",".join(obs.label.tolist()) for _, obs in expected_batches ])
    # actual_labels = ", ".join([ ",".join(obs.label.tolist()) for _, obs in actual_batches ])
    # raise AssertionError(f"Batch {idx} mismatched:\nExpected:\n{expected_labels}\nActual:\n{actual_labels}")
    # actual_labels = [ obs.label.astype(int).tolist()[0] for _, obs in actual_batches ]
    # io_batches = list(batched(actual_labels, io_batch_size))
    # io_batches_shuffled = ' '.join([
    #     '1'
    #     if any(
    #         max(shuffle_chunk) - min(shuffle_chunk) >= shuffle_chunk_size
    #         for shuffle_chunk in batched(io_batch, shuffle_chunk_size)
    #     )
    #     else '-'
    #     for io_batch in io_batches
    # ])
    # print(f" {io_batches_shuffled}")


# fmt: off
@sweep_cases(
    # When shuffle=False, IO batch sizes make no difference to emitted batches
    Case(obs_range=10, io_batch_size=io_batch_size, batch_size=3, expected=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
    for io_batch_size in [ 2, 6, 10 ]
)
def test_unshuffled(soma_experiment, case):
    check(soma_experiment, case)


@sweep_cases(
    Case(obs_range=10, io_batch_size=2, shuffle_chunk_size=2, batch_size=3, seed=seed, expected=expected)
    for seed, expected in [
        # `shuffle_chunk_size=2` means consecutive integers come in pairs (even across batch divisions)
        (111, [[2, 3, 1], [0, 9, 8], [5, 4, 6], [7]]),
        (222, [[0, 1, 7], [6, 3, 2], [8, 9, 4], [5]]),
    ]
)
def test_10_2_2_3(soma_experiment, case):
    check(soma_experiment, case)


@cases(
    Case(obs_range=30, io_batch_size=6, shuffle_chunk_size=3, batch_size=4, seed=seed, expected=expected)
    for seed, expected in [
        # Batches are size 4, but every 6 consecutive idxs are an "IO batch" consisting of two "shuffled chunks" of size 3:
        (111, [[26,  4,  3, 25], [ 5, 24,  1,  2], [ 0, 29, 27, 28], [17, 22, 15, 21], [23, 16, 14, 18], [20, 12, 13, 19], [ 8,  6,  9, 11], [ 7, 10]]),
        (222, [[25, 26, 27, 24], [28, 29, 17, 15], [21, 23, 22, 16], [ 0, 14,  1,  2], [12, 13, 10,  5], [11,  3,  9,  4], [ 7, 19,  6, 18], [20,  8]]),
    ]
)
def test_30_6_3_4(soma_experiment, case):
    check(soma_experiment, case)


# 11: 13
# 111: 9
# 1111: 12
# 11111: 10
@cases(
    Case(obs_range=10, io_batch_size=4, shuffle_chunk_size=2, batch_size=1, seed=seed, expected=[ [idx] for idx in expected_idxs ])
    for seed, expected_idxs in [
        # IO batches:      |             |
        (111, [ 3, 2, 0, 1 ,  5, 4, 9, 8 ,  7, 6 ]),
        (222, [ 1, 0, 6, 7 ,  9, 8, 2, 3 ,  4, 5 ]),
        (333, [ 4, 7, 5, 6 ,  3, 2, 8, 9 ,  1, 0 ]),
        (444, [ 7, 6, 8, 9 ,  1, 5, 4, 0 ,  3, 2 ]),
        (555, [ 9, 3, 8, 2 ,  6, 5, 7, 4 ,  0, 1 ]),
        (666, [ 0, 6, 1, 7 ,  2, 3, 5, 4 ,  8, 9 ]),
        (777, [ 3, 5, 2, 4 ,  9, 1, 0, 8 ,  7, 6 ]),
        (888, [ 9, 8, 5, 4 ,  6, 2, 3, 7 ,  0, 1 ]),
        (999, [ 5, 4, 2, 3 ,  1, 8, 0, 9 ,  6, 7 ]),
    ]
)
def test_10_4_2_1(soma_experiment, case):
    check(soma_experiment, case)


# 11 + 100
# 118 (59.0%) IO batches with some shuffle chunks interleaved
# 82 (41.0%) IO batches with shuffle chunks all separate
#
# 1111 + 100:
# 127 (63.5%) IO batches with some shuffle chunks interleaved
# 73 (36.5%) IO batches with shuffle chunks all separate
#
# 11111 + 300:
# 398 (66.3%) IO batches with some shuffle chunks interleaved
# 202 (33.7%) IO batches with shuffle chunks all separate
#
# 22222 + 300:
# 402 (67.0%) IO batches with some shuffle chunks interleaved
# 198 (33.0%) IO batches with shuffle chunks all separate
@parametrize("obs_range,var_range,X_value_gen", [(10, 3, pytorch_x_value_gen)])
def test_10_4_2_1_shuffle_audit(soma_experiment):
    cases = [
        Case(obs_range=10, io_batch_size=4, shuffle_chunk_size=2, batch_size=1, seed=22222+seed)
        for seed in range(300)
    ]
    n_shuffle_chunks_interleaved = 0
    n_shuffle_chunks_separate = 0
    for case in cases:
        actual_batches, _ = case.batches(soma_experiment)
        actual_labels = [ obs.label.astype(int).tolist()[0] for _, obs in actual_batches ]
        io_batches = list(batched(actual_labels, case.io_batch_size))
        io_batches_are_shuffled = [
            any(
                max(shuffle_chunk) - min(shuffle_chunk) >= case.shuffle_chunk_size
                for shuffle_chunk in batched(io_batch, case.shuffle_chunk_size)
            )
            for io_batch in io_batches
        ]
        assert len(io_batches_are_shuffled) == 3
        assert not io_batches_are_shuffled[-1]
        io_batches_are_shuffled = io_batches_are_shuffled[:-1]
        n_shuffled_io_batches = sum(io_batches_are_shuffled)
        n_shuffle_chunks_interleaved += n_shuffled_io_batches
        n_shuffle_chunks_separate += len(io_batches_are_shuffled) - n_shuffled_io_batches

    n_io_batches = n_shuffle_chunks_interleaved + n_shuffle_chunks_separate
    print()
    print(f"{n_shuffle_chunks_interleaved} ({n_shuffle_chunks_interleaved / n_io_batches:.1%}) IO batches with some shuffle chunks interleaved")
    print(f"{n_shuffle_chunks_separate} ({n_shuffle_chunks_separate / n_io_batches:.1%}) IO batches with shuffle chunks all separate")


@cases(
    Case(obs_range=3, io_batch_size=3, batch_size=3, **kwargs)
    for kwargs in [
        dict(expected=[[0, 1, 2]]),
        dict(shuffle_chunk_size=3, seed=111, expected=[[1, 0, 2]]),
    ]
)
def test_exactly_one_batch(soma_experiment, case):
    check(soma_experiment, case)


@cases(
    Case(obs_range=6, io_batch_size=3, batch_size=3, obs_query=AxisQuery(coords=([],)), expected=[])
)
def test_empty_query(soma_experiment, case):
    check(soma_experiment, case)


@cases(
    Case(
        # 120 rows, split among 3 GPUs with 2 DataLoader workers each ⇒ 20 indices assigned to each worker.
        # Each worker's shuffled indices are the same, but offset by a multiple of 20.
        obs_range=n_obs, batch_size=4, io_batch_size=10, shuffle_chunk_size=5, seed=333, gpu=(rank, world_size), worker=(worker_id, n_workers),
        expected=[
            [
                # Each GPU is offset by multiples of 40 (`rank * n_obs // world_size`).
                # Each worker is then offset by an additional 0 or 20 (within its GPU's range of 40)
                idx + (rank * n_obs // world_size) + (worker_id * n_obs // world_size // n_workers)
                for idx in batch
            ]
            # This pattern of indices is processed by each worker, beginning from that worker's base index (a multiple of 20)
            # 10-element IO batches (comprised of 2 5-element shuffle-chunks) are visible across the batches
            for batch in [[ 5,  8, 12, 11], [14,  9,  7,  6], [13, 10,  2, 18], [ 3, 16,  4, 17], [ 1, 19, 15,  0]]
        ],
    )
    for n_obs in [120]
    for world_size in [3] for rank in range(world_size)
    for n_workers in [2] for worker_id in range(n_workers)
)
def test_distributed_120_10_5_4(soma_experiment, case):
    check(soma_experiment, case)


@parametrize(
    "obs_range,var_range,X_value_gen",
    [(range(100_000_000, 100_000_003), 3, pytorch_x_value_gen)],
)
@parametrize("use_eager_fetch", [True, False])
def test_soma_joinids(
    soma_experiment: Experiment,
    use_eager_fetch: bool,
):
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

        soma_joinids = np.concatenate([obs["soma_joinid"].to_numpy() for _, obs in ds])
        assert_array_equal(soma_joinids, np.arange(100_000_000, 100_000_003))


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
):
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


@parametrize("obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)])
def test_experiment_axis_query_iterable_error_checks(
    soma_experiment: Experiment,
):
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
