# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
import logging
from contextlib import contextmanager
from os.path import splitext
from typing import (
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
from attrs import define, evolve
from somacore import ExperimentAxisQuery
from tiledbsoma import (
    DataFrame,
    Experiment,
    SOMATileDBContext,
    SparseNDArray,
)

from tiledbsoma_ml._distributed import get_distributed_world_rank, get_worker_world_rank
from tiledbsoma_ml._utils import batched, splits
from tiledbsoma_ml.common import NDArrayJoinId

logger = logging.getLogger(f"tiledbsoma_ml.{splitext(__file__)[0]}")


Chunks = List[Tuple[int, ...]]


@define(frozen=True)
class Partition:
    rank: int
    world_size: int
    worker_id: int
    n_workers: int


@define(frozen=True, kw_only=True)
class ObsIDs:
    """State required to open the Experiment.

    Serializable across multiple processes.
    """

    uri: str
    measurement_name: str
    layer_name: Optional[str]
    obs_joinids: NDArrayJoinId
    var_joinids: NDArrayJoinId
    tiledb_timestamp_ms: int
    tiledb_config: Dict[str, Union[str, float]]

    def __getitem__(self, index: int) -> np.int64:
        return cast(np.int64, self.obs_joinids[index])

    @classmethod
    def create(
        cls,
        query: ExperimentAxisQuery,
        layer_name: Optional[str],
    ) -> "ObsIDs":
        exp: Experiment = query.experiment
        obs_joinids = query.obs_joinids().to_numpy()
        var_joinids = query.var_joinids().to_numpy()
        return ObsIDs(
            uri=exp.uri,
            measurement_name=query.measurement_name,
            layer_name=layer_name,
            obs_joinids=obs_joinids,
            var_joinids=var_joinids,
            tiledb_timestamp_ms=exp.tiledb_timestamp_ms,
            tiledb_config=exp.context.tiledb_config,
        )

    def partition(
        self,
        partition: Optional[Partition] = None,
    ) -> "ObsIDs":
        obs_joinids = self.obs_joinids

        if partition:
            rank = partition.rank
            world_size = partition.world_size
            worker_id = partition.worker_id
            n_workers = partition.n_workers
        else:
            world_size, rank = get_distributed_world_rank()
            n_workers, worker_id = get_worker_world_rank()

        gpu_splits = splits(len(obs_joinids), world_size)
        gpu_split = obs_joinids[gpu_splits[rank] : gpu_splits[rank + 1]]

        # Trim all GPU splits to be of equal length (equivalent to a "drop_last"); required for distributed training.
        # TODO: may need to add an option to do padding as well.
        min_len = np.diff(gpu_splits).min()
        assert 0 <= (np.diff(gpu_splits).min() - min_len) <= 1
        gpu_split = gpu_split[:min_len]

        # Partition each GPU split among DataLoader workers
        worker_splits = splits(len(gpu_split), n_workers)
        worker_joinids = gpu_split[
            worker_splits[worker_id] : worker_splits[worker_id + 1]
        ].copy()

        logger.debug(
            f"Partitioned IDs: {rank=}, {world_size=}, {worker_id=}, {n_workers=}"
        )

        return evolve(self, obs_joinids=worker_joinids)

    def sample(
        self,
        frac: float,
        seed: Optional[int] = None,
    ) -> Tuple["ObsIDs", "ObsIDs"]:
        obs_joinids = self.obs_joinids
        n_obs = len(obs_joinids)
        n_acc = round(n_obs * frac)
        if seed is not None:
            np.random.seed(seed)
        acc_idxs = np.random.choice(n_obs, n_acc, replace=False)
        rej_idxs = np.setdiff1d(np.arange(n_obs), acc_idxs)
        acc_joinids = obs_joinids[acc_idxs]
        rej_joinids = obs_joinids[rej_idxs]
        return (
            evolve(self, obs_joinids=acc_joinids),
            evolve(self, obs_joinids=rej_joinids),
        )

    def shuffle_chunks(
        self,
        shuffle_chunk_size: int,
        seed: Optional[int] = None,
    ) -> Chunks:
        shuffle_chunks: List[Tuple[int, ...]] = list(
            batched(self.obs_joinids, shuffle_chunk_size)
        )
        shuffle_rng = np.random.default_rng(seed)
        shuffle_rng.shuffle(shuffle_chunks)
        return shuffle_chunks

    @contextmanager
    def open(self) -> Generator[Tuple[SparseNDArray, DataFrame], None, None]:
        context = SOMATileDBContext(tiledb_config=self.tiledb_config)
        with Experiment.open(
            self.uri, tiledb_timestamp=self.tiledb_timestamp_ms, context=context
        ) as exp:
            yield exp.ms[self.measurement_name].X[self.layer_name], exp.obs
