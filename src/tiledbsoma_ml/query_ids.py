# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
"""Shuffle-chunk and partition (across GPU and DataLoader-worker processes) while reading from a SOMA |Experiment|."""

import logging
from contextlib import contextmanager
from typing import (
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from attrs import define, evolve
from tiledbsoma import (
    DataFrame,
    Experiment,
    ExperimentAxisQuery,
    SOMATileDBContext,
    SparseNDArray,
)

from tiledbsoma_ml._distributed import get_distributed_world_rank, get_worker_world_rank
from tiledbsoma_ml._utils import batched, splits
from tiledbsoma_ml.common import NDArrayJoinId

logger = logging.getLogger("tiledbsoma_ml.query_ids")


Chunks = List[NDArrayJoinId]
"""Return-type of :meth:`QueryIDs.shuffle_chunks <tiledbsoma_ml.query_ids.QueryIDs.shuffle_chunks>`."""


@define(frozen=True)
class Partition:
    rank: int
    """GPU-process rank."""
    world_size: int
    """Number of GPU processes."""
    worker_id: int
    """DataLoader-worker index."""
    n_workers: int
    """Number of DataLoader-workers (within this GPU process)"""


@define(frozen=True, kw_only=True)
class QueryIDs:
    """Wrapper for obs and var IDs from an |ExperimentAxisQuery|.

    Serializable across multiple processes.
    """

    uri: str
    """SOMA |Experiment| URI."""
    measurement_name: str
    """SOMA |Experiment| measurement name (containing the ``X`` array to read)"""
    layer_name: Optional[str]
    """``X`` array name to read (within the measurement)"""
    obs_joinids: NDArrayJoinId
    """``obs`` row coordinates to read."""
    var_joinids: NDArrayJoinId
    """``var`` column coordinates to read."""
    tiledb_timestamp_ms: int
    """Timestamp to open the |Experiment| at."""
    tiledb_config: Dict[str, Union[str, float]]
    """Configs to open the |Experiment| with."""
    partition: Optional[Partition] = None
    """GPU/Worker-partition info; typically populated by :meth:`partitioned`"""

    @classmethod
    def create(
        cls,
        query: ExperimentAxisQuery,
        layer_name: Optional[str],
    ) -> "QueryIDs":
        """Initialize a |QueryIDs| object from an |ExperimentAxisQuery| and :attr:`.layer_name`."""
        exp: Experiment = query.experiment
        obs_joinids = query.obs_joinids().to_numpy()
        var_joinids = query.var_joinids().to_numpy()
        return QueryIDs(
            uri=exp.uri,
            measurement_name=query.measurement_name,
            layer_name=layer_name,
            obs_joinids=obs_joinids,
            var_joinids=var_joinids,
            tiledb_timestamp_ms=exp.tiledb_timestamp_ms,
            tiledb_config=exp.context.tiledb_config,
        )

    def partitioned(
        self,
        partition: Optional[Partition] = None,
    ) -> "QueryIDs":
        """Create a new |QueryIDs| with |Q.obs_joinids| corresponding to a given GPU/worker |Partition|.

        If ``None`` is provided, world size, rank, num workers, and worker ID will be inferred using helper functions
        that read env vars (see |get_distributed_world_rank|, |get_worker_world_rank|).

        When ``WORLD_SIZE > 1``, each GPU will receive the same number of samples (meaning up to ``WORLD_SIZE-1``
        samples may be dropped).
        """
        if self.partition:
            raise ValueError(
                f"QueryIDs has already been partitioned ({self.partition})"
            )

        obs_joinids = self.obs_joinids
        if partition:
            rank = partition.rank
            world_size = partition.world_size
            worker_id = partition.worker_id
            n_workers = partition.n_workers
        else:
            world_size, rank = get_distributed_world_rank()
            n_workers, worker_id = get_worker_world_rank()

        partition = Partition(
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            n_workers=n_workers,
        )
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
        return evolve(self, obs_joinids=worker_joinids, partition=partition)

    def shuffle_chunks(
        self,
        shuffle_chunk_size: int,
        seed: Optional[int] = None,
    ) -> Chunks:
        """Divide ``obs_joinids`` into chunks of size ``shuffle_chunk_size``, and shuffle them.

        Used as a compromise between a full random shuffle (optimal for training performance/convergence) and a
        sequential, un-shuffled traversal (optimal for I/O efficiency).
        """
        shuffle_chunks: List[NDArrayJoinId] = [
            np.array(chunk) for chunk in batched(self.obs_joinids, shuffle_chunk_size)
        ]
        shuffle_rng = np.random.default_rng(seed)
        shuffle_rng.shuffle(shuffle_chunks)
        return shuffle_chunks

    @contextmanager
    def open(self) -> Generator[Tuple[SparseNDArray, DataFrame], None, None]:
        """Open the |Experiment| associated with this |QueryIDs|, yield its ``X`` and ``obs`` TileDB-SOMA objects."""
        context = SOMATileDBContext(tiledb_config=self.tiledb_config)
        with Experiment.open(
            self.uri, tiledb_timestamp=self.tiledb_timestamp_ms, context=context
        ) as exp:
            yield exp.ms[self.measurement_name].X[self.layer_name], exp.obs
