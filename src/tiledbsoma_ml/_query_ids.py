# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
"""Shuffle-chunk and partition (across GPU and DataLoader-worker processes) while reading from a SOMA |Experiment|."""
from __future__ import annotations

import logging
import typing
from typing import (
    List,
    Literal,
    Optional,
    Tuple,
)

import numpy as np
from attrs import define, evolve
from tiledbsoma import ExperimentAxisQuery

from tiledbsoma_ml._common import NDArrayJoinId
from tiledbsoma_ml._utils import batched, splits

logger = logging.getLogger("tiledbsoma_ml._query_ids")


Chunks = List[NDArrayJoinId]
r"""Return-type of |QueryIDs.shuffle_chunks|, |List| of |ndarray|\ s."""
SamplingMethod = Literal["deterministic", "multinomial", "stochastic_rounding"]
r"""Enum arg to |QueryIDs.random_split|:

- ``"deterministic"``: number of each class returned will always be :math:`frac \times N`, rounded to nearest int, e.g.
  ``n=12, fracs=[.7,.3]`` will always produce 8 and 4 elements, resp.
- ``"multinomial"``: each element is assigned to a class independently; no guarantees are made about resulting class
  sizes.
- ``"stochastic_rounding"``: guarantee each class gets assigned at least :math:`\lfloor frac \times N \rfloor` elements.
  The remainder are then distributed so that class-size expected-values match the provided ``fracs``.
"""
SamplingMethods = typing.get_args(SamplingMethod)
"""Possible values of |SamplingMethod|."""


@define(frozen=True)
class Partition:
    rank: int
    """GPU-process rank."""
    world_size: int
    """Number of GPU processes."""
    worker_id: int
    """|DataLoader|-worker index."""
    n_workers: int
    """Number of |DataLoader|-workers (within this GPU process)"""


@define(frozen=True, kw_only=True)
class QueryIDs:
    """Wrapper for obs and var IDs from an |ExperimentAxisQuery|.

    Serializable across multiple processes.
    """

    obs_joinids: NDArrayJoinId
    """``obs`` row coordinates to read."""
    var_joinids: NDArrayJoinId
    """``var`` column coordinates to read."""
    partition: Optional[Partition] = None
    """GPU/Worker-partition info; typically populated by :meth:`partitioned`"""

    @classmethod
    def create(cls, query: ExperimentAxisQuery) -> "QueryIDs":
        """Initialize a |QueryIDs| object from an |ExperimentAxisQuery|."""
        obs_joinids = query.obs_joinids().to_numpy()
        var_joinids = query.var_joinids().to_numpy()
        return QueryIDs(
            obs_joinids=obs_joinids,
            var_joinids=var_joinids,
        )

    def random_split(
        self,
        *fracs: float,
        seed: int | None = None,
        method: SamplingMethod = "stochastic_rounding",
    ) -> Tuple["QueryIDs", ...]:
        """Split this |QueryIDs| into 1 or more |QueryIDs|, randomly sampled according ``fracs``.

        - ``fracs`` must sum to $1$
        - ``seed`` is optional
        - ``method``: see |SamplingMethod| for details
        """
        split_fracs = np.cumsum(fracs)
        assert fracs and np.isclose(split_fracs[-1], 1.0), "Fractions must sum to 1"

        obs_joinids = self.obs_joinids
        n_obs = len(obs_joinids)

        rng = np.random.default_rng(seed)
        shuffled_joinids = rng.permutation(obs_joinids)

        if method == "deterministic":
            split_idxs = np.round(split_fracs * n_obs).astype(int)
        elif method == "multinomial":
            split_idxs = np.cumsum(rng.multinomial(n_obs, np.array(fracs)))
        elif method == "stochastic_rounding":
            fracs_arr = np.array(fracs) * n_obs
            split_bases = fracs_arr.astype(int)
            split_idxs = split_bases.cumsum()
            remainder_fracs = fracs_arr - split_bases
            n = len(fracs)
            remainders = np.zeros(n, dtype=int)
            total_remainders = round(np.sum(remainder_fracs))
            while total_remainders > 0:
                pvals = remainder_fracs / np.sum(remainder_fracs)
                choice = rng.choice(n, p=pvals)
                remainders[choice] += 1
                remainder_fracs[choice] = 0
                total_remainders -= 1
            split_idxs += remainders.cumsum()
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        splits = [
            np.sort(split)
            for split in np.array_split(shuffled_joinids, split_idxs[:-1])
        ]
        return tuple(evolve(self, obs_joinids=split) for split in splits)

    def partitioned(
        self,
        partition: Partition,
    ) -> "QueryIDs":
        """Create a new |QueryIDs| with |obs_joinids| corresponding to a given GPU/worker |Partition|.

        If ``None`` is provided, world size, rank, num workers, and worker ID will be inferred using helper functions
        that read env vars (see |get_distributed_rank_and_world_size|, |get_worker_id_and_num|).

        When ``WORLD_SIZE > 1``, each GPU will receive the same number of samples (meaning up to ``WORLD_SIZE-1``
        samples may be dropped).
        """
        if self.partition:
            raise ValueError(
                f"QueryIDs has already been partitioned ({self.partition})"
            )

        obs_joinids = self.obs_joinids
        rank = partition.rank
        world_size = partition.world_size
        worker_id = partition.worker_id
        n_workers = partition.n_workers

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
        """Divide |obs_joinids| into chunks of size ``shuffle_chunk_size``, and shuffle them.

        Used as a compromise between a full random shuffle (optimal for training performance/convergence) and a
        sequential, un-shuffled traversal (optimal for I/O efficiency).
        """
        shuffle_chunks: Chunks = [
            np.array(chunk) for chunk in batched(self.obs_joinids, shuffle_chunk_size)
        ]
        shuffle_rng = np.random.default_rng(seed)
        shuffle_rng.shuffle(shuffle_chunks)
        return shuffle_chunks
