# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from attr import evolve
from attrs import define, field
from attrs.validators import gt
from tiledbsoma import ExperimentAxisQuery
from torch.utils.data import IterableDataset

from tiledbsoma_ml._common import MiniBatch
from tiledbsoma_ml._distributed import (
    get_distributed_rank_and_world_size,
    get_worker_id_and_num,
)
from tiledbsoma_ml._io_batch_iterable import IOBatchIterable
from tiledbsoma_ml._mini_batch_iterable import MiniBatchIterable
from tiledbsoma_ml._query_ids import Partition, QueryIDs, SamplingMethod
from tiledbsoma_ml.x_locator import XLocator

logger = logging.getLogger("tiledbsoma_ml.dataset")

DEFAULT_OBS_COLUMN_NAMES = ("soma_joinid",)
DEFAULT_SHUFFLE_CHUNK_SIZE = 64
DEFAULT_IO_BATCH_SIZE = 2**16


@define
class ExperimentDataset(IterableDataset[MiniBatch]):  # type: ignore[misc]
    r"""An |IterableDataset| implementation that reads from an |ExperimentAxisQuery|.

    Provides an |Iterator| over |MiniBatch|\ s of ``obs`` and ``X`` data. Each |MiniBatch| is a tuple containing an
    |ndarray| and a |pd.DataFrame|.

    An |ExperimentDataset| can be passed to |experiment_dataloader| to enable multi-process reading/fetching.

    For example:

    >>> from tiledbsoma import Experiment, AxisQuery
    >>> from tiledbsoma_ml import ExperimentDataset, experiment_dataloader
    >>> with Experiment.open("my_experiment_path") as exp:
    ...     with exp.axis_query(
    ...         measurement_name="RNA",
    ...         obs_query=AxisQuery(value_filter="tissue_type=='lung'")
    ...     ) as query:
    ...         ds = ExperimentDataset(query)
    ...         dl = experiment_dataloader(ds)
    >>> X_batch, obs_batch = next(iter(dl))
    >>> X_batch
    array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)
    >>> obs_batch
    soma_joinid
    0     57905025

    When |__iter__| is invoked, |obs_joinids| goes through several partitioning, shuffling, and batching steps,
    ultimately yielding |mini batches| (tuples of matched ``X`` and ``obs`` rows):

    1. Partitioning (|NDArrayJoinID|):

        .. NOTE: for some reason, the Sphinx mathjax plugin only renders `$` blocks if at least one `:math:` directive is also present.

        a. GPU-partitioning: if this is one of :math:`N>1` GPU processes (see |get_distributed_rank_and_world_size|),
           |obs_joinids| is partitioned so that the $N$ GPUs will each receive the same number of samples (meaning up to
           $N-1$ samples may be dropped). Then, only the partition corresponding to the current GPU is kept, The
           resulting |obs_joinids| is used in subsequent steps.

        b. |DataLoader|-worker partitioning: if this is one of $M>1$ |DataLoader|-worker processes (see
           |get_worker_id_and_num|), |obs_joinids| is further split $M$ ways, and only |obs_joinids| corresponding to
           the current process are kept.

    2. Shuffle-chunking (|List|\[|NDArrayJoinID|\]): if ``shuffle=True``, |obs_joinids| are broken into "shuffle
       chunks". The chunks are then shuffled amongst themselves (but retain their chunk-internal order, at this stage).
       If ``shuffle=False``, one "chunk" is emitted containing all |obs_joinids|.

    3. IO-batching (|Iterable|\[|IOBatch|\]): shuffle-chunks are re-grouped into "IO batches" of size
       ``io_batch_size``. If ``shuffle=True``, each |IOBatch| is shuffled, then the corresponding ``X`` and ``obs`` rows
       are fetched from the underlying ``Experiment``.

    4. Mini-batching (|Iterable|\[|MiniBatch|\]): |IOBatch| tuples are re-grouped into "mini batches" of size
       ``batch_size``.

    Shuffling support (in steps 2. and 3.) is enabled with the ``shuffle`` parameter, and should be used in lieu of
    |DataLoader|'s default shuffling functionality. Similarly, |batch_size| should be used instead of |DataLoader|'s
    default batching. |experiment_dataloader| is the recommended way to wrap an |ExperimentDataset| in a
    |DataLoader|, as it enforces these constraints while passing through other |DataLoader| args.

    Describing the whole process another way, we read randomly selected groups of ``obs`` coordinates from across all
    |ExperimentAxisQuery| results, concatenate those into an I/O buffer, shuffle the buffer element-wise, fetch the full
    row data (``X`` and ``obs``) for each coordinate, and send that on to PyTorch / the GPU, in mini-batches. The
    randomness of the shuffle is determined by:

      - |shuffle_chunk_size|: controls the granularity of the global shuffle. ``shuffle_chunk_size=1`` corresponds to
        a full global shuffle, but decreases I/O performance. Larger values cause chunks of rows to be shuffled,
        increasing I/O performance (by taking advantage of data locality in the underlying |Experiment|) but decreasing
        overall randomness of the yielded data.
      - |io_batch_size|: number of rows to fetch at once (comprised of concatenated shuffle-chunks, and shuffled
        row-wise). Larger values increase shuffle-randomness (by shuffling more "shuffle chunks" together), I/O
        performance, and memory usage.

    Lifecycle:
        experimental
    """

    # Core data fields
    x_locator: XLocator = field()
    """State required to open an ``X`` |SparseNDArray| (and associated ``obs`` |DataFrame|), within an |Experiment|."""
    query_ids: QueryIDs = field()
    """``obs``/``var`` coordinates (from an |ExperimentAxisQuery|) to iterate over."""
    obs_column_names: List[str] = field(factory=lambda: [*DEFAULT_OBS_COLUMN_NAMES])
    """Names of ``obs`` columns to return."""

    # Configuration fields with defaults
    batch_size: int = field(default=1, validator=gt(0))
    """Number of rows of ``X`` and ``obs`` data to yield in each |MiniBatch|."""
    io_batch_size: int = field(default=DEFAULT_IO_BATCH_SIZE, validator=gt(0))
    """Number of ``obs``/``X`` rows to fetch together, when reading from the provided |ExperimentAxisQuery|."""
    shuffle: bool = field(default=True)
    """Whether to shuffle the ``obs`` and ``X`` data being returned."""
    shuffle_chunk_size: int = field(default=DEFAULT_SHUFFLE_CHUNK_SIZE, validator=gt(0))
    r"""Number of contiguous rows shuffled as an atomic unit (before later concatenation and shuffling within |IOBatch|\
    s)."""
    seed: Optional[int] = field(default=None)
    """Random seed used for shuffling."""
    return_sparse_X: bool = field(default=False)
    r"""When ``True``, return ``X`` data as a |csr_matrix| (by default, return |ndarray|\ s)."""
    use_eager_fetch: bool = field(default=True)
    """Pre-fetch one "IO batch" and one "mini batch"."""

    # Internal state
    epoch: int = field(default=0, init=False)
    rank: int = field(init=False)
    world_size: int = field(init=False)

    def __init__(
        self,
        query: ExperimentAxisQuery | None = None,
        layer_name: str | None = None,
        x_locator: XLocator | None = None,
        query_ids: QueryIDs | None = None,
        obs_column_names: Sequence[str] = DEFAULT_OBS_COLUMN_NAMES,
        batch_size: int = 1,
        io_batch_size: int = DEFAULT_IO_BATCH_SIZE,
        shuffle: bool = True,
        shuffle_chunk_size: int = DEFAULT_SHUFFLE_CHUNK_SIZE,
        seed: Optional[int] = None,
        return_sparse_X: bool = False,
        use_eager_fetch: bool = True,
    ):
        r"""Construct a new |ExperimentDataset|.

        Args:
            query:
                |ExperimentAxisQuery| defining data to iterate over.
                This constructor requires `{query,layer_name}` xor `{x_locator,query_ids}`.
            layer_name:
                ``X`` layer to read.
                This constructor requires `{query,layer_name}` xor `{x_locator,query_ids}`.
            x_locator:
                |XLocator| pointing to an ``X`` array to read.
                This constructor requires `{query,layer_name}` xor `{x_locator,query_ids}`.
            query_ids:
                |QueryIDs| containing ``obs`` and ``var`` joinids to read.
                This constructor requires `{query,layer_name}` xor `{x_locator,query_ids}`.
            obs_column_names:
                The names of the ``obs`` columns to return. At least one column name must be specified.
                Default is ``('soma_joinid',)``.
            batch_size:
                The number of rows of ``X`` and ``obs`` data to yield in each |MiniBatch|. When |batch_size| is 1 (the
                default) and |return_sparse_X| is ``False`` (also default), the yielded |ndarray|\ s will have rank 1
                (representing a single row); larger values of |batch_size| (or |return_sparse_X| is ``True``) will
                result in arrays of rank 2 (multiple rows).
                Note that a |batch_size| of 1 allows this |IterableDataset| to be used with |DataLoader| batching, but
                higher performance can be achieved by performing batching in this class, and setting the |DataLoader|\ s
                |batch_size| parameter to ``None``.
            io_batch_size:
                The number of ``obs``/``X`` rows to retrieve when reading data from SOMA. This impacts:

                1. Maximum memory utilization, larger values provide better read performance, but require more memory.
                2. The number of rows read prior to shuffling (see the ``shuffle`` parameter for details).

                The default value of 65,536 provides high performance but may need to be reduced in memory-limited hosts
                or when using a large number of |DataLoader| workers.
            shuffle:
                Whether to shuffle the ``obs`` and ``X`` data being returned. Defaults to ``True``.
            shuffle_chunk_size:
                Global-shuffle granularity; larger numbers correspond to less randomness, but greater read performance.
                "Shuffle chunks" are contiguous rows in the underlying ``Experiment``, and are shuffled among themselves
                before being combined into IO batches (which are internally shuffled, before fetching and finally
                mini-batching).
                If ``shuffle == False``, this parameter is ignored.
            seed:
                The random seed used for shuffling. Defaults to ``None`` (no seed). This argument *MUST* be specified
                when using |DistributedDataParallel| to ensure data partitions are disjoint across worker processes.
            return_sparse_X:
                If ``True``, will return the ``X`` data as a |csr_matrix|. If ``False`` (the default), will return ``X``
                data as a |ndarray|.
            use_eager_fetch:
                Fetch the next SOMA chunk of ``obs`` and ``X`` data immediately after a previously fetched SOMA chunk is
                made available for processing via the iterator. This allows network (or filesystem) requests to be made
                in parallel with client-side processing of the SOMA data, potentially improving overall performance at
                the cost of doubling memory utilization. Defaults to ``True``.

        Raises:
            ValueError: on unsupported or malformed parameter values.

        Lifecycle:
            experimental

        .. warning::
            When using this class in any distributed mode, calling the :meth:`set_epoch` method at the beginning of each
            epoch **before** creating the |DataLoader| iterator is necessary to make shuffling work properly across
            multiple epochs. Otherwise, the same ordering will always be used.

            In addition, when using shuffling in a distributed configuration (e.g., ``DDP``), you must provide a seed,
            ensuring that the same shuffle is used across all replicas.
        """
        if query and layer_name:
            if x_locator or query_ids:
                raise ValueError(
                    "Expected `{query,layer_name}` xor `{x_locator,query_ids}`"
                )
            query_ids = QueryIDs.create(query=query)
            x_locator = XLocator.create(
                query.experiment,
                measurement_name=query.measurement_name,
                layer_name=layer_name,
            )
        elif x_locator and query_ids:
            if query or layer_name:
                raise ValueError(
                    "Expected `{query,layer_name}` xor `{x_locator,query_ids}`"
                )
        else:
            raise ValueError(
                "Expected `{query,layer_name}` xor `{x_locator,query_ids}`"
            )

        self.__attrs_init__(
            x_locator=x_locator,
            query_ids=query_ids,
            obs_column_names=list(obs_column_names),
            batch_size=batch_size,
            io_batch_size=io_batch_size,
            shuffle=shuffle,
            shuffle_chunk_size=shuffle_chunk_size,
            seed=seed,
            return_sparse_X=return_sparse_X,
            use_eager_fetch=use_eager_fetch,
        )

    def __attrs_post_init__(self) -> None:
        """Validate configuration and initialize distributed state."""
        obs_column_names = self.obs_column_names
        if not obs_column_names:
            raise ValueError("Must specify at least one value in `obs_column_names`")

        if self.shuffle:
            # Verify `io_batch_size` is a multiple of `shuffle_chunk_size`
            if self.io_batch_size % self.shuffle_chunk_size:
                raise ValueError(
                    f"{self.io_batch_size=} is not a multiple of {self.shuffle_chunk_size=}"
                )

        if self.seed is None:
            object.__setattr__(
                self, "seed", np.random.default_rng().integers(0, 2**32 - 1)
            )

        # Set distributed state
        rank, world_size = get_distributed_rank_and_world_size()
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "world_size", world_size)

    @property
    def measurement_name(self) -> str:
        return self.x_locator.measurement_name

    @property
    def layer_name(self) -> Optional[str]:
        return self.x_locator.layer_name

    def random_split(
        self,
        *fracs: float,
        seed: Optional[int] = None,
        method: SamplingMethod = "stochastic_rounding",
    ) -> Tuple[ExperimentDataset, ...]:
        r"""Split this |ExperimentDataset| into 1 or more |ExperimentDataset|\ 's, randomly sampled according ``fracs``.

        - ``fracs`` must sum to $1$
        - ``seed`` is optional
        - ``method``: see |SamplingMethod| for details
        """
        split_query_ids = self.query_ids.random_split(*fracs, seed=seed, method=method)
        return tuple(evolve(self, query_ids=q) for q in split_query_ids)

    def _multiproc_check(self) -> None:
        """Rule out config combinations that are invalid in multiprocess mode."""
        if self.return_sparse_X:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info and worker_info.num_workers > 0:
                raise NotImplementedError(
                    "torch does not work with sparse tensors in multi-processing mode "
                    "(see https://github.com/pytorch/pytorch/issues/20248)"
                )

        rank, world_size = get_distributed_rank_and_world_size()
        worker_id, n_workers = get_worker_id_and_num()
        logger.debug(
            f"Iterator created {rank=}, {world_size=}, {worker_id=}, {n_workers=}, seed={self.seed}, epoch={self.epoch}"
        )
        if world_size > 1 and self.shuffle and self.seed is None:
            raise ValueError(
                "Experiment requires an explicit `seed` when shuffle is used in a multi-process configuration."
            )

    def __iter__(self) -> Iterator[MiniBatch]:
        r"""Emit |MiniBatch|\ s (aligned ``X`` and ``obs`` rows).

        Returns:
            |Iterator|\[|MiniBatch|\]

        Lifecycle:
            experimental
        """
        self._multiproc_check()

        worker_id, n_workers = get_worker_id_and_num()
        partition = Partition(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=worker_id,
            n_workers=n_workers,
        )
        query_ids = self.query_ids.partitioned(partition)
        if self.shuffle:
            chunks = query_ids.shuffle_chunks(
                shuffle_chunk_size=self.shuffle_chunk_size,
                seed=self.seed,
            )
        else:
            # In no-shuffle mode, all the `obs_joinids` can be treated as one "shuffle chunk",
            # which IO-batches will stride over.
            chunks = [query_ids.obs_joinids]

        with self.x_locator.open() as (X, obs):
            io_batch_iter = IOBatchIterable(
                chunks=chunks,
                io_batch_size=self.io_batch_size,
                obs=obs,
                var_joinids=query_ids.var_joinids,
                X=X,
                obs_column_names=self.obs_column_names,
                seed=self.seed,
                shuffle=self.shuffle,
                use_eager_fetch=self.use_eager_fetch,
            )

            yield from MiniBatchIterable(
                io_batch_iter=io_batch_iter,
                batch_size=self.batch_size,
                use_eager_fetch=self.use_eager_fetch,
                return_sparse_X=self.return_sparse_X,
            )

        self.epoch += 1

    def __len__(self) -> int:
        """Return the number of batches this iterable will produce. If run in the context of |torch.distributed| or as a
        multi-process loader (i.e., |DataLoader| instantiated with num_workers > 0), the batch count will reflect the
        size of the data partition assigned to the active process.

        See important caveats in the PyTorch |DataLoader| documentation regarding ``len(dataloader)``, which also apply
        to this class.

        Returns:
            ``int`` (number of batches).

        Lifecycle:
            experimental
        """
        return self.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the number of batches and features that will be yielded from this |Experiment|.

        If used in multiprocessing mode (i.e. |DataLoader| instantiated with num_workers > 0), the number of batches
        will reflect the size of the data partition assigned to the active process.

        Returns:
            A tuple of two ``int`` values: number of batches, number of vars.

        Lifecycle:
            experimental
        """
        rank, world_size = get_distributed_rank_and_world_size()
        worker_id, n_workers = get_worker_id_and_num()
        # Every "distributed" process must receive the same number of "obs" rows; the last ≤world_size may be dropped
        # (see _create_obs_joinids_partition).
        obs_per_proc = len(self.query_ids.obs_joinids) // world_size
        obs_per_worker, obs_rem = divmod(obs_per_proc, n_workers)
        # obs rows assigned to this worker process
        n_worker_obs = obs_per_worker + bool(worker_id < obs_rem)
        n_batches, rem = divmod(n_worker_obs, self.batch_size)
        # (num batches this worker will produce, num features)
        return n_batches + bool(rem), len(self.query_ids.var_joinids)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this Data iterator.

        When :attr:`~tiledbsoma_ml.ExperimentDataset.shuffle` is ``True``, this will ensure that all replicas use a
        different random ordering for each epoch. Failure to call this method before each epoch will result in the same
        data ordering across all epochs.

        This call must be made before the per-epoch iterator is created.
        """
        self.epoch = epoch

    def __getitem__(self, index: int) -> MiniBatch:
        raise NotImplementedError(
            "`Experiment` can only be iterated - does not support mapping"
        )
