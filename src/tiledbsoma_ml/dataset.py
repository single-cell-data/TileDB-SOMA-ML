# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

import logging
from math import ceil
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
from tiledbsoma import ExperimentAxisQuery
from torch.utils.data import IterableDataset

from tiledbsoma_ml._distributed import get_distributed_world_rank, get_worker_world_rank
from tiledbsoma_ml.common import Batch, NDArrayJoinId
from tiledbsoma_ml.gpu_batches import GPUBatches
from tiledbsoma_ml.io_batches import IOBatches
from tiledbsoma_ml.query_ids import QueryIDs

logger = logging.getLogger("tiledbsoma_ml.dataset")


class ExperimentDataset(IterableDataset[Batch]):  # type: ignore[misc]
    """An |IterableDataset| implementation that reads from an |ExperimentAxisQuery|.

    Provides an |Iterator| over |Batch|'s of ``obs`` and ``X`` data. Each |Batch| is a tuple containing an |ndarray| and a
    |pd.DataFrame|.

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

    .. |obs_joinids| replace:: :obj:`~tiledbsoma_ml.ExperimentDataset.obs_joinids`

    When :obj:`__iter__ <.__iter__>` is invoked, |obs_joinids|  goes through several partitioning, shuffling, and
    batching steps, ultimately yielding :class:`GPU batches <tiledbsoma_ml.common.Batch>` (tuples of matched ``X`` and
    ``obs`` rows):

    1. Partitioning (|NDArrayJoinID|):

        .. NOTE: for some reason, `$` math blocks only render if there is at least one `:math:` directive.

        a. GPU-partitioning: if this is one of :math:`N>1` GPU processes (see |get_distributed_world_rank|),
           |obs_joinids| is partitioned so that the $N$ GPUs will each receive the same number of samples (meaning up
           to $N-1$ samples may be dropped). Then, only the partition corresponding to the current GPU is kept, The
           resulting |obs_joinids| is used in subsequent steps.

        b. |DataLoader|-worker partitioning: if this is one of $M>1$ |DataLoader|-worker processes (see
           |get_worker_world_rank|), |obs_joinids| is further split $M$ ways, and only |obs_joinids| corresponding to
           the current process are kept.

    2. Shuffle-chunking (|List|\\[|NDArrayJoinID|\\]): if ``shuffle=True``, |obs_joinids| are broken into "shuffle
       chunks". The chunks are then shuffled amongst themselves (but retain their internal order, at this stage). If
       ``shuffle=False``, one "chunk" is emitted containing all |obs_joinids|.

    3. IO-batching (|Iterable|\\[|IOBatch|\\]): shuffle-chunks are re-grouped into "IO batches" of size
       ``io_batch_size``. If ``shuffle=True``, each |IOBatch| is shuffled, then the corresponding ``X`` and ``obs`` rows
       are fetched from the underlying ``Experiment``.

    4. GPU-batching (|Iterable|\\[|Batch|\\]): |IOBatch| tuples are re-grouped into "GPU batches" of size
       ``batch_size``.

    Shuffling support (in steps 2. and 3.) is enabled with the ``shuffle`` parameter, and should be used in lieu of
    |DataLoader|'s default shuffling functionality. Similarly, ``batch_size`` should be used instead of |DataLoader|'s
    default batching. |experiment_dataloader| is the recommended way to wrap an |ExperimentDataset| in a
    |DataLoader|, as it enforces these constraints while passing through other |DataLoader| args.

    Describing the whole process another way, we read randomly selected groups of ``obs`` coordinates from across all
    |ExperimentAxisQuery| results, concatenate those into an I/O buffer, shuffle the buffer element-wise, fetch the full
    row data (``X`` and ``obs``) for each coordinate, and send that on to PyTorch / the GPU, in batches. The randomness
    of the shuffle is determined by:

      - ``shuffle_chunk_size``: controls the granularity of the global shuffle. ``shuffle_chunk_size=1`` corresponds to
        a full global shuffle, but decreases I/O performance. Larger values cause chunks of rows to be shuffled,
        increasing I/O performance (by taking advantage of data locality in the underlying |Experiment|) but decreasing
        overall randomness of the yielded data.
      - ``io_batch_size``: number of rows to fetch at once (comprised of concatenated shuffle-chunks, and shuffled
        row-wise). Larger values increase shuffle-randomness (by shuffling more "shuffle chunks" together), I/O
        performance, and memory usage.

    Lifecycle:
        experimental
    """

    query_ids: QueryIDs
    """Coordinates (from an |ExperimentAxisQuery|) to iterate over."""
    obs_column_names: Sequence[str]
    """Names of ``obs`` columns to return."""
    batch_size: int
    """Number of rows of ``X`` and ``obs`` data to yield in each |Batch|."""
    io_batch_size: int
    """Number of ``obs``/``X`` rows to fetch together, when reading from the provided |ExperimentAxisQuery|."""
    shuffle: bool
    """Whether to shuffle the ``obs`` and ``X`` data being returned."""
    shuffle_chunk_size: int
    """Number of contiguous rows shuffled as an atomic unit (before later concatenation and shuffling within "IO
    batches")."""
    seed: int
    """Random seed used for shuffling."""
    return_sparse_X: bool
    """When ``True``, return ``X`` data as a |csr_matrix| (by default, return |ndarray|'s)."""
    use_eager_fetch: bool
    """Pre-fetch one "IO batch" and one "mini batch"."""

    def __init__(
        self,
        query: Optional[ExperimentAxisQuery] = None,
        query_ids: Optional[QueryIDs] = None,
        layer_name: Optional[str] = None,
        obs_column_names: Sequence[str] = ("soma_joinid",),
        batch_size: int = 1,
        io_batch_size: int = 2**16,
        shuffle: bool = True,
        shuffle_chunk_size: int = 64,
        seed: int | None = None,
        sample: float | None = None,
        return_sparse_X: bool = False,
        use_eager_fetch: bool = True,
    ):
        """Construct a new |ExperimentDataset|.

        Args:
            query:
                Optional |ExperimentAxisQuery|, defining data to iterate over.
                ``query`` xor ``query_ids`` should be provided.
            query_ids:
                Optional |QueryIDs|, defining data to iterate over.
                ``query`` xor ``query_ids`` should be provided.
            layer_name:
                The name of the X layer to read.
            obs_column_names:
                The names of the ``obs`` columns to return. At least one column name must be specified.
                Default is ``('soma_joinid',)``.
            batch_size:
                The number of rows of ``X`` and ``obs`` data to yield in each |Batch|. When ``batch_size=1`` (the
                default) and ``return_sparse_X=False`` (also default), the yielded |ndarray|'s will have rank 1
                (representing a single rows); larger values of ``batch_size`` (or ``return_sparse_X=True``) will result
                in arrays of rank 2 (multiple rows).
                Note that a ``batch_size`` of 1 allows this ``IterableDataset`` to be used with |DataLoader| batching,
                but higher performance can be achieved by performing batching in this class, and setting the
                ``DataLoader``'s ``batch_size`` parameter to ``None``.
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
                GPU-batching).
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
        super().__init__()

        if query_ids:
            if query:
                raise ValueError("Provide only one of {exp_loc,query}")
            self.original_query_ids = query_ids
        else:
            if not query:
                raise ValueError("Provide one of {exp_loc,query}")
            self.original_query_ids = QueryIDs.create(
                query=query,
                layer_name=layer_name,
            )

        # Anything set in the instance needs to be pickle-able for multi-process DataLoaders
        self.obs_column_names = list(obs_column_names)
        if not self.obs_column_names:
            raise ValueError("Must specify at least one value in `obs_column_names`")

        self.batch_size = batch_size
        self.io_batch_size = io_batch_size
        self.shuffle = shuffle
        self.shuffle_chunk_size = shuffle_chunk_size
        if shuffle:
            # Verify `io_batch_size` is a multiple of `shuffle_chunk_size`
            self.io_batch_size = (
                ceil(io_batch_size / shuffle_chunk_size) * shuffle_chunk_size
            )
            if io_batch_size != self.io_batch_size:
                raise ValueError(
                    f"{io_batch_size=} is not a multiple of {shuffle_chunk_size=}"
                )

        self.seed = (
            seed if seed is not None else np.random.default_rng().integers(0, 2**32 - 1)
        )
        self.sample = sample
        self.sample_inverted = False
        self.sampled_query_ids: QueryIDs | None = None
        if sample is not None:
            acc, _ = self.original_query_ids.sample(abs(sample), seed=seed)
            self.sampled_query_ids = acc

        self.return_sparse_X = return_sparse_X
        self.use_eager_fetch = use_eager_fetch
        self.epoch = 0

    def invert(self) -> None:
        sample = self.sample
        if not sample:
            raise RuntimeError("Can only invert sampled ExperimentDatasets")

        self.sample_inverted = not self.sample_inverted
        acc, rej = self.original_query_ids.sample(abs(sample), seed=self.seed)
        self.sampled_query_ids = rej if self.sample_inverted else acc

    @property
    def query_ids(self) -> QueryIDs:
        return self.sampled_query_ids or self.original_query_ids

    @property
    def measurement_name(self) -> str:
        return self.query_ids.measurement_name

    @property
    def layer_name(self) -> Optional[str]:
        return self.query_ids.layer_name

    @property
    def obs_joinids(self) -> NDArrayJoinId:
        """Return this |ExperimentDataset|'s obs coordinates (possibly partitioned for a GPU process / DataLoader
        worker.

        Returns
        -------
        |NDArrayJoinId|
            A NumPy array of 64-bit integers containing the observation join IDs.
        """
        return self.query_ids.obs_joinids

    @property
    def var_joinids(self) -> NDArrayJoinId:
        return self.query_ids.var_joinids

    def _multiproc_check(self) -> None:
        """Rule out config combinations that are invalid in multiprocess mode."""
        if self.return_sparse_X:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info and worker_info.num_workers > 0:
                raise NotImplementedError(
                    "torch does not work with sparse tensors in multi-processing mode "
                    "(see https://github.com/pytorch/pytorch/issues/20248)"
                )

        world_size, rank = get_distributed_world_rank()
        n_workers, worker_id = get_worker_world_rank()
        logger.debug(
            f"Iterator created {rank=}, {world_size=}, {worker_id=}, {n_workers=}, seed={self.seed}, epoch={self.epoch}"
        )
        if world_size > 1 and self.shuffle and self.seed is None:
            raise ValueError(
                "Experiment requires an explicit `seed` when shuffle is used in a multi-process configuration."
            )

    def __iter__(self) -> Iterator[Batch]:
        """Emit batches of aligned X and obs rows.

        Returns
        -------
        |Iterator|\\[|Batch|\\]
            An iterator that yields |Batch| objects containing aligned X and obs rows.

        Notes
        -----
        Lifecycle
            experimental
        """
        self._multiproc_check()

        io_batch_size = self.io_batch_size
        shuffle = self.shuffle
        batch_size = self.batch_size
        use_eager_fetch = self.use_eager_fetch
        seed = self.seed

        query_ids = self.query_ids.partitioned()
        if shuffle:
            chunks = query_ids.shuffle_chunks(
                shuffle_chunk_size=self.shuffle_chunk_size,
                seed=seed,
            )
        else:
            # In no-shuffle mode, all the `obs_joinids` can be treated as one "shuffle chunk",
            # which IO-batches will stride over.
            chunks = [query_ids.obs_joinids]

        with query_ids.open() as (X, obs):
            io_batches = IOBatches(
                chunks=chunks,
                io_batch_size=io_batch_size,
                obs=obs,
                var_joinids=query_ids.var_joinids,
                X=X,
                obs_column_names=self.obs_column_names,
                seed=seed,
                shuffle=shuffle,
                use_eager_fetch=use_eager_fetch,
            )

            gpu_batches = GPUBatches(
                io_batches=io_batches,
                batch_size=batch_size,
                use_eager_fetch=use_eager_fetch,
                return_sparse_X=self.return_sparse_X,
            )

            for X_batch, obs_batch in gpu_batches:
                if batch_size == 1:
                    # This is a no-op for `csr_matrix`s
                    yield X_batch[0], obs_batch
                else:
                    yield X_batch, obs_batch

        self.epoch += 1

    def __len__(self) -> int:
        """Return the number of batches this iterable will produce. If run in the context of :mod:`torch.distributed` or
        as a multi-process loader (i.e., |DataLoader| instantiated with num_workers > 0), the batch count will reflect
        the size of the data partition assigned to the active process.

        See important caveats in the PyTorch |DataLoader| documentation regarding ``len(dataloader)``, which also apply
        to this class.

        Returns:
            ``int`` (Number of batches).

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
        world_size, rank = get_distributed_world_rank()
        n_workers, worker_id = get_worker_world_rank()
        # Every "distributed" process must receive the same number of "obs" rows; the last ≤world_size may be dropped
        # (see _create_obs_joinids_partition).
        obs_per_proc = len(self.obs_joinids) // world_size
        obs_per_worker, obs_rem = divmod(obs_per_proc, n_workers)
        # obs rows assigned to this worker process
        n_worker_obs = obs_per_worker + bool(worker_id < obs_rem)
        n_batches, rem = divmod(n_worker_obs, self.batch_size)
        # (num batches this worker will produce, num features)
        return n_batches + bool(rem), len(self.var_joinids)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this Data iterator.

        When :attr:`~tiledbsoma_ml.ExperimentDataset.shuffle` is ``True``, this will ensure that all replicas use a different random ordering for each epoch.
        Failure to call this method before each epoch will result in the same data ordering.

        This call must be made before the per-epoch iterator is created.
        """
        self.epoch = epoch

    def __getitem__(self, index: int) -> Batch:
        raise NotImplementedError(
            "`Experiment` can only be iterated - does not support mapping"
        )
