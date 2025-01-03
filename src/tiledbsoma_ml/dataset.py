# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

import logging
from math import ceil
from os.path import splitext
from typing import (
    Iterator,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from somacore import ExperimentAxisQuery
from torch.utils.data import IterableDataset

from tiledbsoma_ml._distributed import (
    get_distributed_world_rank,
    get_worker_world_rank,
)
from tiledbsoma_ml.common import Batch, NDArrayJoinId
from tiledbsoma_ml.gpu_batches import GPUBatches
from tiledbsoma_ml.io_batches import IOBatches
from tiledbsoma_ml.obs_ids import ObsIDs

logger = logging.getLogger(f"tiledbsoma_ml.{splitext(__file__)[0]}")


class ExperimentDataset(IterableDataset[Batch]):  # type: ignore[misc]
    """An :class:`IterableDataset` which reads ``X`` and ``obs`` data from a :class:`tiledbsoma.Experiment`, as
    selected by a user-specified :class:`tiledbsoma.ExperimentAxisQuery`. Each step of the iterator
    produces a batch containing equal-sized ``X`` and ``obs`` data, in the form of a :class:`numpy.ndarray` and
    :class:`pandas.DataFrame`, respectively.

    See :class:`ExperimentDataset` for more details on usage.

    Lifecycle:
        experimental
    """

    def __init__(
        self,
        query: Optional[ExperimentAxisQuery] = None,
        obs_ids: Optional[ObsIDs] = None,
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
        """
        Construct a new ``Experiment``, suitable for use with :class:`torch.utils.data.DataLoader`.

        The resulting iterator will produce a tuple containing associated slices of ``X`` and ``obs`` data, as
        a NumPy :class:`numpy.ndarray` (or optionally, :class:`scipy.sparse.csr_matrix`) and a Pandas
        :class:`pandas.DataFrame`, respectively.

        Args:
            layer_name:
                The name of the X layer to read.
            obs_column_names:
                The names of the ``obs`` columns to return. At least one column name must be specified.
                Default is ``('soma_joinid',)``.
            batch_size:
                The number of rows of ``X`` and ``obs`` data to return in each iteration. Defaults to ``1``. A value of
                ``1`` will result in :class:`torch.Tensor` of rank 1 being returned (a single row); larger values will
                result in :class:`torch.Tensor`s of rank 2 (multiple rows). Note that a ``batch_size`` of 1 allows
                this ``IterableDataset`` to be used with :class:`torch.utils.data.DataLoader` batching, but higher
                performance can be achieved by performing batching in this class, and setting the ``DataLoader``'s
                ``batch_size`` parameter to ``None``.
            shuffle:
                Whether to shuffle the ``obs`` and ``X`` data being returned. Defaults to ``True``.
            io_batch_size:
                The number of ``obs``/``X`` rows to retrieve when reading data from SOMA. This impacts:
                1. Maximum memory utilization, larger values provide better read performance, but require more memory.
                2. The number of rows read prior to shuffling (see the ``shuffle`` parameter for details).
                The default value of 65,536 provides high performance but may need to be reduced in memory-limited hosts
                or when using a large number of :class:`DataLoader` workers.
            shuffle_chunk_size:
                The number of contiguous rows sampled prior to concatenation and shuffling.
                Larger numbers correspond to less randomness, but greater read performance.
                If ``shuffle == False``, this parameter is ignored.
            return_sparse_X:
                If ``True``, will return the ``X`` data as a :class:`scipy.sparse.csr_matrix`. If ``False`` (the
                default), will return ``X`` data as a :class:`numpy.ndarray`.
            seed:
                The random seed used for shuffling. Defaults to ``None`` (no seed). This argument *MUST* be specified
                when using :class:`torch.nn.parallel.DistributedDataParallel` to ensure data partitions are disjoint
                across worker processes.
            use_eager_fetch:
                Fetch the next SOMA chunk of ``obs`` and ``X`` data immediately after a previously fetched SOMA chunk is
                made available for processing via the iterator. This allows network (or filesystem) requests to be made
                in parallel with client-side processing of the SOMA data, potentially improving overall performance at
                the cost of doubling memory utilization. Defaults to ``True``.

        Raises:
            ``ValueError`` on various unsupported or malformed parameter values.

        Lifecycle:
            experimental

        .. warning::
            When using this class in any distributed mode, calling the :meth:`set_epoch` method at
            the beginning of each epoch **before** creating the :class:`DataLoader` iterator
            is necessary to make shuffling work properly across multiple epochs. Otherwise,
            the same ordering will always be used.

            In addition, when using shuffling in a distributed configuration (e.g., ``DDP``), you
            must provide a seed, ensuring that the same shuffle is used across all replicas.
        """

        super().__init__()

        if obs_ids:
            if query:
                raise ValueError("Provide only one of {exp_loc,query}")
            self.original_obs_ids = obs_ids
        else:
            if not query:
                raise ValueError("Provide one of {exp_loc,query}")
            self.original_obs_ids = ObsIDs.create(
                query=query,
                layer_name=layer_name,
            )

        self.obs_column_names = list(obs_column_names)
        self.batch_size = batch_size
        self.io_batch_size = io_batch_size
        self.shuffle = shuffle
        self.return_sparse_X = return_sparse_X
        self.use_eager_fetch = use_eager_fetch
        self.seed = (
            seed if seed is not None else np.random.default_rng().integers(0, 2**32 - 1)
        )
        self.sample = sample
        self.sample_inverted = False
        self.sampled_obs_ids: ObsIDs | None = None
        if sample is not None:
            acc, _ = self.original_obs_ids.sample(abs(sample), seed=seed)
            self.sampled_obs_ids = acc

        self.shuffle_chunk_size = shuffle_chunk_size
        self.epoch = 0

        if shuffle:
            # Verify `io_batch_size` is a multiple of `shuffle_chunk_size`
            self.io_batch_size = (
                ceil(io_batch_size / shuffle_chunk_size) * shuffle_chunk_size
            )
            if io_batch_size != self.io_batch_size:
                raise ValueError(
                    f"{io_batch_size=} is not a multiple of {shuffle_chunk_size=}"
                )

        if not self.obs_column_names:
            raise ValueError("Must specify at least one value in `obs_column_names`")

    def invert(self) -> None:
        sample = self.sample
        if not sample:
            raise RuntimeError("Can only invert sampled ExperimentDatasets")

        self.sample_inverted = not self.sample_inverted
        acc, rej = self.original_obs_ids.sample(abs(sample), seed=self.seed)
        self.sampled_obs_ids = rej if self.sample_inverted else acc

    @property
    def obs_ids(self) -> ObsIDs:
        return self.sampled_obs_ids or self.original_obs_ids

    @property
    def measurement_name(self) -> str:
        return self.obs_ids.measurement_name

    @property
    def layer_name(self) -> Optional[str]:
        return self.obs_ids.layer_name

    @property
    def obs_joinids(self) -> NDArrayJoinId:
        return self.obs_ids.obs_joinids

    @property
    def var_joinids(self) -> NDArrayJoinId:
        return self.obs_ids.var_joinids

    def __iter__(self) -> Iterator[Batch]:
        """Create iterator over query.

        Returns:
            ``iterator``

        Lifecycle:
            experimental
        """
        obs_ids = self.obs_ids.partition()
        io_batch_size = self.io_batch_size
        shuffle = self.shuffle
        shuffle_chunk_size = self.shuffle_chunk_size
        batch_size = self.batch_size
        use_eager_fetch = self.use_eager_fetch
        seed = self.seed
        if shuffle:
            chunks = obs_ids.shuffle_chunks(
                shuffle_chunk_size=shuffle_chunk_size,
                seed=seed,
            )
        else:
            chunks = [tuple(obs_ids.obs_joinids)]

        with obs_ids.open() as (X, obs):
            io_batches = IOBatches(
                chunks=chunks,
                io_batch_size=io_batch_size,
                obs=obs,
                var_joinids=obs_ids.var_joinids,
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
        """Return the number of batches this iterable will produce. If run in the context of :class:`torch.distributed`
        or as a multi-process loader (i.e., :class:`torch.utils.data.DataLoader` instantiated with num_workers > 0), the
        batch count will reflect the size of the data partition assigned to the active process.

        See important caveats in the PyTorch
        [:class:`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        documentation regarding ``len(dataloader)``, which also apply to this class.

        Returns:
            ``int`` (Number of batches).

        Lifecycle:
            experimental
        """
        return self.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the number of batches and features that will be yielded from this :class:`tiledbsoma_ml.Experiment`.

        If used in multiprocessing mode (i.e. :class:`torch.utils.data.DataLoader` instantiated with num_workers > 0),
        the number of batches will reflect the size of the data partition assigned to the active process.

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
        """
        Set the epoch for this Data iterator.

        When :attr:`shuffle=True`, this will ensure that all replicas use a different
        random ordering for each epoch. Failure to call this method before each epoch
        will result in the same data ordering.

        This call must be made before the per-epoch iterator is created.
        """
        self.epoch = epoch

    def __getitem__(self, index: int) -> Batch:
        raise NotImplementedError(
            "`Experiment` can only be iterated - does not support mapping"
        )
