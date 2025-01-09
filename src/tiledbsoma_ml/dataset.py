# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

import gc
import logging
import os
import time
from contextlib import nullcontext
from math import ceil
from typing import ContextManager, Iterator, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from scipy import sparse
from tiledbsoma import (
    DataFrame,
    Experiment,
    ExperimentAxisQuery,
    IntIndexer,
    SparseNDArray,
)
from torch.utils.data import IterableDataset

from tiledbsoma_ml._csr import CSR_IO_Buffer
from tiledbsoma_ml._distributed import get_distributed_world_rank, get_worker_world_rank
from tiledbsoma_ml._experiment_locator import ExperimentLocator
from tiledbsoma_ml._utils import EagerIterator, batched, splits
from tiledbsoma_ml.common import Batch, NDArrayJoinId, NDArrayNumber

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

    When :obj:`__iter__ <.__iter__>` is invoked, ``obs_joinids``  goes through several partitioning, shuffling, and
    batching steps, ultimately yielding :class:`GPU batches <tiledbsoma_ml.common.Batch>` (tuples of matched ``X`` and
    ``obs`` rows):

    1. Partitioning (|NDArrayJoinID|):

        .. NOTE: for some reason, `$` math blocks only render if there is at least one `:math:` directive.

        a. GPU-partitioning: if this is one of :math:`N>1` GPU processes (see |get_distributed_world_rank|),
           ``obs_joinids`` is partitioned so that the $N$ GPUs will each receive the same number of samples (meaning up
           to $N-1$ samples may be dropped). Then, only the partition corresponding to the current GPU is kept, The
           resulting ``obs_joinids`` is used in subsequent steps.

        b. |DataLoader|-worker partitioning: if this is one of $M>1$ |DataLoader|-worker processes (see
           |get_worker_world_rank|), ``obs_joinids`` is further split $M$ ways, and only ``obs_joinids`` corresponding
           to the current process are kept.

    2. Shuffle-chunking (|List|\\[|NDArrayJoinID|\\]): if ``shuffle=True``, ``obs_joinids`` are broken into "shuffle
       chunks". The chunks are then shuffled amongst themselves (but retain their internal order, at this stage). If
       ``shuffle=False``, one "chunk" is emitted containing all ``obs_joinids``.

    3. IO-batching: shuffle-chunks are re-grouped into "IO batches" of size ``io_batch_size``. If ``shuffle=True``, each
       IO-batch is shuffled, then the corresponding ``X`` and ``obs`` rows are fetched from the underlying
       ``Experiment``.

    4. GPU-batching (|Iterable|\\[|Batch|\\]): IO-batch tuples are re-grouped into "GPU batches" of size ``batch_size``.

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

    layer_name: str
    """``X`` layer to read (within |ExperimentAxisQuery|'s |Measurement|)."""
    obs_column_names: Sequence[str]
    """Names of ``obs`` columns to return."""
    batch_size: int
    """Number of rows of ``X`` and ``obs`` data to yield in each |Batch|."""
    io_batch_size: int
    """Number of ``obs``/``X`` rows to fetch together, when reading from the provided |ExperimentAxisQuery|."""
    shuffle: bool
    """Whether to shuffle the ``obs`` and ``X`` data being returned."""
    shuffle_chunk_size: int
    """Number of contiguous rows shuffled as an atomic unit (before later concatenation/shuffling within IO-batches)."""
    seed: int
    """Random seed used for shuffling."""
    return_sparse_X: bool
    """When ``True``, return ``X`` data as a |csr_matrix| (by default, return |ndarray|'s)."""
    use_eager_fetch: bool
    """Pre-fetch one "IO batch" and one "mini batch"."""

    def __init__(
        self,
        query: ExperimentAxisQuery,
        layer_name: str,
        obs_column_names: Sequence[str] = ("soma_joinid",),
        batch_size: int = 1,
        io_batch_size: int = 2**16,
        shuffle: bool = True,
        shuffle_chunk_size: int = 64,
        seed: int | None = None,
        return_sparse_X: bool = False,
        use_eager_fetch: bool = True,
    ):
        """Construct a new |ExperimentDataset|.

        Args:
            query:
                A |ExperimentAxisQuery|, defining the data to iterate over.
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

        # Anything set in the instance needs to be pickle-able for multi-process DataLoaders
        self.experiment_locator = ExperimentLocator.create(query.experiment)
        self.layer_name = layer_name
        self.measurement_name = query.measurement_name
        self.obs_query = query._matrix_axis_query.obs
        self.var_query = query._matrix_axis_query.var
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
        self._user_specified_seed = seed is not None
        self.return_sparse_X = return_sparse_X
        self.use_eager_fetch = use_eager_fetch

        self._obs_joinids: NDArrayJoinId | None = None
        self._var_joinids: NDArrayJoinId | None = None
        self._initialized = False
        self.epoch = 0

    def _create_obs_joinids_partition(self) -> Iterator[NDArrayJoinId]:
        """Create iterator over obs id chunks with split size of (roughly) io_batch_size.

        As appropriate, will chunk, shuffle and apply partitioning per worker.

        IMPORTANT: in any scenario using :mod:`torch.distributed`, where WORLD_SIZE > 1, this will always partition such
        that each process has the same number of samples. Where the number of ``obs_joinids`` is not evenly divisible by
        the number of processes, the number of joinids will be dropped (dropped ids can never exceed WORLD_SIZE-1).

        Abstractly, the steps taken:
        1. Split the joinids into WORLD_SIZE sections (aka number of GPUS in DDP)
        2. Trim the splits to be of equal length
        3. Chunk and optionally shuffle the chunks
        4. Partition by number of data loader workers (to not generate redundant batches
           in cases where the DataLoader is running with `n_workers>1`).

        Private method.
        """
        assert self._obs_joinids is not None
        obs_joinids: NDArrayJoinId = self._obs_joinids

        # 1. Get the split for the model replica/GPU
        world_size, rank = get_distributed_world_rank()
        _gpu_splits = splits(len(obs_joinids), world_size)
        _gpu_split = obs_joinids[_gpu_splits[rank] : _gpu_splits[rank + 1]]

        # 2. Trim to be all of equal length - equivalent to a "drop_last"
        # TODO: may need to add an option to do padding as well.
        min_len = np.diff(_gpu_splits).min()
        assert 0 <= (np.diff(_gpu_splits).min() - min_len) <= 1
        _gpu_split = _gpu_split[:min_len]

        # 3. Chunk and optionally shuffle chunks
        if self.shuffle:
            assert self.io_batch_size % self.shuffle_chunk_size == 0
            shuffle_split = np.array_split(
                _gpu_split, max(1, ceil(min_len / self.shuffle_chunk_size))
            )

            # Deterministically create RNG - state must be same across all processes, ensuring
            # that the joinid partitions are identical across all processes.
            rng = np.random.default_rng(self.seed + self.epoch + 99)
            rng.shuffle(shuffle_split)
            obs_joinids_chunked = [
                np.concatenate(b)
                for b in batched(
                    shuffle_split, self.io_batch_size // self.shuffle_chunk_size
                )
            ]
        else:
            obs_joinids_chunked = np.array_split(
                _gpu_split, max(1, ceil(len(_gpu_split) / self.io_batch_size))
            )

        # 4. Partition by DataLoader worker
        n_workers, worker_id = get_worker_world_rank()
        obs_splits = splits(len(obs_joinids_chunked), n_workers)
        obs_partition_joinids = obs_joinids_chunked[
            obs_splits[worker_id] : obs_splits[worker_id + 1]
        ].copy()

        if logger.isEnabledFor(logging.DEBUG):
            partition_size = sum([len(chunk) for chunk in obs_partition_joinids])
            logger.debug(
                f"Process {os.getpid()} {rank=}, {world_size=}, {worker_id=}, n_workers={n_workers}, epoch={self.epoch}, {partition_size=}"
            )

        return iter(obs_partition_joinids)

    def _init_once(self, exp: Experiment | None = None) -> None:
        """One-time per worker initialization.

        All operations should be idempotent in order to support pipe reset().

        Private method.
        """
        if self._initialized:
            return

        logger.debug(f"Initializing Experiment (shuffle={self.shuffle})")

        if exp is None:
            # If no user-provided Experiment, open/close it ourselves
            exp_cm: ContextManager[Experiment] = (
                self.experiment_locator.open_experiment()
            )
        else:
            # else, it is caller responsibility to open/close the experiment
            exp_cm = nullcontext(exp)

        with exp_cm as exp:
            with exp.axis_query(
                measurement_name=self.measurement_name,
                obs_query=self.obs_query,
                var_query=self.var_query,
            ) as query:
                self._obs_joinids = query.obs_joinids().to_numpy()
                self._var_joinids = query.var_joinids().to_numpy()

        self._initialized = True

    def __iter__(self) -> Iterator[Batch]:
        """Emit |Batch|'s (aligned ``X`` and ``obs`` rows).

        Returns:
            |Iterator|\\[|Batch|\\]

        Lifecycle:
            experimental
        """

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
        if world_size > 1 and self.shuffle and self._user_specified_seed is None:
            raise ValueError(
                "Experiment requires an explicit `seed` when shuffle is used in a multi-process configuration."
            )

        with self.experiment_locator.open_experiment() as exp:
            self._init_once(exp)
            X = exp.ms[self.measurement_name].X[self.layer_name]
            if not isinstance(X, SparseNDArray):
                raise NotImplementedError(
                    "Experiment only supports X layers which are of type SparseNDArray"
                )

            obs_joinid_iter = self._create_obs_joinids_partition()
            _mini_batch_iter = self._mini_batch_iter(exp.obs, X, obs_joinid_iter)
            if self.use_eager_fetch:
                _mini_batch_iter = EagerIterator(
                    _mini_batch_iter, pool=exp.context.threadpool
                )

            if self.batch_size == 1:
                for X, obs in _mini_batch_iter:
                    X = X[0]  # This is a no-op for `csr_matrix`s
                    yield X, obs
            else:
                yield from _mini_batch_iter

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
        self._init_once()
        assert self._obs_joinids is not None
        assert self._var_joinids is not None
        world_size, rank = get_distributed_world_rank()
        n_workers, worker_id = get_worker_world_rank()
        # Every "distributed" process must receive the same number of "obs" rows; the last ≤world_size may be dropped
        # (see _create_obs_joinids_partition).
        obs_per_proc = len(self._obs_joinids) // world_size
        obs_per_worker, obs_rem = divmod(obs_per_proc, n_workers)
        # obs rows assigned to this worker process
        n_worker_obs = obs_per_worker + bool(worker_id < obs_rem)
        n_batches, rem = divmod(n_worker_obs, self.batch_size)
        # (num batches this worker will produce, num features)
        return n_batches + bool(rem), len(self._var_joinids)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this Data iterator.

        When :attr:`~tiledbsoma_ml.ExperimentDataset.shuffle` is ``True``, this will ensure that all replicas use a
        different random ordering for each epoch. Failure to call this method before each epoch will result in the same
        data ordering.

        This call must be made before the per-epoch iterator is created.
        """
        self.epoch = epoch

    def __getitem__(self, index: int) -> Batch:
        raise NotImplementedError(
            "`Experiment` can only be iterated - does not support mapping"
        )

    def _io_batch_iter(
        self,
        obs: DataFrame,
        X: SparseNDArray,
        obs_joinid_iter: Iterator[NDArrayJoinId],
    ) -> Iterator[Tuple[CSR_IO_Buffer, pd.DataFrame]]:
        """Iterate over IO batches, i.e., SOMA query reads, producing tuples of ``(X: csr_array, obs: DataFrame)``.

        ``obs`` joinids read are controlled by the ``obs_joinid_iter``. Iterator results will be reindexed and shuffled
        (if shuffling enabled).

        Private method.
        """
        assert self._var_joinids is not None

        # Create RNG - does not need to be identical across processes, but use the seed anyway
        # for reproducibility.
        shuffle_rng = np.random.default_rng(self.seed + self.epoch)

        obs_column_names = (
            list(self.obs_column_names)
            if "soma_joinid" in self.obs_column_names
            else ["soma_joinid", *self.obs_column_names]
        )
        var_indexer = IntIndexer(self._var_joinids, context=X.context)

        for obs_coords in obs_joinid_iter:
            st_time = time.perf_counter()
            obs_shuffled_coords = (
                obs_coords if not self.shuffle else shuffle_rng.permuted(obs_coords)
            )
            obs_indexer = IntIndexer(obs_shuffled_coords, context=X.context)
            logger.debug(
                f"Retrieving next SOMA IO batch of length {len(obs_coords)}..."
            )

            # To maximize opportunities for concurrency, when in eager_fetch mode,
            # create the X read iterator first, as the eager iterator will begin
            # the read-ahead immediately. Then proceed to fetch obs DataFrame.
            # This matters most on latent backing stores, e.g., S3.
            #
            X_tbl_iter: Iterator[pa.Table] = X.read(
                coords=(obs_coords, self._var_joinids)
            ).tables()

            def make_io_buffer(
                X_tbl: pa.Table,
                obs_coords: NDArrayJoinId,
                var_coords: NDArrayJoinId,
                obs_indexer: IntIndexer,
            ) -> CSR_IO_Buffer:
                """This function provides a GC after we throw off (large) garbage."""
                m = CSR_IO_Buffer.from_ijd(
                    obs_indexer.get_indexer(X_tbl["soma_dim_0"]),
                    var_indexer.get_indexer(X_tbl["soma_dim_1"]),
                    X_tbl["soma_data"].to_numpy(),
                    shape=(len(obs_coords), len(var_coords)),
                )
                gc.collect(generation=0)
                return m

            _io_buf_iter = (
                make_io_buffer(X_tbl, obs_coords, self._var_joinids, obs_indexer)
                for X_tbl in X_tbl_iter
            )
            if self.use_eager_fetch:
                _io_buf_iter = EagerIterator(_io_buf_iter, pool=X.context.threadpool)

            # Now that X read is potentially in progress (in eager mode), go fetch obs data
            # fmt: off
            obs_io_batch = (
                obs.read(coords=(obs_coords,), column_names=obs_column_names)
                .concat()
                .to_pandas()
                .set_index("soma_joinid")
                .reindex(obs_shuffled_coords, copy=False)
                .reset_index()  # demote "soma_joinid" to a column
                [self.obs_column_names]
            )  # fmt: on

            X_io_batch = CSR_IO_Buffer.merge(tuple(_io_buf_iter))

            del obs_indexer, obs_coords, obs_shuffled_coords, _io_buf_iter
            gc.collect()

            tm = time.perf_counter() - st_time
            logger.debug(
                f"Retrieved SOMA IO batch, took {tm:.2f}sec, {X_io_batch.shape[0]/tm:0.1f} samples/sec"
            )
            yield X_io_batch, obs_io_batch

    def _mini_batch_iter(
        self,
        obs: DataFrame,
        X: SparseNDArray,
        obs_joinid_iter: Iterator[NDArrayJoinId],
    ) -> Iterator[Batch]:
        """Break IO batches into shuffled mini-batch-sized chunks.

        Private method.
        """
        assert self._obs_joinids is not None
        assert self._var_joinids is not None

        io_batch_iter = self._io_batch_iter(obs, X, obs_joinid_iter)
        if self.use_eager_fetch:
            io_batch_iter = EagerIterator(io_batch_iter, pool=X.context.threadpool)

        mini_batch_size = self.batch_size
        result: Tuple[NDArrayNumber, pd.DataFrame] | None = None
        for X_io_batch, obs_io_batch in io_batch_iter:
            assert X_io_batch.shape[0] == obs_io_batch.shape[0]
            assert X_io_batch.shape[1] == len(self._var_joinids)
            iob_idx = 0  # current offset into io batch
            iob_len = X_io_batch.shape[0]

            while iob_idx < iob_len:
                if result is None:
                    # perform zero copy slice where possible
                    X_datum = (
                        X_io_batch.slice_toscipy(
                            slice(iob_idx, iob_idx + mini_batch_size)
                        )
                        if self.return_sparse_X
                        else X_io_batch.slice_tonumpy(
                            slice(iob_idx, iob_idx + mini_batch_size)
                        )
                    )
                    result = (
                        X_datum,
                        obs_io_batch.iloc[
                            iob_idx : iob_idx + mini_batch_size
                        ].reset_index(drop=True),
                    )
                    iob_idx += len(result[1])
                else:
                    # Use any remnant from previous IO batch
                    to_take = min(mini_batch_size - len(result[1]), iob_len - iob_idx)
                    X_datum = (
                        sparse.vstack(
                            [result[0], X_io_batch.slice_toscipy(slice(0, to_take))]
                        )
                        if self.return_sparse_X
                        else np.concatenate(
                            [result[0], X_io_batch.slice_tonumpy(slice(0, to_take))]
                        )
                    )
                    result = (
                        X_datum,
                        pd.concat(
                            [result[1], obs_io_batch.iloc[0:to_take]],
                            # Index `obs_batch` from 0 to N-1, instead of disjoint, concatenated pieces of IO batches'
                            # indices
                            ignore_index=True,
                        ),
                    )
                    iob_idx += to_take

                assert result[0].shape[0] == result[1].shape[0]
                if result[0].shape[0] == mini_batch_size:
                    yield result
                    result = None

        else:
            # yield the remnant, if any
            if result is not None:
                yield result
