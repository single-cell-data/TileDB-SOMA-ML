# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

import gc
import logging
import os
import time
from math import ceil
from os.path import splitext
from typing import (
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sparse
import torch
from somacore import ExperimentAxisQuery
from somacore.query._eager_iter import EagerIterator as _EagerIterator
from tiledbsoma import DataFrame, IntIndexer, SparseNDArray

from tiledbsoma_ml._csr import CSR_IO_Buffer
from tiledbsoma_ml._distributed import (
    get_distributed_world_rank,
    get_worker_world_rank,
)
from tiledbsoma_ml._utils import batched, splits
from tiledbsoma_ml.common import Batch, NDArrayJoinId, NDArrayNumber
from tiledbsoma_ml.partition_ids import PartitionIDs

logger = logging.getLogger(f"tiledbsoma_ml.{splitext(__file__)[0]}")


class BatchIterable(Iterable[Batch]):
    """An :class:`Iterable` which reads ``X`` and ``obs`` data from a :class:`tiledbsoma.Experiment`, as
    selected by a user-specified :class:`tiledbsoma.ExperimentAxisQuery`. Each step of the iterator
    produces a batch containing equal-sized ``X`` and ``obs`` data, in the form of a :class:`numpy.ndarray` and
    :class:`pandas.DataFrame`, respectively.

    Private base class for subclasses of :class:`torch.utils.data.IterableDataset` and
    :class:`torchdata.datapipes.iter.IterDataPipe`. Refer to :class:`ExperimentAxisQueryIterableDataset`
    and :class:`ExperimentAxisQueryIterDataPipe` for more details on usage.

    Lifecycle:
        experimental
    """

    def __init__(
        self,
        partition_ids: Optional[PartitionIDs] = None,
        query: Optional[ExperimentAxisQuery] = None,
        layer_name: Optional[str] = None,
        obs_column_names: Sequence[str] = ("soma_joinid",),
        batch_size: int = 1,
        shuffle: bool = True,
        io_batch_size: int = 2**16,
        shuffle_chunk_size: int = 64,
        return_sparse_X: bool = False,
        seed: int | None = None,
        use_eager_fetch: bool = True,
    ):
        """
        Construct a new ``ExperimentAxisQueryIterable``, suitable for use with :class:`torch.utils.data.DataLoader`.

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

        if partition_ids:
            if query:
                raise ValueError("Provide only one of {exp_loc,query}")
            self.partition_ids = partition_ids
        else:
            if not query:
                raise ValueError("Provide one of {exp_loc,query}")
            self.partition_ids = PartitionIDs.create(
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
        self._user_specified_seed = seed is not None
        self.shuffle_chunk_size = shuffle_chunk_size
        self._initialized = False
        self.epoch = 0

        if self.shuffle:
            # round io_batch_size up to a unit of shuffle_chunk_size to simplify code.
            self.io_batch_size = (
                ceil(io_batch_size / shuffle_chunk_size) * shuffle_chunk_size
            )

        if not self.obs_column_names:
            raise ValueError("Must specify at least one value in `obs_column_names`")

    @property
    def measurement_name(self) -> str:
        return self.partition_ids.measurement_name

    @property
    def layer_name(self) -> Optional[str]:
        return self.partition_ids.layer_name

    @property
    def obs_joinids(self) -> NDArrayJoinId:
        return self.partition_ids.obs_joinids

    @property
    def var_joinids(self) -> NDArrayJoinId:
        return self.partition_ids.var_joinids

    def _create_obs_joinids_partition(self) -> Iterator[NDArrayJoinId]:
        """Create iterator over obs id chunks with split size of (roughly) io_batch_size.

        As appropriate, will chunk, shuffle and apply partitioning per worker.

        IMPORTANT: in any scenario using torch.distributed, where WORLD_SIZE > 1, this will
        always partition such that each process has the same number of samples. Where
        the number of obs_joinids is not evenly divisible by the number of processes,
        the number of joinids will be dropped (dropped ids can never exceed WORLD_SIZE-1).

        Abstractly, the steps taken:
        1. Split the joinids into WORLD_SIZE sections (aka number of GPUS in DDP)
        2. Trim the splits to be of equal length
        3. Chunk and optionally shuffle the chunks
        4. Partition by number of data loader workers (to not generate redundant batches
           in cases where the DataLoader is running with `n_workers>1`).

        Private method.
        """
        obs_joinids = self.obs_joinids

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

    def __iter__(self) -> Iterator[Batch]:
        """Create iterator over query.

        Returns:
            ``iterator``

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
                "ExperimentAxisQueryIterable requires an explicit `seed` when shuffle is used in a multi-process configuration."
            )

        with self.partition_ids.open_experiment() as exp:
            X = exp.ms[self.measurement_name].X[self.layer_name]
            if not isinstance(X, SparseNDArray):
                raise NotImplementedError(
                    "ExperimentAxisQueryIterable only supports X layers which are of type SparseNDArray"
                )

            obs_joinid_iter = self._create_obs_joinids_partition()
            _mini_batch_iter = self._mini_batch_iter(exp.obs, X, obs_joinid_iter)
            if self.use_eager_fetch:
                _mini_batch_iter = _EagerIterator(
                    _mini_batch_iter, pool=exp.context.threadpool
                )

            yield from _mini_batch_iter

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
        """Return the number of batches and features that will be yielded from this :class:`tiledbsoma_ml.ExperimentAxisQueryIterable`.

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
            "`ExperimentAxisQueryIterable` can only be iterated - does not support mapping"
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
        # Create RNG - does not need to be identical across processes, but use the seed anyway
        # for reproducibility.
        shuffle_rng = np.random.default_rng(self.seed + self.epoch)

        obs_column_names = (
            list(self.obs_column_names)
            if "soma_joinid" in self.obs_column_names
            else ["soma_joinid", *self.obs_column_names]
        )
        var_indexer = IntIndexer(self.var_joinids, context=X.context)

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
            X_tbl_iter: Iterator[pa.Table] = X.read(
                coords=(obs_coords, self.var_joinids)
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
                make_io_buffer(X_tbl, obs_coords, self.var_joinids, obs_indexer)
                for X_tbl in X_tbl_iter
            )
            if self.use_eager_fetch:
                _io_buf_iter = _EagerIterator(_io_buf_iter, pool=X.context.threadpool)

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
        io_batch_iter = self._io_batch_iter(obs, X, obs_joinid_iter)
        if self.use_eager_fetch:
            io_batch_iter = _EagerIterator(io_batch_iter, pool=X.context.threadpool)

        mini_batch_size = self.batch_size
        result: Tuple[NDArrayNumber, pd.DataFrame] | None = None
        for X_io_batch, obs_io_batch in io_batch_iter:
            assert X_io_batch.shape[0] == obs_io_batch.shape[0]
            assert X_io_batch.shape[1] == len(self.var_joinids)
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
