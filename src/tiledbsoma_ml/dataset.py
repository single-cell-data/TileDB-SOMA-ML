# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

from somacore import ExperimentAxisQuery
from torch.utils.data import IterableDataset

from tiledbsoma_ml.batch_iterable import Batch, BatchIterable


class ExperimentDataset(IterableDataset[Batch]):  # type:ignore[misc]
    """A :class:`torch.utils.data.IterableDataset` implementation that loads from a :class:`tiledbsoma.SOMAExperiment`.

    This class works seamlessly with :class:`torch.utils.data.DataLoader` to load ``obs`` and ``X`` data as
    specified by a SOMA :class:`tiledbsoma.ExperimentAxisQuery`, providing an iterator over batches of
    ``obs`` and ``X`` data. Each iteration will yield a tuple containing an :class:`numpy.ndarray`
    and a :class:`pandas.DataFrame`.

    For example:

    >>> import torch
    >>> import tiledbsoma
    >>> import tiledbsoma_ml
    >>> with tiledbsoma.Experiment.open("my_experiment_path") as exp:
    ...     with exp.axis_query(measurement_name="RNA", obs_query=tiledbsoma.AxisQuery(value_filter="tissue_type=='lung'")) as query:
    ...         ds = tiledbsoma_ml.ExperimentDataset(query)
    ...         dataloader = torch.utils.data.DataLoader(ds)
    >>> data = next(iter(dataloader))
    >>> data
    (array([0., 0., 0., ..., 0., 0., 0.], dtype=float32),
    soma_joinid
    0     57905025)
    >>> data[0]
    array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)
    >>> data[1]
    soma_joinid
    0     57905025

    The ``batch_size`` parameter controls the number of rows of ``obs`` and ``X`` data that are returned in each
    iteration. If the ``batch_size`` is 1, then each result will have rank 1, else it will have rank 2. A ``batch_size``
    of 1 is compatible with :class:`torch.utils.data.DataLoader`-implemented batching, but it will usually be more
    performant to create mini-batches using this class, and set the ``DataLoader`` batch size to `None`.

    The ``obs_column_names`` parameter determines the data columns that are returned in the ``obs`` DataFrame (the
    default is a single column, containing the ``soma_joinid`` for the ``obs`` dimension).

    The ``io_batch_size`` parameter determines the number of rows read, from which mini-batches are yielded. A
    larger value will increase total memory usage and may reduce average read time per row.

    Shuffling support is enabled with the ``shuffle`` parameter, and will normally be more performant than using
    :class:`DataLoader` shuffling. The shuffling algorithm works as follows:

      1. Rows selected by the query are subdivided into groups of size ``shuffle_chunk_size``, aka a "shuffle chunk".
      2. A random selection of shuffle chunks is drawn and read as a single I/O buffer (of size ``io_buffer_size``).
      3. The entire I/O buffer is shuffled.

    Put another way, we read randomly selected groups of observations from across all query results, concatenate
    those into an I/O buffer, and shuffle the buffer before returning mini-batches. The randomness of the shuffle
    is therefore determined by the ``io_buffer_size`` (number of rows read), and the ``shuffle_chunk_size``
    (number of rows in each draw). Decreasing ``shuffle_chunk_size`` will increase shuffling randomness, and decrease I/O
    performance.

    This class will detect when run in a multiprocessing mode, including multi-worker :class:`torch.utils.data.DataLoader`
    and multi-process training such as :class:`torch.nn.parallel.DistributedDataParallel`, and will automatically partition
    data appropriately. In the case of distributed training, sample partitions across all processes must be equal. Any
    data tail will be dropped.

    Lifecycle:
        experimental
    """

    def __init__(
        self,
        query: ExperimentAxisQuery,
        X_name: str = "raw",
        obs_column_names: Sequence[str] = ("soma_joinid",),
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
        io_batch_size: int = 2**16,
        shuffle_chunk_size: int = 64,
        return_sparse_X: bool = False,
        use_eager_fetch: bool = True,
    ):
        """
        Construct a new ``ExperimentDataset``, suitable for use with :class:`torch.utils.data.DataLoader`.

        The resulting iterator will produce a tuple containing associated slices of ``X`` and ``obs`` data, as
        a NumPy ``ndarray`` (or optionally, :class:`scipy.sparse.csr_matrix`) and a Pandas ``DataFrame`` respectively.

        Args:
            query:
                A :class:`tiledbsoma.ExperimentAxisQuery`, defining the data which will be iterated over.
            X_name:
                The name of the ``X`` layer to read.
            obs_column_names:
                The names of the ``obs`` columns to return. At least one column name must be specified.
                Default is ``('soma_joinid',)``.
            batch_size:
                The number of rows of ``X`` and ``obs`` data to return in each iteration. Defaults to ``1``. A value of
                ``1`` will result in :class:`torch.Tensor` of rank 1 being returned (a single row); larger values will
                result in :class:`torch.Tensor`\ s of rank 2 (multiple rows).

                Note that a ``batch_size`` of 1 allows this ``IterableDataset`` to be used with :class:`torch.utils.data.DataLoader`
                batching, but you will achieve higher performance by performing batching in this class, and setting the ``DataLoader``
                batch_size parameter to ``None``.
            shuffle:
                Whether to shuffle the ``obs`` and ``X`` data being returned. Defaults to ``True``.
            io_batch_size:
                The number of ``obs``/``X`` rows to retrieve when reading data from SOMA. This impacts two aspects of
                this class's behavior: 1) The maximum memory utilization, with larger values providing
                better read performance, but also requiring more memory; 2) The number of rows read prior to shuffling
                (see ``shuffle`` parameter for details). The default value of 131,072 provides high performance, but
                may need to be reduced in memory limited hosts (or where a large number of :class:`DataLoader` workers
                are employed).
            shuffle_chunk_size:
                The number of contiguous rows sampled, prior to concatenation and shuffling.
                Larger numbers correspond to less randomness, but greater read performance.
                If ``shuffle == False``, this parameter is ignored.
            return_sparse_X:
                If ``True``, will return the ``X`` data as a :class:`scipy.sparse.csr_matrix`. If ``False`` (the default), will
                return ``X`` data as a :class:`numpy.ndarray`.
            seed:
                The random seed used for shuffling. Defaults to ``None`` (no seed). This argument *must* be specified when using
                :class:`torch.nn.parallel.DistributedDataParallel` to ensure data partitions are disjoint across worker
                processes.
            use_eager_fetch:
                Fetch the next SOMA chunk of ``obs`` and ``X`` data immediately after a previously fetched SOMA chunk is made
                available for processing via the iterator. This allows network (or filesystem) requests to be made in
                parallel with client-side processing of the SOMA data, potentially improving overall performance at the
                cost of doubling memory utilization. Defaults to ``True``.

        Raises:
            ``ValueError`` on various unsupported or malformed parameter values.

        Lifecycle:
            experimental

        """
        super().__init__()
        self._exp_iter = BatchIterable(
            query=query,
            X_name=X_name,
            obs_column_names=obs_column_names,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            io_batch_size=io_batch_size,
            return_sparse_X=return_sparse_X,
            use_eager_fetch=use_eager_fetch,
            shuffle_chunk_size=shuffle_chunk_size,
        )

    def __iter__(self) -> Iterator[Batch]:
        """Create ``Iterator`` yielding "mini-batch" tuples of :class:`numpy.ndarray` (or :class:`scipy.csr_matrix`) and
        :class:`pandas.DataFrame`.

        Returns:
            ``iterator``

        Lifecycle:
            experimental
        """
        batch_size = self._exp_iter.batch_size
        for X, obs in self._exp_iter:
            if batch_size == 1:
                X = X[0]  # This is a no-op for `csr_matrix`s
            yield X, obs

    def __len__(self) -> int:
        """Return number of batches this iterable will produce.

        See important caveats in the PyTorch
        [:class:`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        documentation regarding ``len(dataloader)``, which also apply to this class.

        Returns:
            ``int`` (number of batches).

        Lifecycle:
            experimental
        """
        return len(self._exp_iter)

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
        return self._exp_iter.shape

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this Data iterator.

        When :attr:`shuffle=True`, this will ensure that all replicas use a different
        random ordering for each epoch. Failure to call this method before each epoch
        will result in the same data ordering.

        This call must be made before the per-epoch iterator is created.

        Lifecycle:
            experimental
        """
        self._exp_iter.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return self._exp_iter.epoch
