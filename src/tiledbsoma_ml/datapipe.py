# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

from somacore import ExperimentAxisQuery
from torch.utils.data.dataset import Dataset
from torchdata.datapipes.iter import IterDataPipe

from tiledbsoma_ml.batch_iterable import Batch, BatchIterable


class ExperimentAxisQueryIterDataPipe(
    IterDataPipe[Dataset[Batch]]  # type:ignore[misc]
):
    """A :class:`torchdata.datapipes.iter.IterDataPipe` implementation that loads from a :class:`tiledbsoma.SOMAExperiment`.

    This class is based upon the now-deprecated :class:`torchdata.datapipes` API, and should only be used for
    legacy code. See [GitHub issue #1196](https://github.com/pytorch/data/issues/1196) and the
    TorchData [README](https://github.com/pytorch/data/blob/v0.8.0/README.md) for more information.

    See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

    Lifecycle:
        deprecated
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
        See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

        Lifecycle:
            deprecated
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
        """
        See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

        Lifecycle:
            deprecated
        """
        batch_size = self._exp_iter.batch_size
        for X, obs in self._exp_iter:
            if batch_size == 1:
                X = X[0]  # This is a no-op for `csr_matrix`s
            yield X, obs

    def __len__(self) -> int:
        """
        See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

        Lifecycle:
            deprecated
        """
        return len(self._exp_iter)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        See :class:`tiledbsoma_ml.ExperimentAxisQueryIterableDataset` for more information on using this class.

        Lifecycle:
            deprecated
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
