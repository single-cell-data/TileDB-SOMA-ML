# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from attrs import define
from scipy import sparse

from tiledbsoma_ml._common import MiniBatch, XBatch, XObsTensors
from tiledbsoma_ml._eager_iter import EagerIterator
from tiledbsoma_ml._io_batch_iterable import IOBatchIterable
from tiledbsoma_ml.encoders import Encoder

logger = logging.getLogger("tiledbsoma_ml._mini_batch_iterable")


@define(frozen=True)
class MiniBatchIterable(Iterable[MiniBatch]):
    """From a (possibly shuffled) |IOBatchIterable|, emit |MiniBatch|'s for consumption by PyTorch."""

    io_batch_iter: IOBatchIterable
    batch_size: int
    encoders: list[Encoder] | None = None
    use_eager_fetch: bool = True
    return_sparse_X: bool = False

    def mini_batches(self) -> Iterator[MiniBatch]:
        batch_size = self.batch_size
        result: tuple[XBatch, pd.DataFrame] | None = None
        for X_io_batch, obs_io_batch in self.io_batch_iter:
            assert X_io_batch.shape[0] == obs_io_batch.shape[0]
            iob_idx = 0  # current offset into io batch
            iob_len = X_io_batch.shape[0]

            while iob_idx < iob_len:
                if result is None:
                    # perform zero copy slice where possible
                    X_datum = (
                        X_io_batch.slice_toscipy(slice(iob_idx, iob_idx + batch_size))
                        if self.return_sparse_X
                        else X_io_batch.slice_tonumpy(
                            slice(iob_idx, iob_idx + batch_size)
                        )
                    )
                    result = (
                        X_datum,
                        obs_io_batch.iloc[iob_idx : iob_idx + batch_size].reset_index(
                            drop=True
                        ),
                    )
                    iob_idx += len(result[1])
                else:
                    # Use any remnant from previous IO batch
                    to_take = min(batch_size - len(result[1]), iob_len - iob_idx)
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

                X, obs = result
                assert X.shape[0] == obs.shape[0]

                if X.shape[0] == batch_size:
                    yield MiniBatch(
                        X=X, obs=obs, batch_size=self.batch_size, encoders=self.encoders
                    )
                    result = None
        else:
            # yield the remnant, if any
            if result is not None:
                X, obs = result
                assert X.shape[0] == obs.shape[0]
                yield MiniBatch(
                    X=X, obs=obs, batch_size=self.batch_size, encoders=self.encoders
                )

    def __iter__(self) -> Iterator[MiniBatch]:
        it = self.mini_batches()
        return EagerIterator(it) if self.use_eager_fetch else it

    def tensors(self) -> Iterator[XObsTensors]:
        return map(lambda b: b.tensors, iter(self))
