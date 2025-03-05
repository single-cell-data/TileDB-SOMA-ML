# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import Iterable, Iterator

import attrs
import numpy as np
import pandas as pd
from scipy import sparse

from tiledbsoma_ml._common import MiniBatch
from tiledbsoma_ml._eager_iter import EagerIterator
from tiledbsoma_ml._io_batch_iterable import IOBatchIterable

logger = logging.getLogger("tiledbsoma_ml._mini_batch_iterable")


@attrs.define(frozen=True)
class MiniBatchIterable(Iterable[MiniBatch]):
    """Convert (possibly shuffled) |IOBatchIterable| into |MiniBatch|'s suitable for passing to PyTorch."""

    io_batch_iter: IOBatchIterable
    batch_size: int
    use_eager_fetch: bool = True
    return_sparse_X: bool = False

    def _iter(self) -> Iterator[MiniBatch]:
        batch_size = self.batch_size
        result: MiniBatch | None = None
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
                    yield result
                    result = None
        else:
            # yield the remnant, if any
            if result is not None:
                yield result

    def __iter__(self) -> Iterator[MiniBatch]:
        it = map(self.maybe_squeeze, self._iter())
        return EagerIterator(it) if self.use_eager_fetch else it

    def maybe_squeeze(self, mini_batch: MiniBatch) -> MiniBatch:
        X, obs = mini_batch
        if self.batch_size == 1:
            # This is a no-op for `csr_matrix`s
            return X[0], obs
        else:
            return mini_batch
