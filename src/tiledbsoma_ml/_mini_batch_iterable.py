# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
from __future__ import annotations

import logging
import os
from typing import Iterable, Iterator

import attrs
import numpy as np
import pandas as pd
import torch
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

    gpu_shuffle: bool = False
    gpu_shuffle_mode: str = "iobatch"
    device: torch.device | None = None
    seed: int | None = None
    epoch: int = 0

    def _gpu_perm(self, n: int) -> torch.Tensor:
        """Deterministic permutation of range(n) seeded by (seed, epoch, pid)."""
        base = int(self.seed or 0)
        pid = os.getpid()
        mixed = (base * 1315423911 + self.epoch * 2654435761 + pid) & 0xFFFFFFFF

        gen_device = (
            self.device
            if (
                self.device is not None and getattr(self.device, "type", None) == "cuda"
            )
            else "cpu"
        )
        g = torch.Generator(device=gen_device)
        g.manual_seed(mixed)
        return torch.randperm(n, generator=g, device=gen_device)

    def _iter(self) -> Iterator[MiniBatch]:
        batch_size = self.batch_size
        result: MiniBatch | None = None
        for X_io_batch, obs_io_batch in self.io_batch_iter:
            assert X_io_batch.shape[0] == obs_io_batch.shape[0]
            iob_idx = 0  # current offset into io batch
            iob_len = X_io_batch.shape[0]

            # GPU within-IO-batch shuffle (dense only)
            if self.gpu_shuffle and self.gpu_shuffle_mode == "iobatch":
                if self.return_sparse_X:
                    logger.warning(
                        "GPU shuffle requested but return_sparse_X=True; leaving IO-batch order unchanged."
                    )
                else:
                    perm = self._gpu_perm(iob_len)
                    perm_cpu = perm.to("cpu", non_blocking=False).numpy()

                    X_full = X_io_batch.slice_tonumpy(slice(0, iob_len))
                    X_t = torch.from_numpy(X_full)
                    if (
                        self.device is not None
                        and getattr(self.device, "type", None) == "cuda"
                    ):
                        if not X_t.is_pinned():
                            X_t = X_t.pin_memory()  # faster H2D
                        X_t = X_t.to(self.device, non_blocking=True)
                    X_t = X_t.index_select(0, perm).contiguous()
                    X_cpu = X_t.to("cpu", non_blocking=False).numpy()

                    obs_perm = obs_io_batch.iloc[perm_cpu].reset_index(drop=True)

                    # Emit mini-batches from the permuted IO-batch
                    for start in range(0, iob_len, self.batch_size):
                        stop = min(start + self.batch_size, iob_len)
                        yield (
                            X_cpu[start:stop],
                            obs_perm.iloc[start:stop].reset_index(drop=True),
                        )
                    continue  # done with this IO-batch

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

                if (
                    self.gpu_shuffle
                    and self.gpu_shuffle_mode == "minibatch"
                    and not self.return_sparse_X
                ):
                    mb_n = X.shape[0]
                    perm = self._gpu_perm(mb_n)
                    perm_cpu = perm.to("cpu", non_blocking=False).numpy()

                    X_t = torch.from_numpy(X)
                    if (
                        self.device is not None
                        and getattr(self.device, "type", None) == "cuda"
                    ):
                        if not X_t.is_pinned():
                            X_t = X_t.pin_memory()
                        X_t = X_t.to(self.device, non_blocking=True)
                    X_t = X_t.index_select(0, perm).contiguous()
                    X = X_t.to("cpu", non_blocking=False).numpy()
                    obs = obs.iloc[perm_cpu].reset_index(drop=True)

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
