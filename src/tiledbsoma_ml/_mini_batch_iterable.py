# # Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
# #
# # Licensed under the MIT License.
# from __future__ import annotations

# import logging
# from typing import Iterable, Iterator, Optional
# import time, os

# import attrs
# import numpy as np
# import pandas as pd
# from scipy import sparse
# import torch

# from tiledbsoma_ml._common import MiniBatch
# from tiledbsoma_ml._eager_iter import EagerIterator
# from tiledbsoma_ml._io_batch_iterable import IOBatchIterable

# logger = logging.getLogger("tiledbsoma_ml._mini_batch_iterable")


# @attrs.define(frozen=True)
# class MiniBatchIterable(Iterable[MiniBatch]):
#     """Convert (possibly shuffled) |IOBatchIterable| into |MiniBatch|'s suitable for passing to PyTorch."""

#     io_batch_iter: IOBatchIterable
#     batch_size: int
#     use_eager_fetch: bool = True
#     return_sparse_X: bool = False

#     # GPU Shuffle Controls
#     gpu_shuffle: bool = False
#     device: Optional[torch.device] = None
#     seed: Optional[int] = None
#     epoch: int = 0

#     def _gpu_perm(self, n: int) -> torch.Tensor:
#         """Device Side Permutation Tensor with seed + epoch for determinism"""
#         gen = torch.Generator(device=self.device if (self.device and self.device.type == "cuda") else "cpu")
#         if self.seed is not None:
#             gen.manual_seed(int(self.seed) + int(self.epoch))
#         return torch.randperm(n, generator=gen, device=self.device if (self.device and self.device.type == "cuda") else "cpu")

#     def _iter(self) -> Iterator[MiniBatch]:
#         """Yield MiniBatches while printing timing diagnostics."""
#         batch_size = self.batch_size
#         result: MiniBatch | None = None
#         pid = os.getpid()            # helpful when several workers interleave prints
#         mb_counter = 0               # running index of mini-batches produced

#         for X_io_batch, obs_io_batch in self.io_batch_iter:
#             t_iob_start = time.perf_counter()

#             iob_idx = 0
#             iob_len = X_io_batch.shape[0]

#             # GPU Shuffle Path
#             if self.gpu_shuffle:
#                 if self.return_sparse_X:
#                     logger.warning(
#                         "GPU shuffle requested but return_sparse_X=True; "
#                         "leaving IO-batch row order unchanged for this batch."
#                     )
#                 else:
#                     # Build a GPU-side permutation
#                     perm = self._gpu_perm(iob_len)                     # on device
#                     perm_cpu = perm.to("cpu", non_blocking=True).numpy()

#                     # Materialize full dense IO-batch once, then permute ON GPU
#                     X_full = X_io_batch.slice_tonumpy(slice(0, iob_len))  
#                     X_t = torch.from_numpy(X_full)
#                     if self.device is not None:
#                         X_t = X_t.to(self.device, non_blocking=True)       # move to GPU
#                     X_t = X_t.index_select(0, perm)                         # row reordering on device

#                     # Reorder obs on CPU to match
#                     obs_perm = obs_io_batch.iloc[perm_cpu].reset_index(drop=True)

#                     # Emit mini-batches directly from the permuted tensor
#                     for start in range(0, iob_len, batch_size):
#                         stop = min(start + batch_size, iob_len)
#                         mb_counter += 1
#                         yield (X_t[start:stop], obs_perm.iloc[start:stop].reset_index(drop=True))
#                     # Done with this IO batch
#                     continue

#             # print(f"[PID {pid}] ▼ received IO-batch of {iob_len:,} rows")
#             while iob_idx < iob_len:
#                 # -----------------------------------------------------------------
#                 #  initialise per-iteration timing buckets
#                 # -----------------------------------------------------------------
#                 t0              = time.perf_counter()
#                 slice_ms        = 0.0   # time to slice X from the IO-batch
#                 obs_slice_ms    = 0.0   # time to slice obs rows
#                 stack_ms        = 0.0   # time spent in vstack / concatenate
#                 obs_cat_ms      = 0.0   # time spent in pd.concat

#                 # ------- build (or extend) the in-progress mini-batch ------------
#                 if result is None:
#                     # ---- X slice ----
#                     t1 = time.perf_counter()
#                     X_datum = (
#                         X_io_batch.slice_toscipy(slice(iob_idx, iob_idx + batch_size))
#                         if self.return_sparse_X
#                         else X_io_batch.slice_tonumpy(
#                             slice(iob_idx, iob_idx + batch_size)
#                         )
#                     )
#                     t2 = time.perf_counter()
#                     slice_ms = (t2 - t1) * 1_000

#                     # ---- obs slice ----
#                     t3 = time.perf_counter()
#                     obs_datum = obs_io_batch.iloc[iob_idx : iob_idx + batch_size].reset_index(
#                         drop=True
#                     )
#                     t4 = time.perf_counter()
#                     obs_slice_ms = (t4 - t3) * 1_000

#                     result = (X_datum, obs_datum)
#                     iob_idx += len(result[1])

#                 else:
#                     to_take = min(batch_size - len(result[1]), iob_len - iob_idx)

#                     # ---- X slice ----
#                     t1 = time.perf_counter()
#                     X_part = (
#                         X_io_batch.slice_toscipy(slice(0, to_take))
#                         if self.return_sparse_X
#                         else X_io_batch.slice_tonumpy(slice(0, to_take))
#                     )
#                     t2 = time.perf_counter()
#                     slice_ms = (t2 - t1) * 1_000

#                     # ---- X stack / concatenate ----
#                     t3 = time.perf_counter()
#                     X_datum = (
#                         sparse.vstack([result[0], X_part])
#                         if self.return_sparse_X
#                         else np.concatenate([result[0], X_part])
#                     )
#                     t4 = time.perf_counter()
#                     stack_ms = (t4 - t3) * 1_000

#                     # ---- obs concat ----
#                     t5 = time.perf_counter()
#                     obs_datum = pd.concat(
#                         [result[1], obs_io_batch.iloc[0:to_take]],
#                         ignore_index=True,
#                     )
#                     t6 = time.perf_counter()
#                     obs_cat_ms = (t6 - t5) * 1_000

#                     result = (X_datum, obs_datum)
#                     iob_idx += to_take

#                 # -----------------------------------------------------------------
#                 #  summary print for THIS loop iteration (may be < batch_size big)
#                 # -----------------------------------------------------------------
#                 pid = os.getpid()
#                 # print(
#                 #     f"[PID {pid}]   slice {slice_ms:7.1f} ms  "
#                 #     f"obs_slice {obs_slice_ms:7.1f} ms  "
#                 #     f"stack {stack_ms:7.1f} ms  "
#                 #     f"obs_cat {obs_cat_ms:7.1f} ms"
#                 # )

#                 X, obs = result
#                 if X.shape[0] == batch_size:
#                     # print(
#                     #     f"[PID {pid}]   • mini-batch {mb_counter:>5} ready "
#                     #     f"(total rows {X.shape[0]:,})"
#                     # )
#                     mb_counter += 1
#                     yield result
#                     result = None

#             t_iob_end = time.perf_counter()
#             # print(
#             #     f"[PID {pid}] ▲ finished IO-batch in {(t_iob_end - t_iob_start):.3f} s"
#             # )

#         # Emit any leftover rows that didn’t reach batch_size
#         if result is not None:
#             # print(
#             #     f"[PID {pid}]   • mini-batch {mb_counter:>5} (final, partial) "
#             #     f"size {result[0].shape[0]} rows"
#             # )
#             yield result

#     def __iter__(self) -> Iterator[MiniBatch]:
#         it = map(self.maybe_squeeze, self._iter())
#         return EagerIterator(it) if self.use_eager_fetch else it

#     def maybe_squeeze(self, mini_batch: MiniBatch) -> MiniBatch:
#         X, obs = mini_batch
#         if self.batch_size == 1:
#             # This is a no-op for `csr_matrix`s
#             return X[0], obs
#         else:
#             return mini_batch


# Copyright (c) TileDB, Inc. and The Chan Zuckerberg Initiative Foundation
#
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import Iterable, Iterator, Optional
import time, os

import attrs
import numpy as np
import pandas as pd
from scipy import sparse
import torch  # ← needed for randperm / device ops

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

    # --- GPU shuffle controls (must be passed from ExperimentDataset) ---
    gpu_shuffle: bool = False
    device: Optional[torch.device] = None
    seed: Optional[int] = None
    epoch: int = 0

    def _gpu_perm(self, n: int) -> torch.Tensor:
        """Create a permutation on device (or CPU) using seed+epoch for determinism."""
        is_cuda = self.device is not None and getattr(self.device, "type", None) == "cuda"
        gen = torch.Generator(device=self.device if is_cuda else "cpu")
        if self.seed is not None:
            gen.manual_seed(int(self.seed) + int(self.epoch))
        return torch.randperm(n, generator=gen, device=self.device if is_cuda else "cpu")

    def _iter(self) -> Iterator[MiniBatch]:
        """Yield MiniBatches while printing timing diagnostics."""
        batch_size = self.batch_size
        result: MiniBatch | None = None
        pid = os.getpid()            # helpful when several workers interleave prints
        mb_counter = 0               # running index of mini-batches produced

        for X_io_batch, obs_io_batch in self.io_batch_iter:
            t_iob_start = time.perf_counter()

            iob_idx = 0
            iob_len = X_io_batch.shape[0]

            # ---------- GPU within-IO-batch shuffle (dense only) ----------
            if self.gpu_shuffle:
                if self.return_sparse_X:
                    logger.warning(
                        "GPU shuffle requested but return_sparse_X=True; "
                        "leaving IO-batch row order unchanged for this batch."
                    )
                else:
                    t_gpu0 = time.perf_counter()

                    # 1) Build permutation (GPU if cuda device set)
                    perm = self._gpu_perm(iob_len)
                    perm_cpu = perm.to("cpu", non_blocking=False).numpy()

                    # 2) Materialize dense IO-batch once, shuffle rows on device
                    X_full = X_io_batch.slice_tonumpy(slice(0, iob_len))  # np.ndarray (C-contiguous)
                    X_t = torch.from_numpy(X_full)
                    if self.device is not None and getattr(self.device, "type", None) == "cuda":
                        X_t = X_t.to(self.device, non_blocking=True)
                    X_t = X_t.index_select(0, perm).contiguous()

                    # 3) Move back to CPU BEFORE yielding (DataLoader & pin_memory expect CPU)
                    X_cpu = X_t.to("cpu", non_blocking=False).numpy()
                    obs_perm = obs_io_batch.iloc[perm_cpu].reset_index(drop=True)

                    t_gpu1 = time.perf_counter()
                    logger.debug(
                        f"[PID {pid}] GPU shuffle+permute (rows={iob_len}) took {(t_gpu1 - t_gpu0):.3f}s"
                    )

                    # 4) Emit CPU mini-batches (keeps workers/pin-memory thread happy)
                    for start in range(0, iob_len, batch_size):
                        stop = min(start + batch_size, iob_len)
                        mb_counter += 1
                        yield (X_cpu[start:stop], obs_perm.iloc[start:stop].reset_index(drop=True))

                    t_iob_end = time.perf_counter()
                    logger.debug(
                        f"[PID {pid}] finished IO-batch (GPU path) in {(t_iob_end - t_iob_start):.3f}s"
                    )

                    # Ensure no lingering device tensors
                    del X_t, perm
                    continue
            # -------------------- end GPU path -----------------------------

            # print(f"[PID {pid}] ▼ received IO-batch of {iob_len:,} rows")
            while iob_idx < iob_len:
                # -----------------------------------------------------------------
                #  initialise per-iteration timing buckets
                # -----------------------------------------------------------------
                t0              = time.perf_counter()
                slice_ms        = 0.0   # time to slice X from the IO-batch
                obs_slice_ms    = 0.0   # time to slice obs rows
                stack_ms        = 0.0   # time spent in vstack / concatenate
                obs_cat_ms      = 0.0   # time spent in pd.concat

                # ------- build (or extend) the in-progress mini-batch ------------
                if result is None:
                    # ---- X slice ----
                    t1 = time.perf_counter()
                    X_datum = (
                        X_io_batch.slice_toscipy(slice(iob_idx, iob_idx + batch_size))
                        if self.return_sparse_X
                        else X_io_batch.slice_tonumpy(slice(iob_idx, iob_idx + batch_size))
                    )
                    t2 = time.perf_counter()
                    slice_ms = (t2 - t1) * 1_000

                    # ---- obs slice ----
                    t3 = time.perf_counter()
                    obs_datum = obs_io_batch.iloc[iob_idx : iob_idx + batch_size].reset_index(drop=True)
                    t4 = time.perf_counter()
                    obs_slice_ms = (t4 - t3) * 1_000

                    result = (X_datum, obs_datum)
                    iob_idx += len(result[1])

                else:
                    to_take = min(batch_size - len(result[1]), iob_len - iob_idx)

                    # ---- X slice ----
                    t1 = time.perf_counter()
                    X_part = (
                        X_io_batch.slice_toscipy(slice(0, to_take))
                        if self.return_sparse_X
                        else X_io_batch.slice_tonumpy(slice(0, to_take))
                    )
                    t2 = time.perf_counter()
                    slice_ms = (t2 - t1) * 1_000

                    # ---- X stack / concatenate ----
                    t3 = time.perf_counter()
                    X_datum = (
                        sparse.vstack([result[0], X_part])
                        if self.return_sparse_X
                        else np.concatenate([result[0], X_part])
                    )
                    t4 = time.perf_counter()
                    stack_ms = (t4 - t3) * 1_000

                    # ---- obs concat ----
                    t5 = time.perf_counter()
                    obs_datum = pd.concat([result[1], obs_io_batch.iloc[0:to_take]], ignore_index=True)
                    t6 = time.perf_counter()
                    obs_cat_ms = (t6 - t5) * 1_000

                    result = (X_datum, obs_datum)
                    iob_idx += to_take

                # -----------------------------------------------------------------
                #  summary print for THIS loop iteration (may be < batch_size big)
                # -----------------------------------------------------------------
                # logger.debug(
                #     f"[PID {pid}] slice {slice_ms:7.1f} ms | obs_slice {obs_slice_ms:7.1f} ms | "
                #     f"stack {stack_ms:7.1f} ms | obs_cat {obs_cat_ms:7.1f} ms"
                # )

                X, obs = result
                if X.shape[0] == batch_size:
                    mb_counter += 1
                    yield result
                    result = None

            t_iob_end = time.perf_counter()
            logger.debug(f"[PID {pid}] finished IO-batch (CPU path) in {(t_iob_end - t_iob_start):.3f}s")

        # Emit any leftover rows that didn’t reach batch_size
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
