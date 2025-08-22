set -euo pipefail

# ─── hyper-param grids ────────────────────────────────────────────────────────
IO_SIZES=(16384)          # 64 K  128 K  256 K rows
SHUFFLE_SIZES=(4096 8192 16384)


PY_SCRIPT="flat.py"                     # name of your training script

# ─── grid search loop ────────────────────────────────────────────────────────
for io in "${IO_SIZES[@]}"; do
  for sh in "${SHUFFLE_SIZES[@]}"; do
    echo ">>> io_batch_size=${io} | shuffle_chunk_size=${sh}"
    python "$PY_SCRIPT" \
        --io-batch-size "$io" \
        --shuffle-chunk-size "$sh"
    echo "<<< finished (io=${io}, shuffle=${sh})"
    echo
  done
done