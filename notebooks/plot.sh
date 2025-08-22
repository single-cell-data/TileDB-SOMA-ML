set -euo pipefail

# ─── hyper-param grids ────────────────────────────────────────────────────────
IO_SIZES=(4096 8192 16384 32768 65536 131072 262144)          # 64 K  128 K  256 K rows

PY_SCRIPT="viz.py"                     # name of your training script
PY_SCRIPT2="viz2.py"

# ─── grid search loop ────────────────────────────────────────────────────────
for io in "${IO_SIZES[@]}"; do
    echo ">>> io_batch_size=${io}"
    python "$PY_SCRIPT" \
        --io_batch_size="$io" 
    python "$PY_SCRIPT2" \
        --io_batch_size="$io" 
    echo "<<< finished (io=${io}"
    echo
done