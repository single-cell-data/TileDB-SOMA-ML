from math import log, lgamma, log2, gcd

LOG2 = log(2.0)

def log2_fact(n: int) -> float:
    if n <= 1:
        return 0.0
    return lgamma(n + 1) / LOG2

# ---------------- CPU (your original model) ----------------
def shuffle_entropy_bits(N: int, s: int, I: int):
    """
    CPU: chunk mixing (chunk size s) + IO-batch row shuffle (size I).
    Returns:
      - H_actual_bits, H_ideal_bits, ratio
      - spread_bits (per-row position entropy), reachable_frac
    """
    C = (N + s - 1) // s                     # number of chunks
    K, R = divmod(N, I)                       # full IO-batches and tail
    H_chunks = log2_fact(C)                   # chunk-order permutations
    H_io = K * log2_fact(I) + (log2_fact(R) if R else 0.0)   # in-batch permutations
    H_actual = H_chunks + H_io
    H_ideal = log2_fact(N)                    # full N! permutations

    # Per-row "spread":
    # With chunk mixing + in-batch shuffle, a row is ~uniform over all N positions.
    spread_bits = log2(N) if N > 1 else 0.0
    reachable_frac = 1.0

    return (H_actual, H_ideal, H_actual / H_ideal, spread_bits, reachable_frac)

# ---------------- GPU variants (closed-form) ----------------
def _io_batch_block_entropy(I: int, g: int | None, intra_block: bool) -> float:
    """
    Entropy (bits) for one IO-batch of size I when you block-shuffle with block size g.
      - full shuffle (g None/<=1/>=I): log2(I!)
      - block-permute only: log2(ceil(I/g)!)
      - block-permute + intra-block shuffle: add sum log2(block_size!) over blocks
    """
    if I <= 1:
        return 0.0
    if g is None or g <= 1 or g >= I:
        return log2_fact(I)  # full randperm

    b = (I + g - 1) // g     # number of blocks
    H = log2_fact(b)         # permute blocks
    if intra_block:
        full_blocks = I // g
        tail = I - full_blocks * g
        H += full_blocks * log2_fact(g)
        if tail:
            H += log2_fact(tail)
    return H

def _per_row_spread_iobatch(N: int, I: int, *,
                            chunk_mixing: bool,
                            g: int | None,
                            intra_block: bool) -> tuple[float, float]:
    """
    Per-row spread (bits) and reachable fraction for IO-batch-scoped shuffles.
    """
    if I <= 0 or N <= 1:
        return 0.0, 0.0

    # Full or intra-block shuffle ⇒ row can reach ~any of N positions if we mix chunks.
    if g is None or g <= 1 or g >= I or intra_block:
        if chunk_mixing:
            return (log2(N), 1.0)
        else:
            # No chunk mixing: row stays in its fixed IO-batch; positions spread within that batch.
            return (log2(I), I / N)

    # Block-permute only (no intra), block size g in an IO-batch of size I.
    b = (I + g - 1) // g  # blocks per IO-batch

    if chunk_mixing:
        # With chunk mixing, absolute positions are constrained modulo d = gcd(I, g).
        # Reachable fraction ≈ 1/d; entropy ≈ log2(N/d).
        d = gcd(I, g)
        d = max(1, d)
        return (log2(N) - log2(d), 1.0 / d)
    else:
        # Without chunk mixing, row is confined to its IO-batch and its in-block offset is fixed.
        # It can land in any of the b block slots within that batch => b positions total.
        return (log2(b), b / N)

def _per_row_spread_mini(N: int, M: int, *, chunk_mixing: bool) -> tuple[float, float]:
    """
    Per-row spread (bits) and reachable fraction for mini-batch-only shuffles.
    Typical use here: chunk_mixing=False.
    """
    M = max(1, min(M, N))
    if not chunk_mixing:
        return (log2(M), M / N)
    else:
        # If you *also* randomize chunk/order globally while only shuffling mini windows,
        # exact spread depends on how windows slide. As a conservative bound we return log2(min(M,N)).
        return (log2(M), min(1.0, M / N))

def gpu_entropy_bits(
    N: int,
    I: int,
    *,
    chunk_mixing: bool,          # True if you still randomize chunk order on CPU
    s: int | None = None,        # CPU shuffle_chunk_size (required if chunk_mixing)
    scope: str = "iobatch",      # "iobatch" or "mini"
    M: int | None = None,        # mini-batch size if scope="mini"
    g: int | None = None,        # gpu block size for block-shuffle (None => full)
    intra_block: bool = False    # also shuffle within each block
):
    """
    GPU global-entropy mirror of CPU model + per-row spread.
    Returns:
      - H_actual_bits, H_ideal_bits, ratio
      - spread_bits (per-row), reachable_frac
    """
    # Global entropy: chunk mixing contributes log2(C!)
    if chunk_mixing:
        if not s or s <= 0:
            raise ValueError("Provide shuffle_chunk_size `s` when chunk_mixing=True")
        C = (N + s - 1) // s
        H_chunks = log2_fact(C)
    else:
        H_chunks = 0.0

    # In-batch (row) entropy depends on scope & policy
    if scope == "iobatch":
        K, R = divmod(N, I)
        H_one = _io_batch_block_entropy(I, g, intra_block)
        H_tail = _io_batch_block_entropy(R, g, intra_block) if R else 0.0
        H_io = K * H_one + H_tail

        spread_bits, reachable_frac = _per_row_spread_iobatch(
            N, I, chunk_mixing=chunk_mixing, g=g, intra_block=intra_block
        )

    elif scope == "mini":
        if M is None or M <= 0:
            raise ValueError("Provide mini-batch size `M` when scope='mini'")
        K, R = divmod(N, M)
        H_io = K * log2_fact(M) + (log2_fact(R) if R else 0.0)
        spread_bits, reachable_frac = _per_row_spread_mini(
            N, M, chunk_mixing=chunk_mixing
        )
    else:
        raise ValueError("scope must be 'iobatch' or 'mini'")

    H_actual = H_chunks + H_io
    H_ideal = log2_fact(N)
    return (H_actual, H_ideal, H_actual / H_ideal, spread_bits, reachable_frac)

# ---------------- Examples ----------------
if __name__ == "__main__":
    N = 500_000
    s = 1024
    I = 2048

    # CPU baseline (chunk mixing + in-IO-batch full shuffle)
    Hc, Hi, rc, sc, rc_frac = shuffle_entropy_bits(N, s, I)
    print(f"CPU: global ratio={rc:.6f}, spread_bits={sc:.3f}, reachable_frac={rc_frac:.6f}")

    # GPU IO-batch full shuffle, keep CPU chunk mixing
    Hg, Hi2, rg, sg, rg_frac = gpu_entropy_bits(
        N, I, chunk_mixing=True, s=s, scope="iobatch", g=None, intra_block=False
    )
    print(f"GPU (iobatch, full): global ratio={rg:.6f}, spread_bits={sg:.3f}, reachable_frac={rg_frac:.6f}")

    # GPU IO-batch block-permute only (match shuffle_chunk_size granularity), keep chunk mixing
    Hg2, _, rg2, sg2, rg2_frac = gpu_entropy_bits(
        N, I, chunk_mixing=True, s=s, scope="iobatch", g=s, intra_block=False
    )
    print(f"GPU (iobatch, block only, g=s): global ratio={rg2:.6f}, spread_bits={sg2:.3f}, reachable_frac={rg2_frac:.6f}")

    # GPU IO-batch block-permute + intra-block shuffle (often ~full reach)
    Hg3, _, rg3, sg3, rg3_frac = gpu_entropy_bits(
        N, I, chunk_mixing=True, s=s, scope="iobatch", g=s, intra_block=True
    )
    print(f"GPU (iobatch, block + intra, g=s): global ratio={rg3:.6f}, spread_bits={sg3:.3f}, reachable_frac={rg3_frac:.6f}")

    # GPU mini-batch shuffle only (no chunk mixing)
    M = 1024
    Hg4, _, rg4, sg4, rg4_frac = gpu_entropy_bits(
        N, I, chunk_mixing=False, scope="mini", M=M
    )
    print(f"GPU (mini only, M={M}): global ratio={rg4:.6f}, spread_bits={sg4:.3f}, reachable_frac={rg4_frac:.6f}")
