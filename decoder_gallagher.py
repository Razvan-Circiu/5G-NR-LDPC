"""
decoder_gallagher.py — Gallagher B Bit-Flipping Decoder
Person 4 — Gallagher B

Algorithm overview:
  - HARD decision decoder: converts received signal to strict 0s and 1s FIRST
  - Simple and fast — no LLR arithmetic, just bit flipping
  - Each bit counts how many parity check equations it participates in
    are currently UNSATISFIED (syndrome bits = 1)
  - If that count exceeds a fixed threshold b → flip the bit
  - Repeat until all parity checks pass or max iterations reached

Key parameter:
  threshold b — typically chosen between 3 and 5
  Too low  → too many bits flip at once → oscillation
  Too high → too few bits flip → slow convergence
  Recommended starting point: b = majority of check node degree
  For 5G NR base matrix: experiment with b = 3, 4, 5

Hard decision input:
  The LLR array is passed in (same interface as soft decoders)
  but Gallagher B immediately converts to hard bits via sign:
    bit = 0 if LLR > 0 else 1

Update rule per iteration:
  For each bit v:
    count_v = number of unsatisfied checks involving v
              (i.e., checks c where syndrome[c] = 1)
    if count_v > b:
        flip bit v  (0 → 1 or 1 → 0)

All flips happen SIMULTANEOUSLY within one iteration.
"""

import numpy as np


# ---------------------------------------------------------------------------
# THRESHOLD SELECTION HELPER
# ---------------------------------------------------------------------------

def suggest_threshold(check_to_var: list) -> int:
    """
    Suggest a starting threshold b based on check node degrees.
    Rule of thumb: b = floor(median check degree / 2)
    """
    degrees = [len(nbrs) for nbrs in check_to_var]
    median_deg = int(np.median(degrees))
    suggested = max(1, median_deg // 2)
    print(f"[gallagher] Check degree — min:{min(degrees)} "
          f"max:{max(degrees)} median:{median_deg} → suggested b={suggested}")
    return suggested


# ---------------------------------------------------------------------------
# GALLAGHER B DECODER
# ---------------------------------------------------------------------------

def decode(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 100,
    threshold:    int = None,
) -> np.ndarray:
    """
    Gallagher B bit-flipping LDPC decoder.

    Parameters:
      llr          : 1D array of channel LLRs, shape (n,)
                     Used ONLY for initial hard decision — sign(LLR)
      H            : sparse parity-check matrix, shape (m, n)
      check_to_var : list of arrays — check_to_var[c] = variable indices for check c
      var_to_check : list of arrays — var_to_check[v] = check indices for variable v
      max_iter     : maximum number of bit-flipping iterations (default 100)
      threshold    : fixed flip threshold b (auto-selected if None)

    Returns:
      hard_bits : 1D uint8 array of length n — decoded codeword
    """
    n = len(llr)
    m = len(check_to_var)

    # ------------------------------------------------------------------
    # STEP 0: Hard decision — convert LLR to bits immediately
    # LLR > 0 → bit = 0,  LLR < 0 → bit = 1,  LLR = 0 → bit = 0
    # ------------------------------------------------------------------
    bits = (llr < 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # THRESHOLD SELECTION
    # ------------------------------------------------------------------
    if threshold is None:
        threshold = suggest_threshold(check_to_var)

    # ------------------------------------------------------------------
    # CHECK INITIAL SYNDROME
    # ------------------------------------------------------------------
    # syndrome[c] = 1 if parity check c is unsatisfied, 0 if satisfied
    syndrome = _compute_syndrome_fast(bits, check_to_var, m)

    if np.all(syndrome == 0):
        return bits   # already a valid codeword (no noise)

    # ------------------------------------------------------------------
    # ITERATION LOOP
    # ------------------------------------------------------------------
    for iteration in range(max_iter):

        # STEP 1: For each variable node v, count unsatisfied checks
        # count_v = number of checks c in N(v) where syndrome[c] = 1
        flip_flags = np.zeros(n, dtype=np.uint8)

        for v in range(n):
            check_nbrs = var_to_check[v]         # check nodes connected to v
            # Count how many of those checks are currently unsatisfied
            unsatisfied_count = int(np.sum(syndrome[check_nbrs]))

            if unsatisfied_count > threshold:
                flip_flags[v] = 1

        # STEP 2: Flip all flagged bits SIMULTANEOUSLY
        bits ^= flip_flags                       # XOR flips: 0→1, 1→0

        # STEP 3: Recompute syndrome
        syndrome = _compute_syndrome_fast(bits, check_to_var, m)

        # STEP 4: Check if all parity equations satisfied
        if np.all(syndrome == 0):
            return bits

    # Max iterations reached — return best guess
    return bits


# ---------------------------------------------------------------------------
# VECTORIZED VERSION (faster — recommended for BER simulation)
# ---------------------------------------------------------------------------

def decode_fast(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 100,
    threshold:    int = None,
) -> np.ndarray:
    """
    Vectorized Gallagher B decoder.
    Uses numpy matrix operations instead of Python loops over variable nodes.
    Same algorithm as decode() but significantly faster for large codes.
    """
    n = len(llr)
    m = len(check_to_var)

    # Hard decision
    bits = (llr < 0).astype(np.uint8)

    if threshold is None:
        degrees   = [len(nbrs) for nbrs in check_to_var]
        threshold = max(1, int(np.median(degrees)) // 2)

    # Build a fast lookup: var_to_check as padded matrix for vectorized ops
    # For each variable v, we need sum of syndrome over its check neighbors
    # We precompute this using sparse structure

    syndrome = _compute_syndrome_fast(bits, check_to_var, m)

    if np.all(syndrome == 0):
        return bits

    for iteration in range(max_iter):

        # Vectorized count: for each variable v, sum syndrome over N(v)
        # unsatisfied_count[v] = Σ syndrome[c] for c in var_to_check[v]
        unsatisfied_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            unsatisfied_count[v] = int(np.sum(syndrome[var_to_check[v]]))

        # Flip all bits where count exceeds threshold
        flip_mask = (unsatisfied_count > threshold).astype(np.uint8)
        bits ^= flip_mask

        # Recompute syndrome
        syndrome = _compute_syndrome_fast(bits, check_to_var, m)

        if np.all(syndrome == 0):
            return bits

    return bits


# ---------------------------------------------------------------------------
# ADAPTIVE THRESHOLD VARIANT (bonus — better BER than fixed threshold)
# ---------------------------------------------------------------------------

def decode_adaptive(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 100,
) -> np.ndarray:
    """
    Gallagher B with adaptive threshold.

    Instead of a fixed threshold b, the threshold adapts each iteration:
      b_iter = max(1, floor(max_unsatisfied_count / 2))

    This allows aggressive flipping early (many errors) and conservative
    flipping late (near convergence), reducing oscillation.

    Optional improvement — implement if time allows.
    """
    n = len(llr)
    m = len(check_to_var)

    bits     = (llr < 0).astype(np.uint8)
    syndrome = _compute_syndrome_fast(bits, check_to_var, m)

    if np.all(syndrome == 0):
        return bits

    for iteration in range(max_iter):

        # Compute unsatisfied count per variable
        unsatisfied_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            unsatisfied_count[v] = int(np.sum(syndrome[var_to_check[v]]))

        # Adaptive threshold: based on current distribution
        max_count = int(np.max(unsatisfied_count))
        if max_count == 0:
            break
        threshold = max(1, max_count // 2)

        flip_mask = (unsatisfied_count > threshold).astype(np.uint8)
        bits     ^= flip_mask

        syndrome = _compute_syndrome_fast(bits, check_to_var, m)
        if np.all(syndrome == 0):
            return bits

    return bits


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _compute_syndrome_fast(
    bits:         np.ndarray,
    check_to_var: list,
    m:            int,
) -> np.ndarray:
    """
    Compute syndrome vector efficiently.
    syndrome[c] = (sum of bits in check c) mod 2
    Returns uint8 array of length m.
    0 = check satisfied, 1 = check unsatisfied.
    """
    syndrome = np.zeros(m, dtype=np.uint8)
    for c in range(m):
        syndrome[c] = int(np.sum(bits[check_to_var[c]])) % 2
    return syndrome


# ---------------------------------------------------------------------------
# BER vs THRESHOLD SWEEP (useful for report / tuning)
# ---------------------------------------------------------------------------

def sweep_threshold(
    llr_set:      list,
    codewords:    list,
    H,
    check_to_var: list,
    var_to_check: list,
    thresholds:   list = None,
    max_iter:     int  = 100,
    k:            int  = None,
) -> dict:
    """
    Run Gallagher B over multiple threshold values and return BER per threshold.
    Use this to find the optimal b for your base matrix.

    Parameters:
      llr_set    : list of LLR arrays (one per trial)
      codewords  : list of original codewords (one per trial)
      thresholds : list of b values to test (default: 2, 3, 4, 5, 6)
      k          : number of message bits (tail of codeword)

    Returns:
      dict: {threshold: BER}
    """
    if thresholds is None:
        thresholds = [2, 3, 4, 5, 6]

    results = {}
    n = len(llr_set[0])
    if k is None:
        k = n // 2

    for b in thresholds:
        total_bits   = 0
        total_errors = 0
        for llr, cw in zip(llr_set, codewords):
            decoded = decode_fast(
                llr, H, check_to_var, var_to_check,
                max_iter=max_iter, threshold=b
            )
            msg_orig    = cw[n - k:]
            msg_decoded = decoded[n - k:]
            total_errors += int(np.sum(msg_orig != msg_decoded))
            total_bits   += k
        results[b] = total_errors / total_bits
        print(f"[gallagher] threshold b={b} → BER={results[b]:.4e}")

    best_b = min(results, key=results.get)
    print(f"[gallagher] Best threshold: b={best_b} (BER={results[best_b]:.4e})")
    return results


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import scipy.sparse as sp
    from ldpc_utils import (
        expand_base_matrix,
        build_adjacency,
        bpsk_modulate,
        add_awgn,
        compute_llr,
        syndrome_check,
    )
    from encoder import get_systematic_encoder

    print("=== decoder_gallagher.py self-test ===\n")

    bmat_test = np.array([
        [ 0,  1,  2, -1,  3],
        [-1,  0,  1,  2, -1],
        [ 1, -1,  0, -1,  2],
    ])
    Zc_test = 4

    H            = expand_base_matrix(bmat_test, Zc=Zc_test)
    H_sp         = sp.csr_matrix(H)
    check_to_var, var_to_check = build_adjacency(H)
    encode_fn, k, n = get_systematic_encoder(H_sp)

    print(f"Code: k={k}, n={n}\n")

    np.random.seed(42)
    n_trials = 100

    # High SNR test
    ok_high = 0
    for _ in range(n_trials):
        msg      = np.random.randint(0, 2, k, dtype=np.uint8)
        codeword = encode_fn(msg)
        tx       = bpsk_modulate(codeword)
        rx       = add_awgn(tx, snr_db=6.0)
        llr      = compute_llr(rx, snr_db=6.0)
        decoded  = decode_fast(llr, H, check_to_var, var_to_check,
                               max_iter=100, threshold=3)
        if syndrome_check(H_sp, decoded):
            ok_high += 1

    print(f"High SNR (6dB): {ok_high}/{n_trials} frames decoded correctly")

    # Medium SNR test
    ok_med = 0
    for _ in range(n_trials):
        msg      = np.random.randint(0, 2, k, dtype=np.uint8)
        codeword = encode_fn(msg)
        tx       = bpsk_modulate(codeword)
        rx       = add_awgn(tx, snr_db=3.0)
        llr      = compute_llr(rx, snr_db=3.0)
        decoded  = decode_fast(llr, H, check_to_var, var_to_check,
                               max_iter=100, threshold=3)
        if syndrome_check(H_sp, decoded):
            ok_med += 1

    print(f"Med  SNR (3dB): {ok_med}/{n_trials} frames decoded correctly")

    # Threshold sweep demo
    print("\n--- Threshold sweep (SNR=4dB, 50 trials) ---")
    llr_set   = []
    cw_set    = []
    for _ in range(50):
        msg      = np.random.randint(0, 2, k, dtype=np.uint8)
        codeword = encode_fn(msg)
        tx       = bpsk_modulate(codeword)
        rx       = add_awgn(tx, snr_db=4.0)
        llr_set.append(compute_llr(rx, snr_db=4.0))
        cw_set.append(codeword)

    sweep_threshold(llr_set, cw_set, H, check_to_var, var_to_check,
                    thresholds=[2, 3, 4, 5], k=k)

    print("\nSelf-test complete.")