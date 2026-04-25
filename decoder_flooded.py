"""
decoder_flooded.py — Flooded Min-Sum Decoder
Person 2 — Flooded Min-Sum

Algorithm overview:
  - Soft decision decoder: works with Log-Likelihood Ratios (LLRs)
  - All check nodes update their messages SIMULTANEOUSLY
  - Then all variable nodes update SIMULTANEOUSLY
  - This is the standard "parallel" belief propagation approach

Message types:
  - L[v]       : channel LLR for variable node v  (fixed, from input)
  - Q[c][v]    : variable-to-check message  (v → c)
  - R[c][v]    : check-to-variable message  (c → v)

Update equations (Min-Sum approximation of Belief Propagation):

  Check node update (c → v):
    R[c][v] = ( ∏_{v' ∈ N(c)\\v} sign(Q[c][v']) ) × min_{v' ∈ N(c)\\v} |Q[c][v']|

  Variable node update (v → c):
    Q[c][v] = L[v] + Σ_{c' ∈ N(v)\\c} R[c'][v]

  Total LLR (used for hard decision):
    LLR_total[v] = L[v] + Σ_{c ∈ N(v)} R[c][v]

  Hard decision:
    bit[v] = 0 if LLR_total[v] > 0 else 1
"""

import numpy as np


# ---------------------------------------------------------------------------
# FLOODED MIN-SUM DECODER
# ---------------------------------------------------------------------------

def decode(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 30,
) -> np.ndarray:
    """
    Flooded Min-Sum LDPC decoder.

    Parameters:
      llr          : 1D array of channel LLRs, shape (n,)
                     Convention: LLR > 0 → bit=0 more likely
      H            : sparse parity-check matrix, shape (m, n)
      check_to_var : list of arrays — check_to_var[c] = variable node indices for check c
      var_to_check : list of arrays — var_to_check[v] = check node indices for variable v
      max_iter     : maximum number of decoding iterations

    Returns:
      hard_bits : 1D uint8 array of length n — decoded codeword
    """
    n = len(llr)
    m = len(check_to_var)

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------
    # Variable-to-check messages Q[c][v]: initialized to channel LLR
    # We store as a flat array indexed by edge index.
    # For simplicity at lab scale, use a dict: (c, v) → message value.
    # For speed, we use arrays per check node.

    # R[c] = array of check-to-variable messages for check node c
    # Q[c] = array of variable-to-check messages for check node c
    # Both indexed in the same order as check_to_var[c]

    R = [np.zeros(len(check_to_var[c]), dtype=np.float64) for c in range(m)]
    Q = [llr[check_to_var[c]].copy()                      for c in range(m)]
    # Q[c][i] = message from variable check_to_var[c][i] to check node c
    # Initialized to channel LLR (standard initialization)

    # ------------------------------------------------------------------
    # ITERATION LOOP
    # ------------------------------------------------------------------
    for iteration in range(max_iter):

        # --------------------------------------------------------------
        # STEP 1: Check node update (ALL check nodes simultaneously)
        # R[c][i] = sign_product × min_magnitude
        # where product/min are over all v' in N(c) except v = check_to_var[c][i]
        # --------------------------------------------------------------
        for c in range(m):
            msgs = Q[c]                          # shape: (deg_c,)
            signs = np.where(msgs >= 0, 1.0, -1.0)                # +1 or -1
            mags  = np.abs(msgs)

            # Total sign product for the whole check node
            total_sign = np.prod(signs)

            # Min magnitude and second-min (needed for leave-one-out min)
            sorted_idx  = np.argsort(mags)
            min1_val    = mags[sorted_idx[0]]    # smallest magnitude
            min2_val    = mags[sorted_idx[1]] if len(mags) > 1 else min1_val
            min1_idx    = sorted_idx[0]

            for i in range(len(msgs)):
                # Sign: total product divided by sign of this variable
                sign_i = total_sign * signs[i]   # = product of all OTHER signs

                # Min magnitude excluding variable i
                mag_i = min2_val if i == min1_idx else min1_val

                R[c][i] = sign_i * mag_i

        # --------------------------------------------------------------
        # STEP 2: Variable node update (ALL variable nodes simultaneously)
        # Q[c][i] = L[v] + sum of all R[c'][v] for c' in N(v), c' ≠ c
        # --------------------------------------------------------------

        # First, compute total incoming R sum per variable node
        total_R = np.zeros(n, dtype=np.float64)
        for c in range(m):
            for i, v in enumerate(check_to_var[c]):
                total_R[v] += R[c][i]

        # Then subtract the contribution of each specific check node
        for c in range(m):
            for i, v in enumerate(check_to_var[c]):
                Q[c][i] = llr[v] + total_R[v] - R[c][i]

        # --------------------------------------------------------------
        # STEP 3: Hard decision on total LLR
        # LLR_total[v] = L[v] + sum_{c in N(v)} R[c][v]
        # --------------------------------------------------------------
        llr_total  = llr + total_R
        hard_bits  = (llr_total < 0).astype(np.uint8)

        # --------------------------------------------------------------
        # STEP 4: Syndrome check — stop if all parity checks satisfied
        # --------------------------------------------------------------
        syndrome_ok = True
        for c in range(m):
            parity = int(np.sum(hard_bits[check_to_var[c]])) % 2
            if parity != 0:
                syndrome_ok = False
                break

        if syndrome_ok:
            return hard_bits

    # Max iterations reached — return best guess
    return hard_bits


# ---------------------------------------------------------------------------
# VECTORIZED VERSION (faster — recommended for simulation)
# ---------------------------------------------------------------------------

def decode_fast(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 30,
) -> np.ndarray:
    """
    Vectorized Flooded Min-Sum decoder.
    Same algorithm as decode() but uses numpy operations per check node
    instead of Python loops over individual edges — significantly faster.

    Use this version in simulate.py for BER curves.
    """
    n = len(llr)
    m = len(check_to_var)

    # Initialize variable-to-check messages to channel LLR
    Q = [llr[check_to_var[c]].copy() for c in range(m)]
    R = [np.zeros(len(check_to_var[c]), dtype=np.float64) for c in range(m)]

    for iteration in range(max_iter):

        # --- Check node updates (vectorized per check node) ---
        for c in range(m):
            msgs  = Q[c]
            signs = np.where(msgs >= 0, 1.0, -1.0)
            mags  = np.abs(msgs)

            # Sign product (leave-one-out): cumulative products from left and right
            n_c         = len(msgs)
            sign_prefix = np.ones(n_c + 1)
            sign_suffix = np.ones(n_c + 1)
            for i in range(n_c):
                sign_prefix[i + 1] = sign_prefix[i] * signs[i]
            for i in range(n_c - 1, -1, -1):
                sign_suffix[i] = sign_suffix[i + 1] * signs[i]
            sign_loo = sign_prefix[:n_c] * sign_suffix[1:]  # leave-one-out sign

            # Min magnitude (leave-one-out): track min and second-min
            sorted_idx = np.argsort(mags)
            min1_val   = mags[sorted_idx[0]]
            min2_val   = mags[sorted_idx[1]] if n_c > 1 else min1_val
            min1_pos   = sorted_idx[0]

            mag_loo = np.where(np.arange(n_c) == min1_pos, min2_val, min1_val)

            R[c] = sign_loo * mag_loo

        # --- Variable node total LLR accumulation ---
        total_R = np.zeros(n, dtype=np.float64)
        for c in range(m):
            np.add.at(total_R, check_to_var[c], R[c])

        # --- Variable-to-check message update ---
        for c in range(m):
            Q[c] = llr[check_to_var[c]] + total_R[check_to_var[c]] - R[c]

        # --- Hard decision and syndrome check ---
        llr_total = llr + total_R
        hard_bits = (llr_total < 0).astype(np.uint8)

        # Vectorized syndrome check
        syndrome_failed = False
        for c in range(m):
            if int(np.sum(hard_bits[check_to_var[c]])) % 2 != 0:
                syndrome_failed = True
                break

        if not syndrome_failed:
            return hard_bits

    return hard_bits


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

    print("=== decoder_flooded.py self-test ===\n")

    # Small test: 3-row base matrix, Zc=4 for quick verification
    bmat_test = np.array([
        [ 0,  1,  2, -1,  3],
        [-1,  0,  1,  2, -1],
        [ 1, -1,  0, -1,  2],
    ])
    Zc_test = 4

    H = expand_base_matrix(bmat_test, Zc=Zc_test)
    check_to_var, var_to_check = build_adjacency(H)
    encode_fn, k, n = get_systematic_encoder(sp.csr_matrix(H))

    print(f"Code: k={k}, n={n}\n")

    # Test at high SNR — should decode perfectly
    np.random.seed(42)
    n_trials = 50
    errors   = 0

    for _ in range(n_trials):
        msg      = np.random.randint(0, 2, k, dtype=np.uint8)
        codeword = encode_fn(msg)
        tx       = bpsk_modulate(codeword)
        rx       = add_awgn(tx, snr_db=6.0)
        llr      = compute_llr(rx, snr_db=6.0)

        decoded  = decode_fast(llr, H, check_to_var, var_to_check, max_iter=30)
        if not syndrome_check(sp.csr_matrix(H), decoded):
            errors += 1

    print(f"High SNR (6dB): {n_trials - errors}/{n_trials} frames decoded correctly")

    # Test at low SNR — expect some failures
    errors_low = 0
    for _ in range(n_trials):
        msg      = np.random.randint(0, 2, k, dtype=np.uint8)
        codeword = encode_fn(msg)
        tx       = bpsk_modulate(codeword)
        rx       = add_awgn(tx, snr_db=1.0)
        llr      = compute_llr(rx, snr_db=1.0)

        decoded  = decode_fast(llr, H, check_to_var, var_to_check, max_iter=30)
        if not syndrome_check(sp.csr_matrix(H), decoded):
            errors_low += 1

    print(f"Low  SNR (1dB): {n_trials - errors_low}/{n_trials} frames decoded correctly")
    print(f"\nSelf-test complete.")