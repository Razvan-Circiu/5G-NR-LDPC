"""
decoder_layered.py — Layered Min-Sum Decoder
Person 3 — Layered Min-Sum

Algorithm overview:
  - Soft decision decoder: works with Log-Likelihood Ratios (LLRs)
  - Check nodes are processed ONE ROW (LAYER) AT A TIME
  - Each row immediately uses the UPDATED LLRs from previous rows
    within the same iteration — information propagates faster
  - Converges in ~half the iterations of Flooded Min-Sum

Key difference from Flooded:
  Flooded  → all check nodes update, THEN all variable nodes update
  Layered  → for each check node row:
               1. REMOVE old contribution of this row from LLR
               2. Recompute check-to-variable messages
               3. ADD new contribution back to LLR immediately
             next row already sees the updated LLR values

Message storage:
  llr_total[v]  : running total LLR per variable node (updated in place)
  R[c]          : stored check-to-variable messages from PREVIOUS iteration
                  needed to subtract old contribution before adding new one

Update equations per layer c:

  Extrinsic LLR (remove old row contribution):
    Q[c][v] = llr_total[v] - R_old[c][v]    for v in N(c)

  Check node update (Min-Sum):
    R_new[c][v] = sign_product × min_magnitude    (leave-one-out, same as Flooded)

  Variable node update (add new contribution immediately):
    llr_total[v] += R_new[c][v] - R_old[c][v]    for v in N(c)

  Hard decision (after all rows processed):
    bit[v] = 0 if llr_total[v] > 0 else 1
"""

import numpy as np


# ---------------------------------------------------------------------------
# LAYERED MIN-SUM DECODER
# ---------------------------------------------------------------------------

def decode(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 15,
) -> np.ndarray:
    """
    Layered Min-Sum LDPC decoder.

    Parameters:
      llr          : 1D array of channel LLRs, shape (n,)
                     Convention: LLR > 0 → bit=0 more likely
      H            : sparse parity-check matrix, shape (m, n)
      check_to_var : list of arrays — check_to_var[c] = variable indices for check c
      var_to_check : list of arrays — var_to_check[v] = check indices for variable v
      max_iter     : maximum number of decoding iterations
                     NOTE: Layered needs ~half the iterations of Flooded
                     Default 15 ≈ equivalent to Flooded with 30 iterations

    Returns:
      hard_bits : 1D uint8 array of length n — decoded codeword
    """
    n = len(llr)
    m = len(check_to_var)

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    # llr_total: running LLR for each variable node
    # Starts as channel LLR and gets updated in-place as each row is processed
    llr_total = llr.copy().astype(np.float64)

    # R[c]: stored check-to-variable messages from previous pass through row c
    # Must be stored to allow subtraction of old contribution
    # Initialized to zero (no prior contribution from any check node)
    R = [np.zeros(len(check_to_var[c]), dtype=np.float64) for c in range(m)]

    # ------------------------------------------------------------------
    # ITERATION LOOP
    # ------------------------------------------------------------------
    for iteration in range(max_iter):

        # --------------------------------------------------------------
        # Process each check node (layer) SEQUENTIALLY
        # --------------------------------------------------------------
        for c in range(m):
            var_nodes = check_to_var[c]          # variable nodes connected to check c
            deg_c     = len(var_nodes)

            # STEP 1: Compute extrinsic LLR for each variable in this row
            # Remove the OLD contribution of check c from the running total
            # This gives us the LLR from all OTHER check nodes + channel
            Q = llr_total[var_nodes] - R[c]      # shape: (deg_c,)

            # STEP 2: Check node update — Min-Sum (same as Flooded)
            signs = np.where(Q >= 0, 1.0, -1.0)
            mags  = np.abs(Q)

            # --- Leave-one-out sign product via prefix/suffix ---
            sign_prefix    = np.ones(deg_c + 1)
            sign_suffix    = np.ones(deg_c + 1)
            for i in range(deg_c):
                sign_prefix[i + 1] = sign_prefix[i] * signs[i]
            for i in range(deg_c - 1, -1, -1):
                sign_suffix[i] = sign_suffix[i + 1] * signs[i]
            sign_loo = sign_prefix[:deg_c] * sign_suffix[1:]

            # --- Leave-one-out minimum magnitude ---
            sorted_idx = np.argsort(mags)
            min1_val   = mags[sorted_idx[0]]
            min2_val   = mags[sorted_idx[1]] if deg_c > 1 else min1_val
            min1_pos   = sorted_idx[0]
            mag_loo    = np.where(np.arange(deg_c) == min1_pos, min2_val, min1_val)

            # New check-to-variable message
            R_new = sign_loo * mag_loo           # shape: (deg_c,)

            # STEP 3: Update running LLR immediately (this is what makes it "Layered")
            # The next check node row will already see these updated values
            llr_total[var_nodes] += R_new - R[c]

            # STEP 4: Store new messages for next iteration
            R[c] = R_new

        # --------------------------------------------------------------
        # Hard decision after all rows processed
        # --------------------------------------------------------------
        hard_bits = (llr_total < 0).astype(np.uint8)

        # --------------------------------------------------------------
        # Syndrome check — stop if all parity checks satisfied
        # --------------------------------------------------------------
        syndrome_ok = True
        for c in range(m):
            if int(np.sum(hard_bits[check_to_var[c]])) % 2 != 0:
                syndrome_ok = False
                break

        if syndrome_ok:
            return hard_bits

    return hard_bits


# ---------------------------------------------------------------------------
# VECTORIZED VERSION (faster — recommended for BER simulation)
# ---------------------------------------------------------------------------

def decode_fast(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 15,
    scaling:      float = 0.8,
) -> np.ndarray:
    """
    Vectorized Layered Min-Sum with optional scaling factor.

    Scaling factor (0.75–0.85) compensates for Min-Sum's tendency to
    overestimate message magnitudes. Improves BER by ~0.2–0.5 dB.
    Set scaling=1.0 to disable.

    Same algorithm as decode() but tighter numpy operations.
    """
    n = len(llr)
    m = len(check_to_var)

    llr_total = llr.copy().astype(np.float64)
    R = [np.zeros(len(check_to_var[c]), dtype=np.float64) for c in range(m)]

    for iteration in range(max_iter):

        for c in range(m):
            var_nodes = check_to_var[c]
            deg_c     = len(var_nodes)

            # Extrinsic LLR: remove old check c contribution
            Q     = llr_total[var_nodes] - R[c]
            signs = np.where(Q >= 0, 1.0, -1.0)
            mags  = np.abs(Q)

            # Leave-one-out sign (prefix/suffix product)
            sp_ = np.ones(deg_c + 1)
            ss_ = np.ones(deg_c + 1)
            sp_[1:]  = np.cumprod(signs)
            ss_[:-1] = np.cumprod(signs[::-1])[::-1]
            sign_loo = sp_[:deg_c] * ss_[1:]

            # Leave-one-out min magnitude
            idx_sort = np.argsort(mags)
            min1     = mags[idx_sort[0]]
            min2     = mags[idx_sort[1]] if deg_c > 1 else min1
            mag_loo  = np.where(np.arange(deg_c) == idx_sort[0], min2, min1)

            # New message with scaling
            R_new = scaling * sign_loo * mag_loo

            # Update running LLR in place
            llr_total[var_nodes] += R_new - R[c]
            R[c] = R_new

        # Hard decision + syndrome check
        hard_bits = (llr_total < 0).astype(np.uint8)

        ok = True
        for c in range(m):
            if int(np.sum(hard_bits[check_to_var[c]])) % 2 != 0:
                ok = False
                break
        if ok:
            return hard_bits

    return hard_bits


# ---------------------------------------------------------------------------
# CONVERGENCE ANALYSIS HELPER  (useful for presentation / report)
# ---------------------------------------------------------------------------

def decode_with_convergence_trace(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 20,
) -> tuple:
    """
    Same as decode() but also returns per-iteration statistics.
    Useful for plotting convergence speed vs Flooded.

    Returns:
      hard_bits          : final decoded codeword
      unsatisfied_checks : list of int — number of failed parity checks per iteration
      converged_at       : iteration index where syndrome = 0, or None
    """
    n = len(llr)
    m = len(check_to_var)

    llr_total        = llr.copy().astype(np.float64)
    R                = [np.zeros(len(check_to_var[c]), dtype=np.float64) for c in range(m)]
    unsatisfied_log  = []
    converged_at     = None

    for iteration in range(max_iter):

        for c in range(m):
            var_nodes = check_to_var[c]
            deg_c     = len(var_nodes)

            Q     = llr_total[var_nodes] - R[c]
            signs = np.where(Q >= 0, 1.0, -1.0)
            mags  = np.abs(Q)

            sp_ = np.ones(deg_c + 1)
            ss_ = np.ones(deg_c + 1)
            sp_[1:]  = np.cumprod(signs)
            ss_[:-1] = np.cumprod(signs[::-1])[::-1]
            sign_loo = sp_[:deg_c] * ss_[1:]

            idx_sort = np.argsort(mags)
            min1     = mags[idx_sort[0]]
            min2     = mags[idx_sort[1]] if deg_c > 1 else min1
            mag_loo  = np.where(np.arange(deg_c) == idx_sort[0], min2, min1)

            R_new = sign_loo * mag_loo
            llr_total[var_nodes] += R_new - R[c]
            R[c] = R_new

        hard_bits   = (llr_total < 0).astype(np.uint8)
        n_failed    = sum(
            1 for c in range(m)
            if int(np.sum(hard_bits[check_to_var[c]])) % 2 != 0
        )
        unsatisfied_log.append(n_failed)

        if n_failed == 0 and converged_at is None:
            converged_at = iteration
            break

    return hard_bits, unsatisfied_log, converged_at


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

    print("=== decoder_layered.py self-test ===\n")

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
    n_trials = 50

    # High SNR test
    ok_high = 0
    for _ in range(n_trials):
        msg      = np.random.randint(0, 2, k, dtype=np.uint8)
        codeword = encode_fn(msg)
        tx       = bpsk_modulate(codeword)
        rx       = add_awgn(tx, snr_db=6.0)
        llr      = compute_llr(rx, snr_db=6.0)
        decoded  = decode_fast(llr, H, check_to_var, var_to_check, max_iter=15)
        if syndrome_check(H_sp, decoded):
            ok_high += 1

    print(f"High SNR (6dB): {ok_high}/{n_trials} frames decoded correctly")

    # Low SNR test
    ok_low = 0
    for _ in range(n_trials):
        msg      = np.random.randint(0, 2, k, dtype=np.uint8)
        codeword = encode_fn(msg)
        tx       = bpsk_modulate(codeword)
        rx       = add_awgn(tx, snr_db=1.0)
        llr      = compute_llr(rx, snr_db=1.0)
        decoded  = decode_fast(llr, H, check_to_var, var_to_check, max_iter=15)
        if syndrome_check(H_sp, decoded):
            ok_low += 1

    print(f"Low  SNR (1dB): {ok_low}/{n_trials} frames decoded correctly")

    # Convergence trace demo
    print("\n--- Convergence trace (single frame, SNR=3dB) ---")
    msg      = np.random.randint(0, 2, k, dtype=np.uint8)
    codeword = encode_fn(msg)
    tx       = bpsk_modulate(codeword)
    rx       = add_awgn(tx, snr_db=3.0)
    llr      = compute_llr(rx, snr_db=3.0)
    _, trace, conv_at = decode_with_convergence_trace(
        llr, H, check_to_var, var_to_check, max_iter=20
    )
    for i, n_fail in enumerate(trace):
        print(f"  Iter {i+1:2d}: {n_fail} unsatisfied check nodes")
    if conv_at is not None:
        print(f"  → Converged at iteration {conv_at + 1}")
    else:
        print(f"  → Did not converge within {len(trace)} iterations")

    print("\nSelf-test complete.")