"""
decoder_gdbf.py — GDBF and PGDBF Decoders
Person 5 — Gradient Descent Bit Flipping + Probabilistic variant

=== GDBF — Gradient Descent Bit Flipping ===

Smarter than Gallagher B: instead of flipping ALL bits above a fixed
threshold, GDBF computes an ENERGY FUNCTION per bit and flips only
the bit(s) with MAXIMUM energy each iteration.

Energy function for variable node v:
  E(v) = Σ_{c in N(v)} (1 - 2*syndrome[c])
       = (satisfied checks in N(v)) - (unsatisfied checks in N(v))

Interpretation:
  E(v) > 0  → more satisfied than unsatisfied → bit is probably CORRECT
  E(v) < 0  → more unsatisfied than satisfied → bit is probably WRONG
  E(v) very negative → strong signal to flip this bit

Flip rule: flip the bit v* with MINIMUM energy (most negative)
  v* = argmin E(v)
  flip bits[v*]

This is "gradient descent" on the energy landscape — always move
in the direction that most reduces the total number of unsatisfied checks.

=== PGDBF — Probabilistic GDBF ===

Extension of GDBF: instead of flipping exactly one bit, PGDBF flips
multiple bits probabilistically based on their energy.

Flip probability for bit v:
  P(flip v) = sigmoid(-E(v) / temperature)
             = 1 / (1 + exp(E(v) / temperature))

  temperature controls exploration vs exploitation:
    low T  → almost deterministic (approaches GDBF)
    high T → more random flips → escapes local minima

Bits with very negative energy → high flip probability
Bits with positive energy → low flip probability

PGDBF can escape oscillation traps that GDBF gets stuck in,
at the cost of occasional wrong flips.
"""

import numpy as np


# ---------------------------------------------------------------------------
# SHARED HELPER — ENERGY COMPUTATION
# ---------------------------------------------------------------------------

def _compute_energy(
    bits:         np.ndarray,
    check_to_var: list,
    var_to_check: list,
    n:            int,
    m:            int,
) -> tuple:
    """
    Compute syndrome and energy for all variable nodes.

    Returns:
      syndrome : uint8 array (m,) — 0=satisfied, 1=unsatisfied
      energy   : float64 array (n,) — E(v) = satisfied - unsatisfied checks
    """
    # Syndrome
    syndrome = np.zeros(m, dtype=np.uint8)
    for c in range(m):
        syndrome[c] = int(np.sum(bits[check_to_var[c]])) % 2

    # Energy: E(v) = Σ (1 - 2*syndrome[c]) for c in N(v)
    # = count(satisfied) - count(unsatisfied) for checks around v
    energy = np.zeros(n, dtype=np.float64)
    for v in range(n):
        checks = var_to_check[v]
        energy[v] = float(np.sum(1 - 2 * syndrome[checks].astype(np.float64)))

    return syndrome, energy


def _compute_energy_fast(
    bits:         np.ndarray,
    check_to_var: list,
    var_to_check: list,
    n:            int,
    m:            int,
) -> tuple:
    """
    Vectorized energy computation using numpy.add.at for speed.
    """
    # Syndrome (vectorized)
    syndrome = np.array(
        [int(np.sum(bits[check_to_var[c]])) % 2 for c in range(m)],
        dtype=np.uint8
    )

    # Contribution of each check to connected variable nodes
    # check_contribution[c] = 1 - 2*syndrome[c]  (+1 if satisfied, -1 if not)
    check_contribution = (1 - 2 * syndrome.astype(np.float64))

    energy = np.zeros(n, dtype=np.float64)
    for c in range(m):
        np.add.at(energy, check_to_var[c], check_contribution[c])

    return syndrome, energy


# ---------------------------------------------------------------------------
# GDBF DECODER
# ---------------------------------------------------------------------------

def decode(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 200,
    n_flip:       int = 1,
) -> np.ndarray:
    """
    GDBF — Gradient Descent Bit Flipping decoder.

    Parameters:
      llr          : 1D array of channel LLRs, shape (n,)
                     Used only for initial hard decision
      H            : sparse parity-check matrix
      check_to_var : check → variable adjacency list
      var_to_check : variable → check adjacency list
      max_iter     : maximum iterations (GDBF needs more than Min-Sum — use 100-200)
      n_flip       : number of bits to flip per iteration (default 1 = pure GDBF)
                     set > 1 for a middle ground between GDBF and PGDBF

    Returns:
      hard_bits : decoded codeword, uint8 array of length n
    """
    n = len(llr)
    m = len(check_to_var)

    # ------------------------------------------------------------------
    # INITIALIZATION — hard decision from LLR
    # ------------------------------------------------------------------
    bits = (llr < 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # ITERATION LOOP
    # ------------------------------------------------------------------
    for iteration in range(max_iter):

        # STEP 1: Compute syndrome and energy
        syndrome, energy = _compute_energy_fast(
            bits, check_to_var, var_to_check, n, m
        )

        # STEP 2: Check if all parity equations satisfied
        if np.all(syndrome == 0):
            return bits

        # STEP 3: Find the bit(s) with MINIMUM energy (most likely wrong)
        # Flip the n_flip bits with lowest energy
        if n_flip == 1:
            v_star = int(np.argmin(energy))
            bits[v_star] ^= 1
        else:
            # Flip the n_flip bits with most negative energy
            flip_indices = np.argsort(energy)[:n_flip]
            bits[flip_indices] ^= 1

    return bits


def decode_fast(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int = 200,
    n_flip:       int = 1,
) -> np.ndarray:
    """
    Alias for decode() — same algorithm, used by simulate.py.
    GDBF and decode_fast are identical here since the algorithm
    is already lean. Main speed-up is in _compute_energy_fast.
    """
    return decode(llr, H, check_to_var, var_to_check, max_iter, n_flip)


# ---------------------------------------------------------------------------
# PGDBF DECODER — Probabilistic GDBF
# ---------------------------------------------------------------------------

def decode_pgdbf(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int   = 200,
    temperature:  float = 1.0,
    seed:         int   = None,
) -> np.ndarray:
    """
    PGDBF — Probabilistic Gradient Descent Bit Flipping.

    Each bit is flipped with probability:
      P(flip v) = sigmoid(-E(v) / temperature)
                = 1 / (1 + exp(E(v) / temperature))

    Bits with very negative energy → P close to 1.0 → almost always flip
    Bits with positive energy      → P close to 0.0 → rarely flip

    Temperature tuning:
      temperature = 1.0  — balanced (recommended starting point)
      temperature < 1.0  — more deterministic, closer to GDBF
      temperature > 1.0  — more random, better at escaping local minima
                           but risks wrong flips

    Parameters:
      temperature : controls randomness (float, default 1.0)
      seed        : random seed for reproducibility (None = random)

    Returns:
      hard_bits : decoded codeword, uint8 array of length n
    """
    n = len(llr)
    m = len(check_to_var)

    if seed is not None:
        np.random.seed(seed)

    # Hard decision initialization
    bits = (llr < 0).astype(np.uint8)

    for iteration in range(max_iter):

        # Compute syndrome and energy
        syndrome, energy = _compute_energy_fast(
            bits, check_to_var, var_to_check, n, m
        )

        if np.all(syndrome == 0):
            return bits

        # Compute flip probabilities via sigmoid
        # P(flip v) = 1 / (1 + exp(E(v) / T))
        # Clip energy to avoid overflow in exp
        clipped = np.clip(energy / temperature, -30.0, 30.0)
        flip_prob = 1.0 / (1.0 + np.exp(clipped))

        # Sample flips: each bit flips independently with its probability
        random_draw = np.random.rand(n)
        flip_mask   = (random_draw < flip_prob).astype(np.uint8)

        # Safety: if no bits selected to flip, force-flip the worst one
        if np.sum(flip_mask) == 0:
            flip_mask[int(np.argmin(energy))] = 1

        bits ^= flip_mask

    return bits


# ---------------------------------------------------------------------------
# PGDBF WITH RESTARTS (advanced — helps escape deep local minima)
# ---------------------------------------------------------------------------

def decode_pgdbf_restarts(
    llr:          np.ndarray,
    H,
    check_to_var: list,
    var_to_check: list,
    max_iter:     int   = 100,
    temperature:  float = 1.0,
    n_restarts:   int   = 3,
) -> np.ndarray:
    """
    PGDBF with multiple random restarts.

    If the decoder fails to converge in max_iter iterations,
    it restarts from the original hard decision with a new random seed.
    Returns the best result (fewest unsatisfied checks) across all restarts.

    Parameters:
      n_restarts : number of independent decode attempts (default 3)

    Useful when temperature is high and individual runs are noisy.
    """
    n = len(llr)
    m = len(check_to_var)

    best_bits     = (llr < 0).astype(np.uint8)
    best_syndrome = np.array(
        [int(np.sum(best_bits[check_to_var[c]])) % 2 for c in range(m)]
    )
    best_n_fail   = int(np.sum(best_syndrome))

    for restart in range(n_restarts):
        candidate = decode_pgdbf(
            llr, H, check_to_var, var_to_check,
            max_iter=max_iter,
            temperature=temperature,
            seed=restart * 137,
        )
        syndrome = np.array(
            [int(np.sum(candidate[check_to_var[c]])) % 2 for c in range(m)]
        )
        n_fail = int(np.sum(syndrome))

        if n_fail == 0:
            return candidate   # converged — stop early

        if n_fail < best_n_fail:
            best_bits   = candidate
            best_n_fail = n_fail

    return best_bits


# ---------------------------------------------------------------------------
# TEMPERATURE SWEEP HELPER (for tuning PGDBF)
# ---------------------------------------------------------------------------

def sweep_temperature(
    llr_set:      list,
    codewords:    list,
    H,
    check_to_var: list,
    var_to_check: list,
    temperatures: list = None,
    max_iter:     int  = 200,
    k:            int  = None,
) -> dict:
    """
    Sweep over temperature values and report BER per temperature.
    Use this to find the optimal temperature for your base matrix.

    Parameters:
      llr_set      : list of LLR arrays (one per trial)
      codewords    : list of original codewords (one per trial)
      temperatures : list of T values to test (default: 0.5, 1.0, 2.0, 3.0)
      k            : number of message bits

    Returns:
      dict: {temperature: BER}
    """
    if temperatures is None:
        temperatures = [0.5, 1.0, 2.0, 3.0]

    n = len(llr_set[0])
    if k is None:
        k = n // 2

    results = {}
    for T in temperatures:
        total_bits   = 0
        total_errors = 0
        for llr, cw in zip(llr_set, codewords):
            decoded     = decode_pgdbf(
                llr, H, check_to_var, var_to_check,
                max_iter=max_iter, temperature=T
            )
            msg_orig    = cw[n - k:]
            msg_decoded = decoded[n - k:]
            total_errors += int(np.sum(msg_orig != msg_decoded))
            total_bits   += k
        results[T] = total_errors / total_bits
        print(f"[pgdbf] temperature={T:.1f} → BER={results[T]:.4e}")

    best_T = min(results, key=results.get)
    print(f"[pgdbf] Best temperature: T={best_T} (BER={results[best_T]:.4e})")
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

    print("=== decoder_gdbf.py self-test ===\n")

    bmat_test = np.array([
        [ 0,  1,  2, -1,  3],
        [-1,  0,  1,  2, -1],
        [ 1, -1,  0, -1,  2],
    ])
    Zc_test = 4

    H               = expand_base_matrix(bmat_test, Zc=Zc_test)
    H_sp            = sp.csr_matrix(H)
    check_to_var, var_to_check = build_adjacency(H)
    encode_fn, k, n = get_systematic_encoder(H_sp)

    print(f"Code: k={k}, n={n}\n")
    np.random.seed(42)
    n_trials = 100

    # --- GDBF tests ---
    print("--- GDBF ---")
    ok_high = sum(
        1 for _ in range(n_trials)
        if syndrome_check(H_sp, decode_fast(
            compute_llr(add_awgn(bpsk_modulate(encode_fn(
                np.random.randint(0, 2, k, dtype=np.uint8)
            )), 6.0), 6.0),
            H, check_to_var, var_to_check, max_iter=200
        ))
    )
    print(f"High SNR (6dB): {ok_high}/{n_trials} frames decoded correctly")

    ok_med = sum(
        1 for _ in range(n_trials)
        if syndrome_check(H_sp, decode_fast(
            compute_llr(add_awgn(bpsk_modulate(encode_fn(
                np.random.randint(0, 2, k, dtype=np.uint8)
            )), 3.0), 3.0),
            H, check_to_var, var_to_check, max_iter=200
        ))
    )
    print(f"Med  SNR (3dB): {ok_med}/{n_trials} frames decoded correctly")

    # --- PGDBF tests ---
    print("\n--- PGDBF ---")
    np.random.seed(42)
    ok_pgdbf = sum(
        1 for _ in range(n_trials)
        if syndrome_check(H_sp, decode_pgdbf(
            compute_llr(add_awgn(bpsk_modulate(encode_fn(
                np.random.randint(0, 2, k, dtype=np.uint8)
            )), 6.0), 6.0),
            H, check_to_var, var_to_check,
            max_iter=200, temperature=1.0
        ))
    )
    print(f"High SNR (6dB): {ok_pgdbf}/{n_trials} frames decoded correctly")

    # --- Temperature sweep ---
    print("\n--- Temperature sweep (SNR=4dB, 50 trials) ---")
    np.random.seed(99)
    llr_set, cw_set = [], []
    for _ in range(50):
        msg      = np.random.randint(0, 2, k, dtype=np.uint8)
        codeword = encode_fn(msg)
        tx       = bpsk_modulate(codeword)
        rx       = add_awgn(tx, snr_db=4.0)
        llr_set.append(compute_llr(rx, snr_db=4.0))
        cw_set.append(codeword)

    sweep_temperature(
        llr_set, cw_set, H, check_to_var, var_to_check,
        temperatures=[0.5, 1.0, 2.0, 3.0], k=k
    )

    print("\nSelf-test complete.")