"""
encoder.py — LDPC Encoder for 5G NR
Person 1 — Encoder

Derives the generator matrix G from the expanded H matrix
and encodes message bits into valid codewords.

Strategy: Systematic encoding via GF(2) row reduction.
  H = [H_p | H_d]  (parity part | data part)
  Solve: G such that H @ G^T = 0 mod 2
  Systematic codeword: c = [msg | parity_bits]

NOTE: For large H (4608 x 9216), full dense GF(2) inversion is
memory-intensive (~320 MB for uint8). We use a practical
row-reduction approach on H_p (the square parity sub-matrix).
"""

import os

import numpy as np
import scipy.sparse as sp
from ldpc_utils import (
    load_base_matrix,
    expand_base_matrix,
    syndrome_check,
    compute_syndrome,
)


# ---------------------------------------------------------------------------
# GF(2) UTILITIES
# ---------------------------------------------------------------------------

def gf2_row_reduce(M: np.ndarray):
    """
    Gaussian elimination over GF(2) (in-place).
    Returns (M_reduced, pivot_cols) where pivot_cols[i] is the pivot
    column for row i.
    M is modified in place.
    """
    M = M.copy().astype(np.uint8)
    nrows, ncols = M.shape
    pivot_cols = []
    pivot_row = 0

    for col in range(ncols):
        # Find a row with a 1 in this column at or below pivot_row
        rows_with_one = np.where(M[pivot_row:, col] == 1)[0] + pivot_row
        if len(rows_with_one) == 0:
            continue  # No pivot in this column

        # Swap pivot row into position
        swap_row = rows_with_one[0]
        M[[pivot_row, swap_row]] = M[[swap_row, pivot_row]]

        # Eliminate all other 1s in this column
        other_rows = np.where(M[:, col] == 1)[0]
        other_rows = other_rows[other_rows != pivot_row]
        M[other_rows] ^= M[pivot_row]  # XOR = addition in GF(2)

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == nrows:
            break

    return M, pivot_cols


def gf2_inv(M: np.ndarray) -> np.ndarray:
    """
    Invert a square matrix over GF(2).
    Raises ValueError if M is singular.
    """
    n = M.shape[0]
    assert M.shape == (n, n), "Matrix must be square"

    # Augment M with identity: [M | I]
    augmented = np.hstack([M.astype(np.uint8), np.eye(n, dtype=np.uint8)])

    reduced, pivot_cols = gf2_row_reduce(augmented)

    if len(pivot_cols) < n:
        raise ValueError("Matrix is singular over GF(2) — cannot invert.")

    return reduced[:, n:]


# ---------------------------------------------------------------------------
# SYSTEMATIC ENCODER
# ---------------------------------------------------------------------------

def get_systematic_encoder(H: sp.csr_matrix):
    """
    Derive systematic encoding from H.

    Assumes H can be written as H = [H_p | H_d] where:
      H_p : m × m  (parity sub-matrix — must be invertible over GF(2))
      H_d : m × k  (data sub-matrix)

    Systematic codeword: c = [msg | parity]
    Parity bits: p = H_p^{-1} @ H_d^T @ msg  mod 2

    Returns an encode() function that takes msg bits and returns codeword.

    Parameters:
      H : full parity-check matrix (sparse), shape (m, n)

    Returns:
      encode_fn : callable  msg (k bits) → codeword (n bits)
      k         : number of message bits
      n         : codeword length
    """
    H_dense = H.toarray().astype(np.uint8)
    m, n = H_dense.shape
    k = n - m  # number of message (data) bits

    print(f"[encoder] H shape: {H_dense.shape}, k={k}, m={m}, n={n}")
    print(f"[encoder] Code rate: {k/n:.4f}")

    # Split H into parity part [first m cols] and data part [remaining k cols]
    H_p = H_dense[:, :m]   # m × m — should be invertible
    H_d = H_dense[:, m:]   # m × k

    import os

    CACHE_FILE = "parity_transform.npy"

# After computing T, save it:
    if not os.path.exists(CACHE_FILE):
        print(f"[encoder] Inverting H_p ({m}×{m}) over GF(2)...")
        H_p_inv = gf2_inv(H_p)
        T = (H_p_inv @ H_d) % 2
        np.save(CACHE_FILE, T)
        print(f"[encoder] T saved to {CACHE_FILE}")
    else:
        print(f"[encoder] Loading cached T from {CACHE_FILE}")
        T = np.load(CACHE_FILE)
        print(f"[encoder] Parity transform matrix T ready: shape {T.shape}")

    def encode(msg: np.ndarray) -> np.ndarray:
        """
        Encode a message vector of k bits.
        Returns a codeword of n bits: [parity (m bits) | message (k bits)]

        msg : 1D numpy array of length k, dtype uint8 (values 0 or 1)
        """
        assert len(msg) == k, f"Expected {k} bits, got {len(msg)}"
        parity = (T @ msg.astype(np.uint8)) % 2          # shape (m,)
        codeword = np.concatenate([parity, msg])           # shape (n,)
        return codeword.astype(np.uint8)

    return encode, k, n


# ---------------------------------------------------------------------------
# LIGHTWEIGHT ENCODE WITHOUT FULL G (for large H — uses sparse ops)
# ---------------------------------------------------------------------------

def get_sparse_encoder(H: sp.csr_matrix):
    """
    Memory-efficient encoder using sparse matrix operations.
    Same systematic approach but avoids dense H_p inversion for large matrices.

    Uses scipy.linalg to solve the system column by column in GF(2).
    This is the recommended path for the full 5G NR H matrix (4608×9216).
    """
    from scipy.linalg import lu

    H_dense = H.toarray().astype(np.float64)  # float for LU, round back to GF(2)
    m, n = H_dense.shape
    k = n - m

    H_p = H_dense[:, :m]
    H_d = H_dense[:, m:]

    # LU decomposition — use mod-2 arithmetic row by row
    # For exact GF(2), we fall back to gf2_inv (acceptable for lab scale)
    H_p_gf2 = H_p.astype(np.uint8)
    H_d_gf2 = H_d.astype(np.uint8)

    print(f"[encoder] Sparse encoder: inverting H_p ({m}×{m})...")
    H_p_inv = gf2_inv(H_p_gf2)

    T = (H_p_inv @ H_d_gf2) % 2

    def encode(msg: np.ndarray) -> np.ndarray:
        parity   = (T @ msg.astype(np.uint8)) % 2
        codeword = np.concatenate([parity, msg])
        return codeword.astype(np.uint8)

    return encode, k, n


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== encoder.py self-test ===\n")

    # --- Small manual test ---
    # H = [1 1 0 1 0]
    #     [0 1 1 0 1]
    # m=2, n=5, k=3
    H_small = np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
    ], dtype=np.uint8)
    H_sp = sp.csr_matrix(H_small)

    encode_fn, k, n = get_systematic_encoder(H_sp)
    print(f"k={k}, n={n}\n")

    # Test all 2^k messages
    errors = 0
    for val in range(2**k):
        msg = np.array([(val >> i) & 1 for i in range(k)], dtype=np.uint8)
        cw  = encode_fn(msg)
        assert len(cw) == n
        if not syndrome_check(H_sp, cw):
            print(f"  FAIL: msg={msg} → cw={cw}, syndrome={compute_syndrome(H_sp, cw)}")
            errors += 1

    if errors == 0:
        print(f"All 2^{k}={2**k} codewords passed syndrome check!\n")
    else:
        print(f"{errors} codewords FAILED!\n")

    # --- Note for full 5G NR H ---
    print("NOTE: For full 5G NR H (4608×9216), call get_systematic_encoder(H)")
    print("      H_p inversion may take 30–120 seconds for Zc=384.")
    print("      Save T to disk with np.save('parity_transform.npy', T) after first run.")