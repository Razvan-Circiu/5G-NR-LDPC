"""""
ldpc_utils.py — Shared utilities for 5G NR LDPC project
Person 1 — Infrastructure

FIXES vs original:
  - add_awgn / compute_llr now accept code_rate for correct Eb/N0 axis
  - build_adjacency returns .copy() arrays to prevent mutation of H internals
  - syndrome_check uses int32 to avoid uint8 overflow on large H
"""

import numpy as np
import scipy.sparse as sp


def load_base_matrix(filepath: str) -> np.ndarray:
    rows = []
    in_matrix = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'qc_base_matrix':
                in_matrix = True
                continue
            if in_matrix and line:
                values = list(map(int, line.split()))
                rows.append(values)
    bmat = np.array(rows, dtype=int)
    print(f"[ldpc_utils] Base matrix loaded: shape {bmat.shape}")
    return bmat


def expand_base_matrix(bmat: np.ndarray, Zc: int = 384) -> sp.csr_matrix:
    """
    Expand base matrix to full H (sparse CSR).
    Entry -1 → zero block. Entry k → identity shifted right by k.
    """
    mb, nb = bmat.shape
    row_idx, col_idx = [], []

    for i in range(mb):
        for j in range(nb):
            k = bmat[i, j]
            if k == -1:
                continue
            block_rows = np.arange(Zc) + i * Zc
            block_cols = (np.arange(Zc) + k) % Zc + j * Zc
            row_idx.append(block_rows)
            col_idx.append(block_cols)

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    data    = np.ones(len(row_idx), dtype=np.uint8)
    H = sp.coo_matrix((data, (row_idx, col_idx)),
                      shape=(mb * Zc, nb * Zc)).tocsr()
    print(f"[ldpc_utils] H expanded: shape {H.shape}, nnz={H.nnz}")
    return H


def syndrome_check(H: sp.csr_matrix, codeword: np.ndarray) -> bool:
    """True if (H @ codeword) mod 2 == 0. Uses int32 to avoid overflow."""
    s = H.dot(codeword.astype(np.int32)) % 2
    return bool(np.all(s == 0))


def compute_syndrome(H: sp.csr_matrix, bits: np.ndarray) -> np.ndarray:
    return H.dot(bits.astype(np.int32)) % 2


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """0 → +1,  1 → -1"""
    return 1.0 - 2.0 * bits.astype(np.float64)


def bpsk_demodulate(received: np.ndarray) -> np.ndarray:
    return (received < 0).astype(np.uint8)


def add_awgn(signal: np.ndarray, snr_db: float,
             code_rate: float = 1.0) -> np.ndarray:
    """
    Add AWGN. snr_db = Eb/N0 in dB. code_rate = k/n of the LDPC code.
    sigma = 1 / sqrt(2 * snr_linear * code_rate)

    FIX: original formula ignored code_rate → x-axis was Es/N0, not Eb/N0.
    Pass code_rate=k/n for a correct Eb/N0 comparison between decoders.
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    sigma = 1.0 / np.sqrt(2.0 * snr_linear * code_rate)
    return signal + sigma * np.random.randn(*signal.shape)


def compute_llr(received: np.ndarray, snr_db: float,
                code_rate: float = 1.0) -> np.ndarray:
    """
    LLR = (2 / sigma^2) * y.  Convention: LLR > 0 → bit = 0.
    FIX: must use same code_rate as add_awgn().
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    sigma2 = 1.0 / (2.0 * snr_linear * code_rate)
    return (2.0 / sigma2) * received


def count_bit_errors(original: np.ndarray, decoded: np.ndarray) -> int:
    return int(np.sum(original != decoded))


def build_adjacency(H: sp.csr_matrix):
    """
    Build check_to_var and var_to_check adjacency lists.
    FIX: returns .copy() so decoders don't mutate H's internal index arrays.
    """
    H_csr = H.tocsr()
    H_csc = H.tocsc()
    m, n  = H.shape

    check_to_var = [H_csr.indices[H_csr.indptr[c]:H_csr.indptr[c+1]].copy()
                    for c in range(m)]
    var_to_check = [H_csc.indices[H_csc.indptr[v]:H_csc.indptr[v+1]].copy()
                    for v in range(n)]

    print(f"[ldpc_utils] Adjacency built: {m} check nodes, {n} variable nodes")
    return check_to_var, var_to_check


if __name__ == "__main__":
    print("=== ldpc_utils self-test ===\n")
    bmat_test = np.array([[ 0,  1, -1], [-1,  2,  0]])
    H = expand_base_matrix(bmat_test, Zc=4)
    print(f"H dense:\n{H.toarray()}\n")

    bits = np.array([0, 1, 0, 1])
    mod  = bpsk_modulate(bits)
    assert np.all(bpsk_demodulate(mod) == bits), "BPSK failed"
    print("BPSK OK")

    llr = compute_llr(mod, snr_db=3.0, code_rate=0.5)
    assert np.all((llr < 0).astype(np.uint8) == bits), "LLR failed"
    print("LLR OK")

    llr_r1 = compute_llr(mod, snr_db=3.0, code_rate=1.0)
    # R=0.5 → larger sigma → smaller LLR magnitudes (more noise per coded bit)
    # R=1.0 → smaller sigma → larger LLR magnitudes
    assert np.all(np.abs(llr) < np.abs(llr_r1)), "code_rate scaling wrong"
    print("code_rate scaling OK: R=0.5 gives smaller LLRs than R=1.0 ✓")
    print("\nAll self-tests passed.")