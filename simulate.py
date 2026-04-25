"""""
simulate.py — BER vs SNR Simulation Harness
Person 1 — Integration (corrected)

FIXES vs original:
  - load_decoders(): Gallagher B success branch printed MISSING (copy-paste bug) — fixed
  - simulate_snr_point(): passes code_rate to add_awgn/compute_llr for correct Eb/N0 axis
  - MAX_ITER per decoder: hard-decision decoders need more iterations (100-200)
  - Results saved as JSON-friendly dict, not raw numpy (easier to reload)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json

from ldpc_utils import (
    load_base_matrix,
    expand_base_matrix,
    build_adjacency,
    bpsk_modulate,
    add_awgn,
    compute_llr,
    count_bit_errors,
    syndrome_check,
)
from encoder import get_systematic_encoder


# ---------------------------------------------------------------------------
# SIMULATION PARAMETERS
# ---------------------------------------------------------------------------

BMAT_FILE  = "base_matrix/nr_5g_12.bmat"
Zc         = 384
SNR_RANGE  = np.arange(0, 6.5, 0.5)   # Eb/N0 in dB
N_TRIALS   = 200                        # frames per SNR point
MIN_ERRORS = 50                         # early-stop threshold (bit errors)

# Per-decoder max iterations — soft decoders converge fast, hard decoders need more
MAX_ITER = {
    "Flooded Min-Sum":  30,
    "Layered Min-Sum":  15,
    "Gallagher B":     100,
    "GDBF":            200,
}


# ---------------------------------------------------------------------------
# LOAD DECODER MODULES
# ---------------------------------------------------------------------------

def load_decoders() -> dict:
    """
    Import decoder functions dynamically.
    FIX: Gallagher B success branch was printing MISSING — corrected.
    """
    decoders = {}

    try:
        from decoder_flooded import decode_fast as fn
        decoders["Flooded Min-Sum"] = fn
        print("[simulate] Loaded: Flooded Min-Sum")
    except ImportError:
        print("[simulate] MISSING: decoder_flooded.py — skipping")

    try:
        from decoder_layered import decode_fast as fn
        decoders["Layered Min-Sum"] = fn
        print("[simulate] Loaded: Layered Min-Sum")
    except ImportError:
        print("[simulate] MISSING: decoder_layered.py — skipping")

    try:
        from decoder_gallagher import decode_fast as fn
        decoders["Gallagher B"] = fn
        print("[simulate] Loaded: Gallagher B")          # FIX: was printing MISSING
    except ImportError:
        print("[simulate] MISSING: decoder_gallagher.py — skipping")

    try:
        from decoder_gdbf import decode_fast as fn
        decoders["GDBF"] = fn
        print("[simulate] Loaded: GDBF")
    except ImportError:
        print("[simulate] MISSING: decoder_gdbf.py — skipping")

    return decoders


# ---------------------------------------------------------------------------
# SINGLE SNR POINT
# ---------------------------------------------------------------------------

def simulate_snr_point(encode_fn, k, n, H, check_to_var, var_to_check,
                       decode_fn, decoder_name, snr_db, code_rate,
                       n_trials, min_errors) -> float:
    """
    FIX: passes code_rate to add_awgn and compute_llr so snr_db is Eb/N0.
    """
    max_iter     = MAX_ITER.get(decoder_name, 50)
    total_bits   = 0
    total_errors = 0

    for _ in range(n_trials):
        msg      = np.random.randint(0, 2, k, dtype=np.uint8)
        codeword = encode_fn(msg)
        tx       = bpsk_modulate(codeword)
        rx       = add_awgn(tx, snr_db, code_rate=code_rate)    # FIX
        llr      = compute_llr(rx, snr_db, code_rate=code_rate)  # FIX

        decoded_cw  = decode_fn(llr, H, check_to_var, var_to_check, max_iter)
        decoded_msg = decoded_cw[n - k:]   # last k bits = message (systematic)

        total_errors += count_bit_errors(msg, decoded_msg)
        total_bits   += k

        if total_errors >= min_errors:
            break

    return total_errors / total_bits if total_bits > 0 else 1.0


# ---------------------------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------------------------

def run_simulation():
    print("\n" + "="*60)
    print("  5G NR LDPC BER Simulation  (Eb/N0 axis)")
    print("="*60 + "\n")

    print("[simulate] Loading and expanding H matrix...")
    bmat = load_base_matrix(BMAT_FILE)
    H    = expand_base_matrix(bmat, Zc=Zc)

    check_to_var, var_to_check = build_adjacency(H)
    encode_fn, k, n = get_systematic_encoder(H)
    code_rate = k / n
    print(f"[simulate] Code rate R = {k}/{n} = {code_rate:.4f}\n")

    decoders = load_decoders()
    if not decoders:
        print("[simulate] No decoders found.")
        return

    results = {}

    for name, decode_fn in decoders.items():
        print(f"\n[simulate] Running: {name}")
        ber_curve = []
        for snr in SNR_RANGE:
            t0  = time.time()
            ber = simulate_snr_point(
                encode_fn, k, n, H, check_to_var, var_to_check,
                decode_fn, name, snr, code_rate,
                N_TRIALS, MIN_ERRORS
            )
            print(f"  Eb/N0={snr:.1f} dB → BER={ber:.2e}  ({time.time()-t0:.1f}s)")
            ber_curve.append(float(ber))
        results[name] = ber_curve

    # Save
    with open("ber_results.json", "w") as f:
        json.dump({"snr_range": list(SNR_RANGE), "results": results}, f, indent=2)
    print("\n[simulate] Saved to ber_results.json")

    plot_ber(results, SNR_RANGE)


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def plot_ber(results, snr_range, save_path="ber_vs_snr.png"):
    COLORS  = {"Flooded Min-Sum": "royalblue",  "Layered Min-Sum": "darkorange",
                "Gallagher B":    "crimson",     "GDBF":           "mediumseagreen"}
    MARKERS = {"Flooded Min-Sum": "o",  "Layered Min-Sum": "s",
                "Gallagher B":    "^",  "GDBF":            "D"}

    fig, ax = plt.subplots(figsize=(9, 6))

    for name, ber_curve in results.items():
        ber_arr = np.where(np.array(ber_curve) == 0, 1e-6, ber_curve)
        ax.semilogy(snr_range, ber_arr, label=name,
                    color=COLORS.get(name, "gray"),
                    marker=MARKERS.get(name, "x"),
                    linewidth=2, markersize=6)

    ax.set_xlabel("Eb/N0 (dB)", fontsize=13)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=13)
    ax.set_title("5G NR LDPC — BER vs Eb/N0 (Zc=384)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.set_ylim([1e-5, 1.0])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[simulate] Plot saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_simulation()