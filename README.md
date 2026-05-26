# 5G-NR-LDPC
  Files


  Infrastructure

  ldpc_utils.py — Shared utilities
  - load_base_matrix(filepath) — parses .bmat text format into shift matrix
  - expand_base_matrix(bmat, Zc) — expands base matrix to full sparse H (CSR), each entry k → identity shifted by k
  - build_adjacency(H) — returns check_to_var and var_to_check neighbor lists
  - bpsk_modulate / bpsk_demodulate — 0→+1, 1→−1
  - add_awgn(signal, snr_db, code_rate) — AWGN channel; sigma=1/√(2·SNR·R)
  - compute_llr(received, snr_db, code_rate) — LLR = (2/σ²)·y
  - syndrome_check / compute_syndrome — int32 to avoid overflow
  - count_bit_errors — Hamming count

  Encoder

  encoder.py — Systematic encoding via GF(2) row reduction
  - gf2_row_reduce / gf2_inv — Gaussian elimination over GF(2)
  - get_systematic_encoder(H) — splits H = [H_p | H_d], computes T = H_p⁻¹·H_d, caches to parity_transform.npy. Returns encode(msg) → [parity | msg]
  - get_sparse_encoder — alternate path for large H (still uses dense gf2_inv internally)

  Decoders (all expose decode_fast(llr, H, check_to_var, var_to_check, max_iter))

  decoder_flooded.py — Flooded Min-Sum (soft)
  - All check nodes update simultaneously, then all variable nodes
  - Leave-one-out sign product (prefix/suffix) + leave-one-out min magnitude
  - decode() (loop version) and decode_fast() (vectorized)

  decoder_layered.py — Layered Min-Sum (soft)
  - Processes check rows sequentially; running llr_total updated in place per layer
  - decode_fast() includes a scaling factor 0.8 (Min-Sum correction)
  - Converges in ~half the iterations of Flooded
  - Bonus: decode_with_convergence_trace() returns per-iteration unsatisfied-check count

  decoder_gallagher.py — Gallagher B bit-flipping (hard)
  - Converts LLR → bits via sign, flips any bit whose unsatisfied-check count exceeds threshold b
  - suggest_threshold() — auto-pick b = median(check degree)/2
  - decode_adaptive() — adaptive threshold = max_count/2 each iteration
  - sweep_threshold() — BER vs b helper

  decoder_gdbf.py — Gradient Descent Bit Flipping (hard)
  - Energy E(v) = (satisfied − unsatisfied) checks around v; flip argmin(E) each iteration
  - _compute_energy_fast — vectorized via np.add.at
  - decode_pgdbf() — Probabilistic GDBF, flip prob = sigmoid(−E/T)
  - decode_pgdbf_restarts() — multiple seeded restarts, keeps best
  - sweep_temperature() — BER vs T helper

  Driver

  simulate.py — BER vs Eb/N0 harness
  - Sweep: SNR 0.0–6.0 dB step 0.5, 200 trials/point, early-stop at 50 errors
  - Per-decoder MAX_ITER: Flooded=30, Layered=15, Gallagher=100, GDBF=200
  - Dynamically imports each decode_fast; saves ber_results.json and renders ber_vs_snr.png

  Data / artifacts

  - base_matrix/nr_5g_12.bmat — 5G NR base graph (22×44, Zc=384, max col deg 18, max row deg 19)
  - parity_transform.npy — cached T = H_p⁻¹·H_d (~71 MB; first encoder run takes 30–120 s)
  - ber_results.json + ber_vs_snr.png — output of a prior run
  - QCDEC/ — local Python venv (numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.9)

  Observed results (from ber_results.json)

  - Layered Min-Sum is the clear winner: BER drops sharply after 3.5 dB (4×10⁻⁴ at 6 dB)
  - GDBF is the strongest hard-decision decoder (2.5×10⁻² at 6 dB)
  - Gallagher B improves gradually; ~6.5×10⁻² at 6 dB
  - Flooded Min-Sum looks broken — BER stays ~0.1 and even worsens at 6 dB (0.18). Likely a bug worth investigating, since it should track Layered closely.


