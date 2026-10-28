"""OFDM simulation for Wireless Communications, computer assignment 2.

Part A: AWGN at 20 dB.
Part B: BER versus SNR on AWGN.
Part C: BER versus SNR on block Rayleigh fading.
Part D: frame synchronization with an unknown delay.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from ofdm_ca2.channel import (
    awgn_channel,
    equalize_rayleigh_blocks,
    rayleigh_fading_blocks,
)
from ofdm_ca2.config import (
    CP_LENGTH,
    DEFAULT_SEED,
    DEFAULT_SNR_DB,
    IFFT_LENGTH,
    MAX_RANDOM_DELAY_SAMPLES,
    N_C,
    NUM_QPSK_SYMBOLS,
    SFC,
    SNR_DB_RANGE,
)
from ofdm_ca2.receiver import receive_packet
from ofdm_ca2.synchronization import add_random_delay, detect_frame_start
from ofdm_ca2.transmitter import transmit_bits


def generate_source_bits(rng: np.random.Generator) -> np.ndarray:
    """Random payload: 2^13 QPSK symbols, so 2^14 bits."""
    return rng.integers(0, 2, size=NUM_QPSK_SYMBOLS * 2, dtype=np.int64)


def run_awgn_simulation(snr_db: float, seed: int = DEFAULT_SEED) -> dict:
    """One AWGN trial. Returns the BER and the transmitter metadata."""
    rng = np.random.default_rng(seed)
    bits = generate_source_bits(rng)
    tx, meta = transmit_bits(bits, rng)
    rx = awgn_channel(tx, snr_db, rng)
    result = receive_packet(rx, meta, start_index=0)
    result["snr_db"] = snr_db
    result["n_frames"] = meta["n_data_frames"]
    result["meta"] = meta
    return result


def run_rayleigh_simulation(
    snr_db: float,
    seed: int = DEFAULT_SEED,
    block_length: int | None = None,
) -> dict:
    """Block Rayleigh fading, zero-forcing equalization, perfect CSI."""
    rng = np.random.default_rng(seed)
    bits = generate_source_bits(rng)
    tx, meta = transmit_bits(bits, rng)

    symbol_len = IFFT_LENGTH + CP_LENGTH
    if block_length is None:
        block_length = symbol_len

    rx_faded, h_blocks = rayleigh_fading_blocks(tx, snr_db, block_length, rng)
    rx = equalize_rayleigh_blocks(rx_faded, h_blocks, block_length)

    result = receive_packet(rx, meta, start_index=0)
    result["snr_db"] = snr_db
    return result


def run_sync_simulation(snr_db: float = DEFAULT_SNR_DB, seed: int = DEFAULT_SEED) -> dict:
    """Part D: a random delay, then header detection, on AWGN."""
    rng = np.random.default_rng(seed)
    bits = generate_source_bits(rng)
    tx, meta = transmit_bits(bits, rng)
    delayed, true_delay = add_random_delay(tx, MAX_RANDOM_DELAY_SAMPLES, rng)
    rx = awgn_channel(delayed, snr_db, rng)

    detected_start = detect_frame_start(rx, meta["header_template"])
    est_delay = detected_start
    delay_error = abs(est_delay - true_delay)

    result = receive_packet(rx, meta, start_index=detected_start)
    result["true_delay"] = true_delay
    result["estimated_delay"] = est_delay
    result["delay_error"] = delay_error
    return result


def sweep_ber(
    channel: str,
    snr_values: list[float],
    n_trials: int = 5,
    seed: int = DEFAULT_SEED,
) -> tuple[list[float], list[float]]:
    """Mean BER over several trials at each SNR."""
    bers = []
    for i, snr in enumerate(snr_values):
        trial_bers = []
        for t in range(n_trials):
            trial_seed = seed + i * 1000 + t
            if channel == "awgn":
                res = run_awgn_simulation(snr, seed=trial_seed)
            else:
                res = run_rayleigh_simulation(snr, seed=trial_seed)
            trial_bers.append(res["ber"])
        bers.append(float(np.mean(trial_bers)))
    return snr_values, bers


def plot_ber_curve(
    snr_db: list[float],
    ber_awgn: list[float] | None,
    ber_rayleigh: list[float] | None,
    output_path: str,
) -> None:
    """Plot BER against SNR."""
    plt.figure(figsize=(8, 5))
    if ber_awgn is not None:
        plt.semilogy(snr_db, ber_awgn, "o-", label="AWGN", linewidth=2)
    if ber_rayleigh is not None:
        plt.semilogy(snr_db, ber_rayleigh, "s-", label="Rayleigh (unit variance)", linewidth=2)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("OFDM System BER vs SNR")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.ylim(bottom=1e-6, top=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")


def save_simulation_summary(
    res_a: dict,
    res_d: dict,
    snr_vals: list[float],
    ber_awgn: list[float],
    ber_rayleigh: list[float],
    output_path: str,
) -> None:
    """Write the numeric results used in the report."""
    summary = {
        "part_a": {
            "snr_db": DEFAULT_SNR_DB,
            "ber": res_a["ber"],
            "n_data_frames": res_a["meta"]["n_data_frames"],
            "single_frame_length": res_a["meta"]["single_frame_length"],
            "total_length": res_a["meta"]["total_length"],
            "n_padding_symbols": res_a["meta"]["n_padding_symbols"],
        },
        "part_d": {
            "true_delay": res_d["true_delay"],
            "estimated_delay": res_d["estimated_delay"],
            "delay_error": res_d["delay_error"],
            "ber": res_d["ber"],
        },
        "ber_sweep": {
            "snr_db": snr_vals,
            "ber_awgn": ber_awgn,
            "ber_rayleigh": ber_rayleigh,
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {output_path}")


def print_system_summary(meta: dict, ber: float, snr_db: float) -> None:
    """Print the part A configuration and BER."""
    n_rows = meta["n_rows"]
    print("=== Part A: AWGN Simulation @ SNR = {:.0f} dB ===".format(snr_db))
    print(f"  IFFT length     : {IFFT_LENGTH}")
    print(f"  Carriers (n_c)  : {N_C}")
    print(f"  CP length       : {CP_LENGTH} ({CP_LENGTH / IFFT_LENGTH:.0%} of IFFT)")
    print(f"  sfc             : {SFC}  (ceil(2^13 / n_c) = ceil({NUM_QPSK_SYMBOLS}/{N_C}))")
    print(f"  DPSK matrix     : {n_rows} x {N_C}  (1 reference + {SFC} data rows)")
    print(f"  QPSK symbols    : {NUM_QPSK_SYMBOLS}")
    print(f"  Data bits       : {NUM_QPSK_SYMBOLS * 2}")
    print(f"  Padding symbols : {meta['n_padding_symbols']}")
    print(f"  Data frames     : {meta['n_data_frames']}")
    print(f"  Frame length    : {meta['single_frame_length']} samples")
    print(f"  Header length   : {meta['header_length']} samples (8 x frame)")
    print(f"  Total TX length : {meta['total_length']} samples")
    print(f"  BER             : {ber:.6e}")
    print()


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    res_a = run_awgn_simulation(DEFAULT_SNR_DB)
    print_system_summary(res_a["meta"], res_a["ber"], DEFAULT_SNR_DB)

    print("=== Part B: AWGN BER sweep ===")
    snr_vals, ber_awgn = sweep_ber("awgn", SNR_DB_RANGE, n_trials=3)
    plot_ber_curve(snr_vals, ber_awgn, None, os.path.join(out_dir, "ber_awgn.png"))

    print("=== Part C: Rayleigh BER sweep ===")
    _, ber_rayleigh = sweep_ber("rayleigh", SNR_DB_RANGE, n_trials=3)
    plot_ber_curve(
        snr_vals,
        ber_awgn,
        ber_rayleigh,
        os.path.join(out_dir, "ber_awgn_rayleigh.png"),
    )

    print("=== Part D: Frame synchronization with random delay ===")
    res_d = run_sync_simulation()
    print(f"  True random delay      : {res_d['true_delay']} samples")
    print(f"  Detected header index  : {res_d['estimated_delay']} samples")
    print(f"  Delay estimation error : {res_d['delay_error']} samples")
    print(f"  BER after sync         : {res_d['ber']:.6e}")
    print()

    save_simulation_summary(
        res_a,
        res_d,
        snr_vals,
        ber_awgn,
        ber_rayleigh,
        os.path.join(out_dir, "summary.json"),
    )
    print("Simulation complete.")


if __name__ == "__main__":
    main()
