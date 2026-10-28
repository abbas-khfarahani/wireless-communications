"""OFDM receiver: the inverse of the transmitter chain."""

import numpy as np

from .config import CP_LENGTH, N_C
from .modulation import (
    apply_dpsk_decode,
    complex_to_indices,
    qpsk_indices_to_bits,
)
from .transmitter import serial_to_matrix


def remove_cyclic_prefix(cp_matrix: np.ndarray) -> np.ndarray:
    """Drop the leading cyclic-prefix samples of each row."""
    return cp_matrix[:, CP_LENGTH:]


def fft_ofdm_symbols(time_matrix: np.ndarray) -> np.ndarray:
    """FFT each row. This inverts the transmitter IFFT."""
    return np.fft.fft(time_matrix, axis=1)


def extract_data_subcarriers(freq_matrix: np.ndarray) -> np.ndarray:
    """Take the positive-frequency data bins, indices 1 through n_c."""
    return freq_matrix[:, 1 : N_C + 1]


def receive_frame(
    rx_serial: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_data_symbols: int,
    n_padding_symbols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Demodulate one OFDM data frame and return the recovered bits."""
    cp_matrix = serial_to_matrix(rx_serial, n_rows, n_cols)
    time_matrix = remove_cyclic_prefix(cp_matrix)
    freq_matrix = fft_ofdm_symbols(time_matrix)
    data_freq = extract_data_subcarriers(freq_matrix)

    rx_indices = complex_to_indices(data_freq)
    decoded_indices = apply_dpsk_decode(rx_indices)

    data_indices = decoded_indices[1:, :].reshape(-1, order="C")
    recovered_indices = data_indices[:n_data_symbols]
    recovered_bits = qpsk_indices_to_bits(recovered_indices)

    return recovered_bits, decoded_indices


def compute_ber(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    """Bit error rate over the common length."""
    n = min(len(original_bits), len(recovered_bits))
    if n == 0:
        return 0.0
    errors = np.count_nonzero(original_bits[:n] != recovered_bits[:n])
    return errors / n


def receive_packet(
    rx_signal: np.ndarray,
    meta: dict,
    start_index: int = 0,
) -> dict:
    """Extract the data frame from a received packet and compute the BER.

    start_index is the sample where the leading header block begins, after synchronization.
    """
    header_len = meta["header_length"]
    frame_len = meta["single_frame_length"]

    data_start = start_index + header_len
    data_end = data_start + frame_len
    if data_end > len(rx_signal):
        raise ValueError(
            f"Not enough samples for data frame: need {data_end}, have {len(rx_signal)}"
        )
    rx_frame = rx_signal[data_start:data_end]

    n_rows = meta["n_rows"]
    n_cols = meta["n_cols"]
    n_data = meta["n_data_symbols"]
    n_pad = meta["n_padding_symbols"]

    recovered_bits, decoded_indices = receive_frame(
        rx_frame, n_rows, n_cols, n_data, n_pad
    )
    original_bits = meta["original_bits"]
    ber = compute_ber(original_bits, recovered_bits)

    return {
        "ber": ber,
        "recovered_bits": recovered_bits,
        "decoded_indices": decoded_indices,
        "data_start": data_start,
        "data_end": data_end,
    }
