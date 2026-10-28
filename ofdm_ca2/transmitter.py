"""OFDM transmitter: bits in, a real baseband packet out."""

import numpy as np

from .config import (
    CP_LENGTH,
    GUARD_FRAME_ENABLED,
    HEADER_FRAME_MULTIPLIER,
    IFFT_LENGTH,
    N_C,
    NUM_QPSK_SYMBOLS,
    SFC,
)
from .modulation import (
    apply_dpsk_encode,
    bits_to_qpsk_indices,
    indices_to_complex,
)


def build_data_matrix(
    qpsk_indices: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Pad the QPSK indices, reshape to (sfc × n_c), and prepend a random reference row."""
    total_cells = SFC * N_C
    padded = np.zeros(total_cells, dtype=np.int64)
    n_data = min(len(qpsk_indices), total_cells)
    padded[:n_data] = qpsk_indices[:n_data]
    n_padding_symbols = total_cells - n_data

    data_matrix = padded.reshape(SFC, N_C, order="C")
    if rng is None:
        rng = np.random.default_rng()
    reference_row = rng.integers(0, 4, size=N_C, dtype=np.int64)
    matrix_with_ref = np.vstack([reference_row, data_matrix])
    return matrix_with_ref, n_padding_symbols


def map_to_ifft_bins(dpsk_complex: np.ndarray) -> np.ndarray:
    """Place data on bins 1..n_c and the conjugate on the matching negative bins.

    DC (bin 0) stays zero. Hermitian symmetry makes the IFFT real.
    """
    n_rows, _ = dpsk_complex.shape
    freq = np.zeros((n_rows, IFFT_LENGTH), dtype=np.complex128)
    for k in range(N_C):
        bin_pos = k + 1  # skip DC
        bin_neg = IFFT_LENGTH - k - 1
        freq[:, bin_pos] = dpsk_complex[:, k]
        freq[:, bin_neg] = np.conj(dpsk_complex[:, k])
    return freq


def ifft_ofdm_symbols(freq_matrix: np.ndarray) -> np.ndarray:
    """IFFT each row and keep the real part."""
    time_domain = np.fft.ifft(freq_matrix, axis=1)
    return np.real(time_domain)


def add_cyclic_prefix(time_matrix: np.ndarray) -> np.ndarray:
    """Copy the last 25% of each OFDM symbol to the front."""
    cp = time_matrix[:, -CP_LENGTH:]
    return np.hstack([cp, time_matrix])


def matrix_to_serial(matrix: np.ndarray) -> np.ndarray:
    """Parallel to serial. Each row is one time-domain OFDM symbol."""
    return matrix.reshape(-1, order="C")


def serial_to_matrix(serial: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Inverse of matrix_to_serial."""
    return serial.reshape((n_rows, n_cols), order="C")


def generate_header_frame(rng: np.random.Generator | None = None) -> np.ndarray:
    """Build one header OFDM frame.

    If no generator is passed, the header bits are drawn with seed 12345.
    The header uses a fixed all-zero DPSK reference row.
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    header_bits = rng.integers(0, 2, size=NUM_QPSK_SYMBOLS * 2, dtype=np.int64)
    header_indices = bits_to_qpsk_indices(header_bits)
    frame, _ = build_ofdm_frame(header_indices, use_fixed_reference=True)
    return frame


def build_ofdm_frame(
    qpsk_indices: np.ndarray,
    use_fixed_reference: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """One data frame: mapping, differential encoding, IFFT, and cyclic prefix.

    Returns the real serial waveform and a metadata dict.
    """
    data_matrix, n_padding = build_data_matrix(qpsk_indices, rng)
    if use_fixed_reference:
        data_matrix[0, :] = np.zeros(N_C, dtype=np.int64)

    dpsk_encoded = apply_dpsk_encode(data_matrix)
    dpsk_complex = indices_to_complex(dpsk_encoded)
    freq_matrix = map_to_ifft_bins(dpsk_complex)
    time_matrix = ifft_ofdm_symbols(freq_matrix)
    cp_matrix = add_cyclic_prefix(time_matrix)

    n_rows = cp_matrix.shape[0]
    frame_serial = matrix_to_serial(cp_matrix)

    meta = {
        "n_rows": n_rows,
        "n_cols": cp_matrix.shape[1],
        "n_padding_symbols": n_padding,
        "n_data_symbols": NUM_QPSK_SYMBOLS,
        "data_matrix": data_matrix,
        "dpsk_encoded": dpsk_encoded,
        "frame_length": len(frame_serial),
    }
    return frame_serial, meta


def assemble_transmission(
    data_frames: list[np.ndarray],
    header_frame: np.ndarray,
    include_guard: bool = GUARD_FRAME_ENABLED,
) -> tuple[np.ndarray, dict]:
    """Packet layout: [8× header][optional zero guard between data frames][data][8× header]."""
    header_block = np.tile(header_frame, HEADER_FRAME_MULTIPLIER)
    parts = [header_block]

    guard = np.zeros_like(header_frame) if include_guard else None
    for i, frame in enumerate(data_frames):
        if include_guard and i > 0:
            parts.append(guard)
        parts.append(frame)

    parts.append(header_block)
    tx = np.concatenate(parts)

    info = {
        "header_length": len(header_block),
        "single_frame_length": len(data_frames[0]),
        "n_data_frames": len(data_frames),
        "guard_length": len(guard) if guard is not None else 0,
        "total_length": len(tx),
    }
    return tx, info


def transmit_bits(
    bits: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Full transmit chain from source bits to the packet waveform."""
    qpsk_indices = bits_to_qpsk_indices(bits)
    data_frame, frame_meta = build_ofdm_frame(qpsk_indices, rng=rng)
    header_frame = generate_header_frame(rng)
    tx, asm_info = assemble_transmission([data_frame], header_frame)

    meta = {**frame_meta, **asm_info, "header_template": header_frame}
    meta["original_bits"] = bits
    meta["qpsk_indices"] = qpsk_indices
    return tx, meta
