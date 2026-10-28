"""QPSK and DPSK modulation and demodulation."""

import numpy as np

from .config import MODULATION_ORDER, PHASES_RAD


def bits_to_qpsk_indices(bits: np.ndarray) -> np.ndarray:
    """Map a bit stream to QPSK indices in {0, 1, 2, 3}. Each pair of bits is one symbol."""
    n_bits = len(bits)
    if n_bits % 2 != 0:
        raise ValueError("Bit stream length must be even for QPSK.")
    pairs = bits.reshape(-1, 2)
    return (2 * pairs[:, 0] + pairs[:, 1]).astype(np.int64)


def qpsk_indices_to_bits(indices: np.ndarray) -> np.ndarray:
    """Inverse of bits_to_qpsk_indices."""
    indices = np.asarray(indices, dtype=np.int64)
    b0 = (indices // 2) & 1
    b1 = indices & 1
    return np.column_stack([b0, b1]).reshape(-1)


def indices_to_phases(indices: np.ndarray) -> np.ndarray:
    """Map a symbol index to a phase in radians (0°, 90°, 180°, 270°)."""
    return np.array(PHASES_RAD, dtype=np.float64)[np.asarray(indices, dtype=np.int64)]


def indices_to_complex(indices: np.ndarray) -> np.ndarray:
    """Build complex QPSK/DPSK symbols e^{jφ}."""
    return np.exp(1j * indices_to_phases(indices))


def complex_to_indices(symbols: np.ndarray) -> np.ndarray:
    """Hard decision: nearest QPSK index to the received phase."""
    angles = np.angle(symbols)
    angles = np.mod(angles, 2 * np.pi)
    indices = np.round(angles / (2 * np.pi / MODULATION_ORDER)).astype(np.int64) % MODULATION_ORDER
    return indices


def apply_dpsk_encode(data_matrix: np.ndarray) -> np.ndarray:
    """Column-wise differential encoding from the second row onward.

    y[n] = (x[n] + y[n-1]) mod M

    The first row is the reference and is left unchanged.
    """
    encoded = np.array(data_matrix, dtype=np.int64, copy=True)
    n_rows, _ = encoded.shape
    for row in range(1, n_rows):
        encoded[row, :] = (encoded[row, :] + encoded[row - 1, :]) % MODULATION_ORDER
    return encoded


def apply_dpsk_decode(encoded_matrix: np.ndarray) -> np.ndarray:
    """Column-wise differential decoding, the inverse of apply_dpsk_encode.

    x[n] = (y[n] - y[n-1]) mod M
    """
    encoded = np.asarray(encoded_matrix, dtype=np.int64)
    decoded = encoded.copy()
    n_rows, _ = encoded.shape
    for row in range(1, n_rows):
        decoded[row, :] = (encoded[row, :] - encoded[row - 1, :]) % MODULATION_ORDER
    return decoded
