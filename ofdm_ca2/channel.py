"""Channel models: AWGN and block Rayleigh fading."""

import numpy as np


def signal_variance(signal: np.ndarray) -> float:
    """Sample variance of the real baseband signal."""
    return float(np.var(signal))


def noise_std_from_snr(signal: np.ndarray, snr_db: float) -> float:
    """AWGN standard deviation σ = σ_s / sqrt(SNR), with SNR linear and σ_s the signal std."""
    snr_linear = 10 ** (snr_db / 10.0)
    sigma_s = np.sqrt(signal_variance(signal))
    return sigma_s / np.sqrt(snr_linear)


def awgn_channel(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add real Gaussian noise with variance σ_s² / SNR."""
    std = noise_std_from_snr(signal, snr_db)
    noise = rng.normal(0.0, std, size=signal.shape)
    return signal + noise


def rayleigh_fading_blocks(
    signal: np.ndarray,
    snr_db: float,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one CN(0, 1) coefficient per block, then AWGN.

    Returns the faded noisy signal and the coefficient of each block.
    """
    n = len(signal)
    n_blocks = int(np.ceil(n / block_length))
    out = np.zeros_like(signal, dtype=np.float64)
    h_blocks = np.zeros(n_blocks, dtype=np.complex128)

    std = noise_std_from_snr(signal, snr_db)

    for b in range(n_blocks):
        start = b * block_length
        end = min((b + 1) * block_length, n)
        h = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        h_blocks[b] = h
        block = signal[start:end]
        faded = np.real(h * block.astype(np.complex128))
        noise = rng.normal(0.0, std, size=end - start)
        out[start:end] = faded + noise

    return out, h_blocks


def equalize_rayleigh_blocks(
    rx_signal: np.ndarray,
    h_blocks: np.ndarray,
    block_length: int,
) -> np.ndarray:
    """Zero-forcing equalization with perfect channel knowledge."""
    n = len(rx_signal)
    n_blocks = len(h_blocks)
    out = np.zeros_like(rx_signal, dtype=np.float64)
    for b in range(n_blocks):
        start = b * block_length
        end = min((b + 1) * block_length, n)
        h = h_blocks[b]
        if np.abs(h) < 1e-12:
            out[start:end] = rx_signal[start:end]
        else:
            z = rx_signal[start:end].astype(np.complex128) / h
            out[start:end] = np.real(z)
    return out
