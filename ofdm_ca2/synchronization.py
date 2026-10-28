"""Frame synchronization by correlation against the known header."""

import numpy as np


def correlate_header(
    rx_signal: np.ndarray,
    header_template: np.ndarray,
) -> np.ndarray:
    """Cross-correlation of the received signal with one header frame, energy-normalized."""
    corr = np.correlate(rx_signal, header_template, mode="valid")
    energy = np.sqrt(np.sum(header_template**2) * np.sum(rx_signal[: len(header_template)] ** 2) + 1e-12)
    return corr / energy


def detect_frame_start(
    rx_signal: np.ndarray,
    header_template: np.ndarray,
) -> int:
    """Sample index where the leading header block starts.

    The transmitter sends eight header frames at the start and eight at the end.
    Correlation against one header produces several peaks. The first peak that
    reaches 85% of the maximum is the start, so the trailing copy is not chosen.
    """
    corr = correlate_header(rx_signal, header_template)
    if corr.size == 0:
        return 0

    peak = float(np.max(corr))
    threshold = 0.85 * peak
    strong = np.where(corr >= threshold)[0]
    if strong.size == 0:
        return int(np.argmax(corr))

    return int(strong[0])


def add_random_delay(
    signal: np.ndarray,
    max_delay: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Insert a random noise prefix before the packet. This is the unknown delay in part D."""
    delay = int(rng.integers(0, max_delay + 1))
    prefix = rng.normal(0.0, np.std(signal) + 1e-6, size=delay)
    return np.concatenate([prefix, signal]), delay
