"""System parameters for the OFDM assignment.

Values follow the CA2 specification for Wireless Communications.
"""

# IFFT and subcarriers
IFFT_LENGTH = 1024
N_C = 400  # data subcarriers
CP_FRACTION = 0.25  # cyclic prefix is 25% of the IFFT length
CP_LENGTH = int(IFFT_LENGTH * CP_FRACTION)  # 256 samples

# Payload: sfc = ceil(2^13 / n_c) symbols on each subcarrier
NUM_QPSK_SYMBOLS = 2**13  # 8192 QPSK symbols
SFC = int((NUM_QPSK_SYMBOLS + N_C - 1) // N_C)

# Frame structure
HEADER_FRAME_MULTIPLIER = 8  # header block is 8 copies of one header frame
GUARD_FRAME_ENABLED = True

# Modulation
MODULATION_ORDER = 4  # QPSK / DPSK alphabet
PHASES_RAD = [k * (2 * 3.141592653589793 / MODULATION_ORDER) for k in range(MODULATION_ORDER)]

# Simulation defaults
DEFAULT_SNR_DB = 20  # part A
DEFAULT_SEED = 42

# BER curves (parts B and C), 0 dB to 30 dB in steps of 2 dB
SNR_DB_RANGE = list(range(0, 31, 2))

# Part D: maximum random delay, in samples, inserted before the packet
MAX_RANDOM_DELAY_SAMPLES = 5000
