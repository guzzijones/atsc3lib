"""ATSC 3.0 physical-layer constants, in one place, with spec citations.

Every number that a receiver has to know is defined here exactly once, with a
reference to the ATSC A/322 clause or table it comes from.  Modules import from
here instead of carrying unexplained literals, so the receive chain can be read
top-to-bottom and checked against the standard.

Edition referenced: ATSC A/322:2024-04 "Physical Layer Protocol".
"""

from dataclasses import dataclass
from typing import Dict, List

# ---------------------------------------------------------------------------
# Sample rates (A/322 Annex N.2.2, A/321)
# ---------------------------------------------------------------------------
# The bootstrap is fixed at 6.144 MHz. The main signal runs at
# 0.384 * (bsr_coefficient + 16) MHz; the usual bsr_coefficient is 2, giving
# 6.912 MHz.
BOOTSTRAP_RATE_HZ = 6_144_000
BSR_COEFFICIENT = 2
MAIN_RATE_HZ = int(0.384e6 * (BSR_COEFFICIENT + 16))  # 6_912_000

# ---------------------------------------------------------------------------
# Bootstrap structure (A/321 / A/322 Section 7.2)
# ---------------------------------------------------------------------------
BOOTSTRAP_FFT_SIZE = 2048       # fixed 2048-point FFT, independent of the frame
BOOTSTRAP_B_SIZE = 504          # B segment length, samples
BOOTSTRAP_C_SIZE = 520          # C segment length, samples
BOOTSTRAP_NUM_SYMBOLS = 4       # 4 bootstrap symbols per frame
BOOTSTRAP_SYMBOL_SAMPLES = (
    BOOTSTRAP_C_SIZE + BOOTSTRAP_FFT_SIZE + BOOTSTRAP_B_SIZE)  # 3072
BOOTSTRAP_TOTAL_SAMPLES = BOOTSTRAP_SYMBOL_SAMPLES * BOOTSTRAP_NUM_SYMBOLS  # 12288

ZADOFF_CHU_LENGTH = 1499        # ZC sequence length (A/322 7.2.2)
BOOTSTRAP_LEFT_NULLS = 275      # null carriers to the left of the ZC sequence
BOOTSTRAP_VALID_SIGNALLING_BITS = 8

# Zadoff-Chu root q by bootstrap major version (A/322 7.2.2).
ZADOFF_CHU_ROOT = {0: 137, 1: 197}

# PN LFSR seed per (major, minor) bootstrap version (A/322 Table 7.1).
BOOTSTRAP_SEEDS: Dict[tuple, int] = {
    (0, 0): 0x19D, (0, 1): 0x0ED, (0, 2): 0x1E8, (0, 3): 0x0E8,
    (0, 4): 0x0FB, (0, 5): 0x021, (0, 6): 0x054, (0, 7): 0x0EC,
    (1, 0): 0xF110, (1, 1): 0x3D21, (1, 2): 0xE550, (1, 3): 0xBD49,
    (1, 4): 0x23CF, (1, 5): 0x0B50, (1, 6): 0x3D3C, (1, 7): 0xA216,
}

# ---------------------------------------------------------------------------
# Frame geometry (A/322 Table 7.1, Table 8.9)
# ---------------------------------------------------------------------------
# NoCmax = number of carriers at cred_coeff = 0, per FFT size (Table 7.1).
NOC_MAX = {8192: 6913, 16384: 13825, 32768: 27649}
# Each step of cred_coeff removes C_UNIT carriers (Table 7.1).
C_UNIT = {8192: 96, 16384: 192, 32768: 384}

# Guard interval lengths in samples per FFT size (A/322 Table 8.9 / H.1.1).
GUARD_INTERVALS = {
    8192: [192, 384, 512, 768, 1024, 1536, 2048],
    16384: [192, 384, 512, 768, 1024, 1536, 2048, 2432, 3072, 3648, 4096],
    32768: [192, 384, 512, 768, 1024, 1536, 2048, 2432, 3072, 3648, 4096, 4864],
}

# A/322 Table 8.6/9.14 guard-interval signalling value -> sample length.
# Value 0 is reserved and 13..15 are reserved; 1..12 map 192..4864.  Note this
# is NOT the list index: signalling value 6 (GI6_1536) is index 5 in
# ``GUARD_INTERVALS``.  Not every value is legal for every FFT size.
GI_SAMPLES = {
    1: 192, 2: 384, 3: 512, 4: 768, 5: 1024, 6: 1536, 7: 2048,
    8: 2432, 9: 3072, 10: 3648, 11: 4096, 12: 4864,
}


def guard_interval(fft_size: int, value: int) -> int:
    """Guard-interval length for an FFT size and A/322 signalling value."""
    gi = GI_SAMPLES.get(value)
    if gi is None or gi not in GUARD_INTERVALS[fft_size]:
        raise ValueError(
            f"guard-interval value {value} is not legal for FFT {fft_size}")
    return gi


def noc(fft_size: int, cred_coeff: int) -> int:
    """Number of carriers NoC for an FFT size and cred_coeff (Table 7.1)."""
    return NOC_MAX[fft_size] - cred_coeff * C_UNIT[fft_size]


# ---------------------------------------------------------------------------
# Preamble (A/322 Section 7.2.5, Table H.1.1, Table 7.2, Table 8.6)
# ---------------------------------------------------------------------------
# The first Preamble symbol always uses the cred_coeff = 4 carrier count.
PREAMBLE_FIRST_CRED = 4

# preamble_structure -> PreambleStructure, A/322 Table H.1.1.  Each listed
# (GI, DX) pair occupies 5 consecutive structures, one per L1-Basic mode (1..5).
# Note 32K lists GI 3072 and 3648 twice, with DX 8 then DX 3, so the table is
# not a simple repeating pattern.
@dataclass(frozen=True)
class PreambleStructure:
    """One ``preamble_structure`` value (A/322 Table H.1.1)."""
    fft: int            # FFT size (8192 / 16384 / 32768)
    gi: int             # guard interval length in samples
    dx: int             # preamble pilot spacing (DY = 1)
    l1b_mode: int       # L1-Basic FEC mode (1..5)


PREAMBLE_STRUCTURE: Dict[int, PreambleStructure] = {}


def _add_structures(base, fft, groups):
    for i, (gi, dx) in enumerate(groups):
        for mode in range(1, 6):
            PREAMBLE_STRUCTURE[base + i * 5 + (mode - 1)] = PreambleStructure(
                fft=fft, gi=gi, dx=dx, l1b_mode=mode)


# 8K: structures 0..34.
_add_structures(0, 8192, [
    (192, 16), (384, 8), (512, 6), (768, 4), (1024, 3), (1536, 4), (2048, 3),
])
# 16K: structures 35..89.
_add_structures(35, 16384, [
    (192, 32), (384, 16), (512, 12), (768, 8), (1024, 6), (1536, 4),
    (2048, 3), (2432, 3), (3072, 4), (3648, 4), (4096, 3),
])
# 32K: structures 90..159, including the doubled GI 3072/3648 rows.
_add_structures(90, 32768, [
    (192, 32), (384, 32), (512, 24), (768, 16), (1024, 12), (1536, 8),
    (2048, 6), (2432, 6), (3072, 8), (3072, 3), (3648, 8), (3648, 3),
    (4096, 3), (4864, 3),
])

# Preamble pilot amplitude A_Preamble per (FFT, GI), A/322 Table 8.6.
PREAMBLE_PILOT_AMPLITUDE = {
    (8192, 192): 1.841, (8192, 384): 1.514, (8192, 512): 1.396,
    (8192, 768): 1.230, (8192, 1024): 1.109, (8192, 1536): 1.230,
    (8192, 2048): 1.109,
    (16384, 192): 2.188, (16384, 384): 1.841, (16384, 512): 1.698,
    (16384, 768): 1.514, (16384, 1024): 1.396, (16384, 1536): 1.274,
    (16384, 2048): 1.161, (16384, 2432): 1.161, (16384, 3072): 1.274,
    (16384, 3648): 1.274, (16384, 4096): 1.161,
    (32768, 192): 2.188, (32768, 384): 2.188, (32768, 512): 2.042,
    (32768, 768): 1.841, (32768, 1024): 1.698, (32768, 1536): 1.585,
    (32768, 2048): 1.445, (32768, 2432): 1.445, (32768, 3072): 1.585,
    (32768, 3648): 1.585, (32768, 4096): 1.161, (32768, 4864): 1.161,
}

# Number of available data cells in the first Preamble symbol per (FFT, GI),
# cred_coeff = 4 (A/322 Table 7.2).
PREAMBLE_DATA_CELLS_CRED4 = {
    (8192, 192): 6075, (8192, 384): 5667, (8192, 512): 5395,
    (8192, 768): 4851, (8192, 1024): 4307, (8192, 1536): 4851,
    (8192, 2048): 4307,
    (16384, 192): 12558, (16384, 384): 12150, (16384, 512): 11878,
    (16384, 768): 11334, (16384, 1024): 10790, (16384, 1536): 9702,
    (16384, 2048): 8614, (16384, 2432): 8614, (16384, 3072): 9702,
    (16384, 3648): 9702, (16384, 4096): 8614,
    (32768, 192): 25116, (32768, 384): 25116, (32768, 512): 24844,
    (32768, 768): 24300, (32768, 1024): 23756, (32768, 1536): 22764,
    (32768, 2048): 21756, (32768, 2432): 21756, (32768, 3072): 22764,
    (32768, 3648): 22764, (32768, 4096): 21756, (32768, 4864): 21756,
}

# Half-width, in samples, of the pilot-coherence search for the FFT window
# start of a symbol.  The Preamble's coherence plateau is flat across the whole
# guard interval (A/322 7.2.5.1 fixes its pilots), so the search only has to
# span a few samples either side of the bootstrap-anchored nominal position;
# the bootstrap itself has already placed the frame to within one GI.
FINE_TIMING_SPAN = 24

# Iterations of the decision-directed per-symbol common-phase correction
# (CPE), which removes each OFDM symbol's residual complex gain after
# equalisation.  Three iterations are the oracle's own count and converged on
# every banked capture; the estimator is a mean over the pool's ~6000 cells per
# symbol, so it is not iteration-limited.
CPE_ITERATIONS = 3

# ===========================================================================
# Common continual pilots (A/322 Table D.1.1)
# ===========================================================================
# CP32: the 192 absolute carrier indices of the common continual pilot set for
# the 32K FFT. The 8K set is derived from it as CP8 = ceil(CP32[4k] / 4)
# (Table D.1.3); the 16K set as CP16 = ceil(CP32[2k] / 2).
CP32 = [
    236, 316, 356, 412, 668, 716, 868, 1100, 1228, 1268, 1340, 1396,
    1876, 1916, 2140, 2236, 2548, 2644, 2716, 2860, 3004, 3164, 3236, 3436,
    3460, 3700, 3836, 4028, 4124, 4132, 4156, 4316, 4636, 5012, 5132, 5140,
    5332, 5372, 5500, 5524, 5788, 6004, 6020, 6092, 6428, 6452, 6500, 6740,
    7244, 7316, 7372, 7444, 7772, 7844, 7924, 8020, 8164, 8308, 8332, 8348,
    8788, 8804, 9116, 9140, 9292, 9412, 9436, 9604, 10076, 10204, 10340,
    10348, 10420, 10660, 10684, 10708, 11068, 11132, 11228, 11356, 11852,
    11860, 11884, 12044, 12116, 12164, 12268, 12316, 12700, 12772, 12820,
    12988, 13300, 13340, 13564, 13780, 13868, 14084, 14308, 14348, 14660,
    14828, 14876, 14948, 15332, 15380, 15484, 15532, 15604, 15764, 15788,
    15796, 16292, 16420, 16516, 16580, 16940, 16964, 16988, 17228, 17300,
    17308, 17444, 17572, 18044, 18212, 18236, 18356, 18508, 18532, 18844,
    18860, 19300, 19316, 19340, 19484, 19628, 19724, 19804, 19876, 20204,
    20276, 20332, 20404, 20908, 21148, 21196, 21220, 21556, 21628, 21644,
    21860, 22124, 22148, 22276, 22316, 22508, 22516, 22636, 23012, 23332,
    23492, 23516, 23524, 23620, 23812, 23948, 24188, 24212, 24412, 24484,
    24644, 24788, 24932, 25004, 25100, 25412, 25508, 25732, 25772, 26252,
    26308, 26380, 26420, 26548, 26780, 26932, 26980, 27236, 27292, 27332,
    27412,
]

# ===========================================================================
# Reference sequence (A/322 Section 8.1.2)
# ===========================================================================
# LFSR seed and generator G(x) = 1 + x^9 + x^10 + x^12 + x^13. The first 24
# output values are 1101 1000 0000 0001 0100 0000.
REFERENCE_SEED = 0x1B

# ===========================================================================
# CRC-32 (A/322 Section 6.1.2.2)
# ===========================================================================
# G(x) = x^32 + x^21 + x^16 + x^11 + 1, all-ones initialisation, MSB first.
CRC32_POLY = (1 << 21) | (1 << 16) | (1 << 11) | 1
CRC32_INIT = 0xFFFFFFFF

# ===========================================================================
# Baseband scrambler (A/322 Section 5.2.3, used for L1 by 6.5.2.2)
# ===========================================================================
# G(x) = 1 + x + x^3 + x^6 + x^7 + x^11 + x^12 + x^13 + x^16, initial 0xF180.
SCRAMBLER_POLY = 0xD31C        # polynomial taps as a 16-bit mask
SCRAMBLER_SEED = 0x18F         # the 8 feedback taps' bit pattern
SCRAMBLER_FIRST_BITS = "110000000110110100111111"  # spec's printed first 24

# ===========================================================================
# L1-Basic signalling protection (A/322 Section 6.5.2)
# ===========================================================================
L1B_KSIG = 200              # L1-Basic information bits (fixed)
L1B_NINNER = 16200          # LDPC codeword length
# Bit groups are 360 bits throughout ATSC 3.0 shortening/interleaving (A/322 6).
BCH_GROUP_SIZE = 360
L1B_KLDPC = 3240            # LDPC information bits (= 16200 * 3/15)
# BCH(16200) adds Mouter = 168 parity bits (A/322 Table 6.18).
L1B_MOUTER = 168
L1B_NOUTER = L1B_KSIG + L1B_MOUTER          # 368
L1B_NLDPC_PARITY = L1B_NINNER - L1B_KLDPC   # 12960
# All L1-Basic modes use LDPC rate 3/15, because Kldpc = 16200 * 3/15.
L1B_LDPC_RATE = 3

# Zero-padding (shortening) pattern, Table 6.20.
L1B_SHORTENING_PATTERN = [4, 1, 5, 2, 8, 6, 0, 7, 3]

# Group-wise parity permutation pi_p(9..44), Table 6.21.
L1B_GROUPWISE_PATTERN = [
    20, 23, 25, 32, 38, 41, 18, 9, 10, 11, 31, 24,
    14, 15, 26, 40, 33, 19, 28, 34, 16, 39, 27, 30,
    21, 44, 43, 35, 42, 36, 12, 13, 29, 22, 37, 17,
]

# L1-Basic per-mode parameters.
# Constellation: Table 6.17.  Puncturing A, B: Table 6.24.  Nrepeat: Table 6.23
# (Mode 1 only, Nrepeat = 2*floor(C*Nouter) + D with C = 0, D = 3672).
@dataclass(frozen=True)
class L1BasicMode:
    """Parameters for one L1-Basic FEC mode."""
    mode: int
    constellation: str
    eta: int             # bits per QAM symbol
    punct_a: int
    punct_b: int
    n_repeat: int


L1B_MODES: Dict[int, L1BasicMode] = {
    1: L1BasicMode(1, 'QPSK', 2, 0, 9360, 3672),
    2: L1BasicMode(2, 'QPSK', 2, 0, 11460, 0),
    3: L1BasicMode(3, 'QPSK', 2, 0, 12360, 0),
    4: L1BasicMode(4, 'NUC_16', 4, 0, 12292, 0),
    5: L1BasicMode(5, 'NUC_64', 6, 0, 12350, 0),
    6: L1BasicMode(6, 'NUC_256', 8, 0, 12432, 0),
    7: L1BasicMode(7, 'NUC_256', 8, 0, 12776, 0),
}


@dataclass(frozen=True)
class L1BasicLengths:
    """Derived L1-Basic lengths for a mode (A/322 6.5.2.7/6.5.2.8)."""
    n_fec: int           # base coded bits Nfec (before repetition)
    n_tx: int            # transmitted bits Nfec + Nrepeat (6.5.2.9)
    n_cells: int         # modulation cells = (Nfec + Nrepeat) / eta
    n_punc: int          # punctured parity bits Npunc
    n_repeat: int        # repeated bits (Mode 1 only)


def l1b_lengths(mode: int) -> L1BasicLengths:
    """Derive L1-Basic lengths for a mode (A/322 6.5.2.7/6.5.2.8).

    ``Npunc_tmp = A*(Kldpc - Nouter) + B``
    ``Nfec_tmp = Nouter + (Ninner - Kldpc) - Npunc_tmp``
    ``Nfec     = floor(Nfec_tmp / eta) * eta``
    ``Npunc    = Npunc_tmp - (Nfec_tmp - Nfec)``

    Repetition (6.5.2.7, Mode 1 only) appends ``Nrepeat`` bits, so the
    transmitted word is ``Nfec + Nrepeat`` bits; this is the count A/322 prints
    in Table 6.17 (Mode 1 = 3820 cells = 7640 bits / 2).
    """
    cfg = L1B_MODES[mode]
    n_punc_tmp = cfg.punct_a * (L1B_KLDPC - L1B_NOUTER) + cfg.punct_b
    n_fec_tmp = L1B_NOUTER + L1B_NLDPC_PARITY - n_punc_tmp
    n_fec = (n_fec_tmp // cfg.eta) * cfg.eta
    n_punc = n_punc_tmp - (n_fec_tmp - n_fec)
    n_tx = n_fec + cfg.n_repeat
    return L1BasicLengths(
        n_fec=n_fec, n_tx=n_tx, n_cells=n_tx // cfg.eta, n_punc=n_punc,
        n_repeat=cfg.n_repeat)


# ===========================================================================
# L1-Detail signalling protection (A/322 Section 6.5.2, 9.3)
# ===========================================================================
# Qldpc for the Type B parity interleaver, Table 6.7 (Ninner = 16200).
TYPE_B_QLDPC_16200 = {6: 27, 7: 24, 8: 21, 9: 18, 10: 15, 11: 12, 12: 9, 13: 6}


@dataclass(frozen=True)
class L1DetailMode:
    """Per-mode L1-Detail parameters (A/322 Tables 6.19/6.20/6.21/6.22/6.24/6.25)."""
    mode: int
    kldpc: int                 # LDPC information bits (Table 6.19)
    kseg: int                  # segmentation threshold (Table 6.25)
    rate: int                  # LDPC rate numerator / 15
    parity_interleaved: bool   # 6.5.2.6 parity interleaver applies (Modes 3-7)
    eta: int                   # bits per QAM symbol (Table 6.24)
    punct_a_num: int           # puncturing A as a fraction numerator
    punct_a_den: int           # ... over this denominator
    punct_b: int               # puncturing B
    n_ldpc_parity: int         # Ninner - Kldpc
    shortening: List[int]      # Table 6.20
    groupwise: List[int]       # Table 6.21/6.22
    groupwise_first: int       # first parity group index (9 or 18)
    repeat_c_num: int = 0      # Table 6.23 repetition C numerator (Mode 1 only)
    repeat_c_den: int = 1      # Table 6.23 repetition C denominator
    repeat_d: int = 0          # Table 6.23 repetition D (Mode 1 only)


L1D_MODES: Dict[int, L1DetailMode] = {
    1: L1DetailMode(1, 3240, 2352, 3, False, 2, 7, 2, 0, 12960,
                    [7, 8, 5, 4, 1, 2, 6, 3, 0],
                    [16, 22, 27, 30, 37, 44, 20, 23, 25, 32, 38, 41, 9, 10,
                     17, 18, 21, 33, 35, 14, 28, 12, 15, 19, 11, 24, 29, 34,
                     36, 13, 40, 43, 31, 26, 39, 42], 9,
                    repeat_c_num=61, repeat_c_den=16, repeat_d=-508),
    2: L1DetailMode(2, 3240, 3072, 3, False, 2, 2, 1, 6036, 12960,
                    [6, 1, 7, 8, 0, 2, 4, 3, 5],
                    [9, 31, 23, 10, 11, 25, 43, 29, 36, 16, 27, 34, 26, 18,
                     37, 15, 13, 17, 35, 21, 20, 24, 44, 12, 22, 40, 19, 32,
                     38, 41, 30, 33, 14, 28, 39, 42], 9),
    3: L1DetailMode(3, 6480, 6312, 6, True, 2, 11, 16, 4653, 9720,
                    [0, 12, 15, 13, 2, 5, 7, 9, 8, 6, 16, 10, 14, 1, 17, 11,
                     4, 3],
                    [19, 37, 30, 42, 23, 44, 27, 40, 21, 34, 25, 32, 29, 24,
                     26, 35, 39, 20, 18, 43, 31, 36, 38, 22, 33, 28, 41], 18),
    4: L1DetailMode(4, 6480, 6312, 6, True, 4, 29, 32, 3200, 9720,
                    [0, 15, 5, 16, 17, 1, 6, 13, 11, 4, 7, 12, 8, 14, 2, 3,
                     9, 10],
                    [20, 35, 42, 39, 26, 23, 30, 18, 28, 37, 32, 27, 44, 43,
                     41, 40, 38, 36, 34, 33, 31, 29, 25, 24, 22, 21, 19], 18),
    5: L1DetailMode(5, 6480, 6312, 6, True, 6, 3, 4, 4284, 9720,
                    [2, 4, 5, 17, 9, 7, 1, 6, 15, 8, 10, 14, 16, 0, 11, 13,
                     12, 3],
                    [19, 37, 33, 26, 40, 43, 22, 29, 24, 35, 44, 31, 27, 20,
                     21, 39, 25, 42, 34, 18, 32, 38, 23, 30, 28, 36, 41], 18),
    6: L1DetailMode(6, 6480, 6312, 6, True, 8, 11, 16, 4900, 9720,
                    [0, 15, 5, 16, 17, 1, 6, 13, 11, 4, 7, 12, 8, 14, 2, 3,
                     9, 10],
                    [20, 35, 42, 39, 26, 23, 30, 18, 28, 37, 32, 27, 44, 43,
                     41, 40, 38, 36, 34, 33, 31, 29, 25, 24, 22, 21, 19], 18),
    7: L1DetailMode(7, 6480, 6312, 6, True, 8, 49, 256, 8246, 9720,
                    [15, 7, 8, 11, 5, 10, 16, 4, 12, 3, 0, 6, 9, 1, 14, 17,
                     2, 13],
                    [44, 23, 29, 33, 24, 28, 21, 27, 42, 18, 22, 31, 32, 37,
                     43, 30, 25, 35, 20, 34, 39, 36, 19, 41, 40, 26, 38], 18),
}


@dataclass(frozen=True)
class L1DetailLengths:
    """Derived L1-Detail lengths for a mode and Ksig (A/322 6.5.2.4/6.5.2.8)."""
    mode: int
    ksig: int
    nouter: int
    kldpc: int
    ninner: int
    eta: int
    n_fec: int
    n_tx: int
    n_parity_kept: int
    n_punc: int
    n_repeat: int
    n_cells: int
    rate: int


def l1d_lengths(mode: int, ksig: int) -> L1DetailLengths:
    """Derive L1-Detail lengths for a mode and Ksig.

    Raises if Ksig exceeds Kseg (segmentation is not implemented).

    Mode 1 additionally repeats ``Nrepeat = 2*floor(C*Nouter) + D`` parity bits
    (A/322 6.5.2.7, Table 6.23); the transmitted word is ``Nfec + Nrepeat``
    bits.  Every other mode has ``Nrepeat = 0``.
    """
    m = L1D_MODES[mode]
    if ksig > m.kseg:
        raise ValueError(
            f"Ksig {ksig} > Kseg {m.kseg}: L1-Detail needs segmentation")
    nouter = ksig + L1B_MOUTER
    n_punc_tmp = int(m.punct_a_num * (m.kldpc - nouter) / m.punct_a_den) \
        + m.punct_b
    n_fec_tmp = nouter + m.n_ldpc_parity - n_punc_tmp
    n_fec = (n_fec_tmp // m.eta) * m.eta
    n_punc = n_punc_tmp - (n_fec_tmp - n_fec)
    n_repeat = 0
    if m.repeat_c_den:
        n_repeat = 2 * int(m.repeat_c_num * nouter // m.repeat_c_den) \
            + m.repeat_d
        n_repeat = max(n_repeat, 0)
    n_tx = n_fec + n_repeat
    return L1DetailLengths(
        mode=mode, ksig=ksig, nouter=nouter, kldpc=m.kldpc, ninner=L1B_NINNER,
        eta=m.eta, n_fec=n_fec, n_tx=n_tx,
        n_parity_kept=n_fec - nouter, n_punc=n_punc, n_repeat=n_repeat,
        n_cells=n_tx // m.eta, rate=m.rate)


# ===========================================================================
# Scattered pilot patterns and payload data cells (A/322 8.1.3, Tables 7.3/7.4)
# ===========================================================================
# Scattered pilot pattern -> (DX, DY), A/322 Table 8.2.
SP_DXDY = {
    'SP3_2': (3, 2), 'SP3_4': (3, 4), 'SP4_2': (4, 2), 'SP4_4': (4, 4),
    'SP6_2': (6, 2), 'SP6_4': (6, 4), 'SP8_2': (8, 2), 'SP8_4': (8, 4),
    'SP12_2': (12, 2), 'SP12_4': (12, 4), 'SP16_2': (16, 2), 'SP16_4': (16, 4),
    'SP24_2': (24, 2), 'SP24_4': (24, 4), 'SP32_2': (32, 2), 'SP32_4': (32, 4),
}

# Additional continual pilots, available data cells and subframe-boundary-symbol
# geometry are per (FFT size, pilot pattern, cred_coeff) and live in
# ``atsc3lib.pilot_tables`` (fetched from the pinned independent transcription
# and gated by the constant-data-carrier identity).  These helpers expose them.

#: The RF33-class geometry (8K, cred 0, SP4_2) as named constants for the
#: subframe-0 path and its tests.
RF33_PATTERN = 'SP4_2'
RF33_FFT = 8192
RF33_CRED = 0
SBS_TOTAL_8K_CRED0 = 5136      # Table 7.5/7.6
SBS_ACTIVE_8K_CRED0 = 5009     # Annex F (available for cell multiplexing)
SBS_NULL_8K_CRED0 = SBS_TOTAL_8K_CRED0 - SBS_ACTIVE_8K_CRED0  # 127


def additional_cp(pattern: str, fft_size: int = RF33_FFT,
                  cred_coeff: int = RF33_CRED) -> tuple:
    """Additional continual pilots, relative indices (A/322 Table D.1.4/D.1.5)."""
    from . import pilot_tables
    return pilot_tables.additional_cp(fft_size, pattern, cred_coeff)


def avail_data_cells(fft_size: int, cred_coeff: int, pattern: str) -> int:
    """Available data cells per data symbol (A/322 Tables 7.3/7.4)."""
    from . import pilot_tables
    return pilot_tables.avail_data(fft_size, cred_coeff, pattern)


def avail_data_8k(pattern: str) -> int:
    """Available data cells per data symbol for 8K, cred 0 (Tables 7.3/7.4)."""
    return avail_data_cells(RF33_FFT, RF33_CRED, pattern)


# Scattered pilot pattern signaling values (A/322 Table 9.12, SISO).
SP_PATTERN_SIGNALING = {
    0: 'SP3_2', 1: 'SP3_4', 2: 'SP4_2', 3: 'SP4_4',
    4: 'SP6_2', 5: 'SP6_4', 6: 'SP8_2', 7: 'SP8_4',
    8: 'SP12_2', 9: 'SP12_4', 10: 'SP16_2', 11: 'SP16_4',
    12: 'SP24_2', 13: 'SP24_4', 14: 'SP32_2', 15: 'SP32_4',
}

# Allowed scattered-pilot patterns per FFT size, the union over guard intervals
# of A/322 Table 8.3 (SISO).  This is confirmed by the constant-data-carrier
# identity in ``tools/fetch_pilot_tables.py``: a pattern is allowed for an FFT
# size exactly when its data-carrier count is invariant across the pilot lattice
# phases, and the gate asserts that set equals this one.
ALLOWED_SP = {
    8192: frozenset({
        'SP3_2', 'SP3_4', 'SP4_2', 'SP4_4', 'SP6_2', 'SP6_4', 'SP8_2', 'SP8_4',
        'SP12_2', 'SP12_4', 'SP16_2', 'SP16_4', 'SP32_2', 'SP32_4'}),
    16384: frozenset({
        'SP3_2', 'SP3_4', 'SP4_2', 'SP4_4', 'SP6_2', 'SP6_4', 'SP8_2', 'SP8_4',
        'SP12_2', 'SP12_4', 'SP16_2', 'SP16_4', 'SP24_2', 'SP24_4',
        'SP32_2', 'SP32_4'}),
    32768: frozenset({
        'SP3_2', 'SP6_2', 'SP8_2', 'SP12_2', 'SP16_2', 'SP24_2', 'SP32_2'}),
}


def allowed_patterns(fft_size: int) -> frozenset:
    """Scattered-pilot patterns allowed for an FFT size (A/322 Table 8.3 union)."""
    return ALLOWED_SP[fft_size]


# ===========================================================================
# Convolutional Time Interleaver (A/322 7.1.4, Table 9.24)
# ===========================================================================
#: L1D_plp_CTI_depth signalling value -> Nrows (non-extended interleaving),
#: A/322 Table 9.24.  Values 4..7 are reserved.
CTI_NROWS = {0: 512, 1: 724, 2: 887, 3: 1024}

#: Same, for extended interleaving (QPSK only, never with LDM), A/322 7.1.3
#: and Table 9.24 (depth 010 -> 1254, 011 -> 1448).
CTI_NROWS_EXTENDED = {0: 512, 1: 724, 2: 1254, 3: 1448}


def cti_nrows(depth: int, extended: bool = False) -> int:
    """Number of CTI delay lines from ``L1D_plp_CTI_depth`` (A/322 Table 9.24)."""
    table = CTI_NROWS_EXTENDED if extended else CTI_NROWS
    if depth not in table:
        raise ValueError(f"L1D_plp_CTI_depth {depth} is reserved (A/322 Table 9.24)")
    return table[depth]


# ===========================================================================
# Layered Division Multiplexing (A/322 6.4)
# ===========================================================================
#: L1D_plp_ldm_injection_level signalling value -> Enhanced Layer injection
#: level below the Core Layer, in dB (A/322 Table 9.22).  Value 31 is reserved.
LDM_INJECTION_DB = {
    0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0, 7: 3.5,
    8: 4.0, 9: 4.5, 10: 5.0, 11: 6.0, 12: 7.0, 13: 8.0, 14: 9.0,
    15: 10.0, 16: 11.0, 17: 12.0, 18: 13.0, 19: 14.0, 20: 15.0,
    21: 16.0, 22: 17.0, 23: 18.0, 24: 19.0, 25: 20.0, 26: 21.0,
    27: 22.0, 28: 23.0, 29: 24.0, 30: 25.0,
}


@dataclass(frozen=True)
class LdmPower:
    """Layer power distribution for one Enhanced Layer injection level.

    Source: A/322 Table 6.15 (power ratios) and Table 6.16 (the injection-level
    controller scaling factor ``alpha`` and the power normalizer ``beta``).

    Attributes:
        injection_db: Enhanced Layer level below the Core Layer, in dB.
        core_ratio: Core Layer share of total power (Table 6.15), linear.
        enhanced_ratio: Enhanced Layer share of total power (Table 6.15), linear.
        alpha: injection-level controller scaling factor (Table 6.16).
        beta: power normalizer factor (Table 6.16).
    """
    injection_db: float
    core_ratio: float
    enhanced_ratio: float
    alpha: float
    beta: float


def _power_ratios(injection_db: float) -> tuple:
    """Core/Enhanced power split for an injection level (A/322 Table 6.15)."""
    e = 10.0 ** (-injection_db / 10.0)
    core = 1.0 / (1.0 + e)
    return core, 1.0 - core


#: A/322 Table 6.16: injection level (dB) -> (scaling factor alpha, normalizing
#: factor beta).  The Core/Enhanced power ratios follow from Table 6.15 and are
#: derived in :func:`ldm_power`.
_LDM_ALPHA_BETA = {
    0.0: (1.0000000, 0.7071068), 0.5: (0.9440609, 0.7271524),
    1.0: (0.8912509, 0.7465331), 1.5: (0.8413951, 0.7651789),
    2.0: (0.7943282, 0.7830305), 2.5: (0.7498942, 0.8000406),
    3.0: (0.7079458, 0.8161736), 3.5: (0.6683439, 0.8314061),
    4.0: (0.6309573, 0.8457262), 4.5: (0.5956621, 0.8591327),
    5.0: (0.5623413, 0.8716346), 6.0: (0.5011872, 0.8940022),
    7.0: (0.4466836, 0.9130512), 8.0: (0.3981072, 0.9290819),
    9.0: (0.3548134, 0.9424353), 10.0: (0.3162278, 0.9534626),
    11.0: (0.2818383, 0.9625032), 12.0: (0.2511886, 0.9698706),
    13.0: (0.2238721, 0.9758449), 14.0: (0.1995262, 0.9806699),
    15.0: (0.1778279, 0.9845540), 16.0: (0.1584893, 0.9876723),
    17.0: (0.1412538, 0.9901705), 18.0: (0.1258925, 0.9921685),
    19.0: (0.1122018, 0.9937642), 20.0: (0.1000000, 0.9950372),
    21.0: (0.0891251, 0.9960519), 22.0: (0.0794328, 0.9968601),
    23.0: (0.0707946, 0.9975034), 24.0: (0.0630957, 0.9980154),
    25.0: (0.0562341, 0.9984226),
}


def ldm_power(injection_db: float) -> LdmPower:
    """Layer power distribution at an Enhanced Layer injection level.

    Args:
        injection_db: Enhanced Layer level below the Core Layer (dB); one of the
            A/322 Table 9.22 values.

    Returns:
        :class:`LdmPower` (A/322 Tables 6.15 and 6.16).
    """
    if injection_db not in _LDM_ALPHA_BETA:
        raise ValueError(f"injection level {injection_db} dB not tabulated")
    core, enhanced = _power_ratios(injection_db)
    alpha, beta = _LDM_ALPHA_BETA[injection_db]
    return LdmPower(injection_db=injection_db, core_ratio=core,
                    enhanced_ratio=enhanced, alpha=alpha, beta=beta)


def cti_start_c(fec_block_start: int, start_row: int, nrows: int) -> int:
    """Solve A/322 9.3.9.1 for ``C``, the pre-CTI first-FEC-Block offset.

    The transmitter signals::

        L1D_plp_CTI_fec_block_start = C + Nrows * ((start_row + C) mod Nrows)

    which inverts to the equation below.  A valid ``C`` lies in
    ``[0, cells_per_FEC_Block)``; that bound is the CTI reading's gate.
    """
    r = (start_row + fec_block_start) % nrows
    return fec_block_start - nrows * r

# ===========================================================================
# Large tables are defined in their own modules and re-exported here so callers
# have a single import point:
#   CP32 ........................... this module (A/322 Table D.1.1)
#   LDPC Type A/B tables ........... atsc3lib.ldpc_exact
#   BCH generator polynomials ...... atsc3lib.bch
#   group interleaver tables ....... atsc3lib.group_interleaver
# ===========================================================================


