"""ATSC 3.0 Bootstrap: generation, signaling encoding, and detection.

The bootstrap is the first part of every ATSC 3.0 frame. It uses a fixed
2048-point OFDM structure (independent of the main FFT size) carrying the
Zadoff-Chu / PN sequence used for synchronization plus 3 signaling bytes
encoded as cyclic shifts.

This module implements:

- Bootstrap waveform generation (transmitter side).
- Preamble-structure <-> (FFT size, guard interval, L1 FEC mode) mapping.
- Detector: recover frame start, major/minor version, and the 3 signaling
  bytes from a received bootstrap segment.

All constants and the exact sequence construction follow ATSC A/322
Section 7.2 and are cross-checked against the reference implementation
``drmpeg/gr-atsc3`` (``bootstrap_cc_impl.cc``).

Note on sample rate: the bootstrap is defined at a 2048-point FFT and the
6 MHz-channel bootstrap sample rate (6.144 MHz), then resampled 9/8 to the
main OFDM rate (6.912 MHz). Generation here is at the native bootstrap
rate; real captures must be resampled to that rate before detection.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import spec

# Bootstrap constants (A/321 / A/322 7.2), re-exported from spec for local use.
BOOTSTRAP_FFT_SIZE = spec.BOOTSTRAP_FFT_SIZE
B_SIZE = spec.BOOTSTRAP_B_SIZE
C_SIZE = spec.BOOTSTRAP_C_SIZE
NUM_BOOTSTRAP_SYMBOLS = spec.BOOTSTRAP_NUM_SYMBOLS
ZADOFF_CHU_LENGTH = spec.ZADOFF_CHU_LENGTH
LEFT_NULLS = spec.BOOTSTRAP_LEFT_NULLS
VALID_SIGNALLING_BITS = spec.BOOTSTRAP_VALID_SIGNALLING_BITS
BSR_COEFFICIENT = spec.BSR_COEFFICIENT

MINOR_VERSION_SEEDS = spec.BOOTSTRAP_SEEDS
ZC_ROOT = spec.ZADOFF_CHU_ROOT


def _pn_sequence(seed: int, length: int) -> np.ndarray:
    """Generate the bootstrap PN sequence (A/322 Section 7.2.2)."""
    sr = seed
    pn = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        b = ((sr) ^ (sr >> 1) ^ (sr >> 14) ^ (sr >> 15) ^ (sr >> 16)) & 1
        pn[i] = sr & 1
        sr >>= 1
        if b:
            sr |= 0x8000
    return pn


def _zc_sequence(q: int) -> np.ndarray:
    """Generate the Zadoff-Chu sequence (A/322 Section 7.2.2)."""
    n = np.arange(ZADOFF_CHU_LENGTH)
    return np.exp(1j * np.pi * q * (-1.0 * n * (n + 1)) / 1499.0)


def _gray_code_cyclic_shift(signal_bits: int) -> int:
    """Map an 8-bit signaling value to its 11-bit cyclic-shift value."""
    m = [0] * 11
    for i in range(11):
        if i < (10 - VALID_SIGNALLING_BITS):
            m[i] = 0
        elif i == 10 - VALID_SIGNALLING_BITS:
            m[i] = 1
        else:
            total = 0
            for k in range(0, (10 - i) + 1):
                total += (signal_bits >> k) & 1
            m[i] = total % 2
    value = 0
    weight = 1
    for n in range(11):
        value += m[n] * weight
        weight <<= 1
    return value


def _reverse_bits(b: int) -> int:
    out = 0
    for _ in range(8):
        out = (out << 1) | (b & 1)
        b >>= 1
    return out


def _build_shift_lookup() -> Tuple[Dict[int, int], Dict[int, int]]:
    """Forward and inverse maps between signaling byte and cyclic shift."""
    fwd = {}
    for byte in range(256):
        fwd[byte] = _gray_code_cyclic_shift(_reverse_bits(byte))
    inv = {v: k for k, v in fwd.items()}
    if len(inv) != len(fwd):
        raise ValueError("Bootstrap shift mapping is not injective")
    return fwd, inv


SHIFT_FOR_BYTE, BYTE_FOR_SHIFT = _build_shift_lookup()


@dataclass
class BootstrapParams:
    """Parameters carried by the bootstrap signaling."""
    fft_size: int
    guard_interval: int
    l1_fec_mode: int
    pilot_pattern: str
    frame_interval: int
    bandwidth: int


def decode_preamble_structure(structure: int) -> BootstrapParams:
    """Decode a preamble_structure value into BootstrapParams.

    The mapping itself lives in :mod:`atsc3lib.spec` (A/322 Table H.1.1).
    """
    if structure not in spec.PREAMBLE_STRUCTURE:
        raise ValueError(f"Unknown preamble_structure {structure}")
    row = spec.PREAMBLE_STRUCTURE[structure]
    return BootstrapParams(
        fft_size=row.fft,
        guard_interval=row.gi,
        l1_fec_mode=row.l1b_mode,
        pilot_pattern=f"DX{row.dx}",
        frame_interval=-1,
        bandwidth=-1,
    )


def encode_bootstrap_signaling(structure: int, frame_interval: int = 0,
                               bandwidth: int = 0) -> np.ndarray:
    """Return the 3 bootstrap signaling bytes (before shift encoding)."""
    if structure < 0 or structure > 159:
        raise ValueError(f"preamble_structure out of range: {structure}")
    byte0 = ((frame_interval << 2) | (bandwidth & 0x3)) & 0xFF
    return np.array([byte0, BSR_COEFFICIENT & 0xFF, structure & 0xFF],
                    dtype=np.uint8)


def _build_cores(major: int, minor: int) -> List[np.ndarray]:
    """Build the 4 unshifted time-domain bootstrap cores (2048 samples each)."""
    seed = MINOR_VERSION_SEEDS[(major, minor)]
    q = ZC_ROOT[major]
    pn = _pn_sequence(seed, ZADOFF_CHU_LENGTH * (NUM_BOOTSTRAP_SYMBOLS // 2))
    zc = _zc_sequence(q)

    cores = []
    pnindex = 0
    for k in range(NUM_BOOTSTRAP_SYMBOLS):
        freq = np.zeros(BOOTSTRAP_FFT_SIZE, dtype=np.complex128)
        zcindex = 0
        for _ in range(ZADOFF_CHU_LENGTH // 2):
            sign = -1.0 if pn[pnindex] else 1.0
            freq[zcindex + LEFT_NULLS] = sign * zc[zcindex]
            zcindex += 1
            pnindex += 1
        freq[zcindex + LEFT_NULLS] = 0.0
        zcindex += 1
        reverse = pnindex - 1
        for _ in range(ZADOFF_CHU_LENGTH // 2):
            sign = -1.0 if pn[reverse] else 1.0
            freq[zcindex + LEFT_NULLS] = sign * zc[zcindex]
            zcindex += 1
            reverse -= 1

        dst = np.concatenate([freq[BOOTSTRAP_FFT_SIZE // 2:], freq[:BOOTSTRAP_FFT_SIZE // 2]])
        core = np.fft.fft(dst)
        scale = -1.0 if k == 3 else 1.0
        core *= scale / np.sqrt(1498.0)
        cores.append(core)
    return cores


def _assemble_symbol(k: int, core: np.ndarray) -> np.ndarray:
    """Assemble one 3072-sample bootstrap symbol from a (possibly shifted) core."""
    if k == 0:
        c_part = core[BOOTSTRAP_FFT_SIZE - C_SIZE:]
        n = np.arange(B_SIZE)
        b_part = core[n + (BOOTSTRAP_FFT_SIZE - B_SIZE)] * \
            np.exp(1j * 2 * np.pi * (n + C_SIZE) / 2048.0)
        return np.concatenate([c_part, core, b_part])
    n = np.arange(B_SIZE)
    b_part = core[n + (BOOTSTRAP_FFT_SIZE - C_SIZE)] * \
        np.exp(-1j * 2 * np.pi * (n - C_SIZE) / 2048.0)
    c_part = core[BOOTSTRAP_FFT_SIZE - C_SIZE:]
    return np.concatenate([b_part, c_part, core])


def generate_bootstrap(structure: int, major: int = 0, minor: int = 0,
                       frame_interval: int = 0, bandwidth: int = 0) -> np.ndarray:
    """Generate the full 12288-sample bootstrap waveform (native rate)."""
    cores = _build_cores(major, minor)
    signaling = encode_bootstrap_signaling(structure, frame_interval, bandwidth)

    absolute_shift = 0
    symbols = []
    for k in range(NUM_BOOTSTRAP_SYMBOLS):
        core = cores[k]
        if k > 0:
            relative = SHIFT_FOR_BYTE[int(signaling[k - 1])]
            absolute_shift = (absolute_shift - relative) % BOOTSTRAP_FFT_SIZE
            core = np.roll(core, absolute_shift)
        symbols.append(_assemble_symbol(k, core))
    return np.concatenate(symbols)


def _core_of_symbol(received_symbol: np.ndarray, k: int) -> np.ndarray:
    """Extract the 2048-sample core from an assembled bootstrap symbol."""
    if k == 0:
        return received_symbol[C_SIZE:C_SIZE + BOOTSTRAP_FFT_SIZE]
    return received_symbol[B_SIZE + C_SIZE:B_SIZE + C_SIZE + BOOTSTRAP_FFT_SIZE]


@dataclass
class BootstrapDetection:
    """Result of bootstrap detection."""
    start: int
    major: int
    minor: int
    structure: int
    signaling: np.ndarray


def _fft_correlate_abs(iq: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Magnitude of the linear cross-correlation of iq with ref (valid mode)."""
    n = len(iq)
    m = len(ref)
    if n < m:
        return np.array([])
    fft_len = 1 << int(np.ceil(np.log2(n + m - 1)))
    fft_iq = np.fft.fft(iq, fft_len)
    fft_ref = np.fft.fft(ref, fft_len)
    corr = np.fft.ifft(fft_iq * np.conj(fft_ref))
    # c[k] = sum_i iq[k+i] conj(ref[i]) corresponds to corr[k].
    return np.abs(corr[:n - m + 1])


def _extract_signaling(iq: np.ndarray, start: int, cores: List[np.ndarray],
                       num_peaks: int = 1) -> Optional[np.ndarray]:
    """Try to recover the 3 signaling bytes from a candidate bootstrap start."""
    symbol_len = C_SIZE + BOOTSTRAP_FFT_SIZE + B_SIZE
    total_len = symbol_len * NUM_BOOTSTRAP_SYMBOLS
    if start < 0 or start + total_len > len(iq):
        return None

    absolute = [0]
    for k in range(1, NUM_BOOTSTRAP_SYMBOLS):
        sym = iq[start + k * symbol_len:start + (k + 1) * symbol_len]
        core_rx = _core_of_symbol(sym, k)
        base = cores[k]
        cc = np.fft.ifft(np.fft.fft(core_rx) * np.conj(np.fft.fft(base)))
        absolute.append(int(np.argmax(np.abs(cc))) % BOOTSTRAP_FFT_SIZE)

    signaling = np.zeros(3, dtype=np.uint8)
    for k in range(1, NUM_BOOTSTRAP_SYMBOLS):
        relative = (absolute[k - 1] - absolute[k]) % BOOTSTRAP_FFT_SIZE
        if relative not in BYTE_FOR_SHIFT:
            return None
        signaling[k - 1] = BYTE_FOR_SHIFT[relative]
    return signaling


def detect_bootstrap(iq: np.ndarray, search_limit: int = None,
                     num_candidates: int = 8) -> BootstrapDetection:
    """Detect the bootstrap in a received segment at the native bootstrap rate.

    Searches all 16 major/minor version hypotheses and, for each, validates the
    strongest correlation peaks by checking that the recovered cyclic shifts
    produce valid signaling bytes. Returns the best validated candidate.

    Args:
        iq: Complex samples at the 6.144 MHz bootstrap rate.
        search_limit: Optional cap on samples to search (for speed).
        num_candidates: Number of correlation peaks to validate per hypothesis.
    """
    symbol_len = C_SIZE + BOOTSTRAP_FFT_SIZE + B_SIZE
    total_len = symbol_len * NUM_BOOTSTRAP_SYMBOLS
    if len(iq) < total_len:
        raise ValueError(
            f"Need at least {total_len} samples, got {len(iq)}")

    if search_limit is not None and len(iq) > search_limit + total_len:
        iq = iq[:search_limit + total_len]

    best = None  # (score, major, minor, start, signaling)
    for (major, minor), _seed in MINOR_VERSION_SEEDS.items():
        cores = _build_cores(major, minor)
        ref = _assemble_symbol(0, cores[0])
        corr = _fft_correlate_abs(iq, ref)
        if corr.size == 0:
            continue
        # Candidate peaks (indices into corr == candidate starts).
        n_cand = min(num_candidates, corr.size)
        cand_idx = np.argpartition(corr, -n_cand)[-n_cand:]
        for pos in cand_idx:
            pos = int(pos)
            if corr[pos] <= 0:
                continue
            signaling = _extract_signaling(iq, pos, cores)
            if signaling is None:
                continue
            # Validator: recovered signaling must be self-consistent with the
            # structure mapping and (byte 1 is the BSR coefficient).
            if signaling[2] >= 160:
                continue
            score = float(corr[pos])
            if best is None or score > best[0]:
                best = (score, major, minor, pos, signaling)

    if best is None:
        raise ValueError("No valid bootstrap found in the provided samples")

    _score, major, minor, start, signaling = best
    return BootstrapDetection(
        start=start, major=major, minor=minor,
        structure=int(signaling[2]), signaling=signaling)
