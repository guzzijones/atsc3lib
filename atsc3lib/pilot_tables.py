"""Per-FFT pilot and data-cell tables (A/322 7.2.6, Annex D/F).

The numbers are inlined in :mod:`atsc3lib.pilot_data`, generated from the
pinned independent transcription and gated by the constant-data-carrier
identity (A/322 8.1.4.1).  This module is the single accessor: the receive
chain never carries a pattern-specific literal.

Contents (all keyed by FFT size, and cred_coeff where the table varies with it):
    noc          Number of carriers NoC (Table 7.1)
    common_cp    Common continual pilots, absolute indices (Tables D.1.1-D.1.3)
    additional_cp Additional continual pilots, relative (Table D.1.4/D.1.5)
    avail_data   Available data cells per data symbol (Tables 7.3/7.4)
    sbs_total    Total data cells in a subframe boundary symbol (7.5/7.6)
    sbs_active   Active data cells in an SBS (Annex F), by L1D_SPB
"""

from typing import Dict, List

from . import pilot_data


def _pair(key: str):
    fft, cred = key.split(',')
    return int(fft), int(cred)


def _triple(key: str):
    fft, cred, spb = key.split(',')
    return int(fft), int(cred), int(spb)


NOC: Dict[tuple, int] = {_pair(k): v for k, v in pilot_data.NOC.items()}

COMMON_CP: Dict[int, List[int]] = {int(f): v for f, v in
                                   pilot_data.COMMON_CP.items()}

ADDITIONAL_CP: Dict[int, Dict[str, Dict[int, List[int]]]] = {
    int(f): {p: {int(c): v for c, v in cc.items()} for p, cc in pp.items()}
    for f, pp in pilot_data.ADDITIONAL_CP.items()}

AVAIL_DATA: Dict[tuple, Dict[str, int]] = {
    _pair(k): v for k, v in pilot_data.AVAIL_DATA.items()}

SBS_TOTAL: Dict[tuple, Dict[str, int]] = {
    _pair(k): v for k, v in pilot_data.SBS_TOTAL.items()}

SBS_ACTIVE: Dict[tuple, Dict[str, int]] = {
    _triple(k): v for k, v in pilot_data.SBS_ACTIVE.items()}

SOURCE: str = pilot_data.SOURCE


def noc(fft_size: int, cred_coeff: int) -> int:
    """NoC for an FFT size and cred_coeff (Table 7.1)."""
    return NOC[(fft_size, cred_coeff)]


def common_cp_relative(fft_size: int, cred_coeff: int) -> List[int]:
    """Common continual pilots as relative carrier indices (A/322 8.1.4.1)."""
    from . import spec
    n = noc(fft_size, cred_coeff)
    origin = (spec.NOC_MAX[fft_size] - n) // 2
    return [c - origin for c in COMMON_CP[fft_size] if origin <= c < origin + n]


def additional_cp(fft_size: int, pattern: str, cred_coeff: int) -> tuple:
    """Additional continual pilots, relative (Table D.1.4/D.1.5)."""
    return tuple(ADDITIONAL_CP[fft_size][pattern][cred_coeff])


def avail_data(fft_size: int, cred_coeff: int, pattern: str) -> int:
    """Available data cells per data symbol (Tables 7.3/7.4)."""
    return AVAIL_DATA[(fft_size, cred_coeff)][pattern]


def sbs_total(fft_size: int, cred_coeff: int, pattern: str) -> int:
    """Total data cells in a subframe boundary symbol (Tables 7.5/7.6)."""
    return SBS_TOTAL[(fft_size, cred_coeff)][pattern]


def sbs_active(fft_size: int, cred_coeff: int, spb: int, pattern: str) -> int:
    """Active data cells in an SBS (Annex F) for a pilot-boost value."""
    return SBS_ACTIVE[(fft_size, cred_coeff, spb)][pattern]
