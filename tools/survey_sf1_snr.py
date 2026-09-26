"""Quick subframe-1 pilot-SNR survey across ATSC 3.0 captures.

Detects the bootstrap, aligns to the frame, and measures the scattered-pilot
channel-estimate SNR on subframe 1 (16K / SP4_4) without running the expensive
L1 LDPC decode.  Used to find a capture strong enough for 256QAM 11/15
(~22 dB MER); see the repo docs for the link-margin context.

How to create the captures
--------------------------
Capture with an SDR at a rate at or above the main signal rate (6.912 MHz);
10 MS/s HackRF int8 is the format this tool reads by default::

    atsc3-capture -f 587 -t hackrf -g 36 -o out/rf33.iq     # RF33 = WHUT 587 MHz
    # or any SDR writing interleaved int8 I/Q at >= 6.912 MS/s

The capture must contain at least one whole frame (bootstrap + Preamble +
subframe 0 + subframe 1).  RF33's frame is ~0.20 s here, so a 2-4 s capture
(20-40 MB at 10 MS/s) is plenty.  Point the tool at the directory holding the
``.iq`` files::

    python -m tools.survey_sf1_snr --src ../out --src ../out/recapture
    python -m tools.survey_sf1_snr --src ../out --rate 10e6 --verbose
Assumptions (RF33-class multiplex; override where signalled otherwise):
    subframe 0/1 at 8K/16K, GI 1536, cred 0, SP4_2 / SP4_4, one Preamble symbol.
"""

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atsc3lib import pilot_tables, spec  # noqa: E402
from atsc3lib.bootstrap import detect_bootstrap, fine_cfo  # noqa: E402
from atsc3lib.frontend import read_hackrf_iq, resample_iq  # noqa: E402
from atsc3lib.pilot_reference import reference_sequence  # noqa: E402

LOGGER = logging.getLogger('survey_sf1_snr')

#: Subframe-1 geometry for the RF33 multiplex (16K, GI 1536, cred 0, SP4_4).
SF1_FFT, SF1_GI, SF1_CRED = 16384, 1536, 0
SF1_PATTERN = 'SP4_4'
#: Subframe-0 geometry (8K, GI 1536) and the frame's Preamble symbol count.
SF0_FFT, SF0_GI = 8192, 1536
PREAMBLE_STRUCT = 27
SF0_SYMBOLS = 35
#: Scattered-pilot SNR over these data symbols (skip the SBS at 0 and 74).
SNR_SYMBOLS = range(2, 20)

SF1_NOC = pilot_tables.noc(SF1_FFT, SF1_CRED)
SF1_DX, SF1_DY = spec.SP_DXDY[SF1_PATTERN]
SF1_CP = pilot_tables.common_cp_relative(SF1_FFT, SF1_CRED)
SF1_ADD = pilot_tables.additional_cp(SF1_FFT, SF1_PATTERN, SF1_CRED)
SF1_REF = reference_sequence(SF1_NOC).astype(np.int64)


def _timed(label, fn, *args, **kwargs):
    """Run ``fn``, logging its elapsed time at DEBUG under ``label``."""
    t = time.time()
    out = fn(*args, **kwargs)
    LOGGER.debug('%s: %.1fs', label, time.time() - t)
    return out


def pilot_snr_db(y, t0):
    """Median scattered-pilot SNR (dB) over subframe-1 symbols at ``t0``."""
    snrs = []
    for l in SNR_SYMBOLS:
        w = t0 + (SF1_FFT + SF1_GI) * l + SF1_GI
        if w + SF1_FFT > len(y):
            LOGGER.debug('symbol %d past end of buffer; stopping', l)
            break
        x = np.fft.fftshift(np.fft.fft(y[w:w + SF1_FFT]))
        origin = (spec.NOC_MAX[SF1_FFT] - SF1_NOC) // 2
        shift = SF1_FFT // 2 + (origin - (spec.NOC_MAX[SF1_FFT] - 1) // 2)
        carriers = x[shift:shift + SF1_NOC]
        pilots = np.unique(np.concatenate([
            np.arange(SF1_DX * (l % SF1_DY), SF1_NOC, SF1_DX * SF1_DY),
            SF1_CP, SF1_ADD, [0, SF1_NOC - 1]]))
        h = carriers[pilots] / (1.0 - 2.0 * SF1_REF[pilots])
        d = h[1:] - h[:-1]                       # decorrelates the channel
        noise = np.mean(np.abs(d) ** 2) / 2
        sig = np.mean(np.abs(h) ** 2) - noise
        snrs.append(10 * np.log10(max(sig, 1e-12) / max(noise, 1e-12)))
    return float(np.median(snrs)) if snrs else float('nan')


def collect(src_dirs):
    paths = []
    for d in src_dirs:
        for ext in ('*.iq', '*.cs16', '*.cfile'):
            paths += glob.glob(os.path.join(d, ext))
    return sorted(set(paths))


def analyse(path, rate, boot_span, t0_sf1, versions=None):
    """Return (structure, snr_db) for one capture, logging each step."""
    name = os.path.basename(path)
    LOGGER.info('[%s] reading capture', name)
    iq = _timed(f'[{name}] read', read_hackrf_iq, path)
    LOGGER.debug('[%s] %d samples', name, len(iq))

    LOGGER.info('[%s] resampling to bootstrap rate', name)
    boot = _timed(f'[{name}] resample->bootstrap', resample_iq, iq, rate,
                  spec.BOOTSTRAP_RATE_HZ)
    det = _timed(f'[{name}] detect bootstrap', detect_bootstrap, boot,
                 None, 8, versions)
    LOGGER.info('[%s] bootstrap structure=%d', name, det.structure)

    LOGGER.info('[%s] resampling to main rate', name)
    main_iq = _timed(f'[{name}] resample->main', resample_iq, iq, rate,
                     spec.MAIN_RATE_HZ)
    start = int(round(det.start * spec.MAIN_RATE_HZ / spec.BOOTSTRAP_RATE_HZ))
    cfo = _timed(f'[{name}] fine cfo', fine_cfo, boot, det.start)
    y = main_iq[start:]
    if cfo:
        n = np.arange(len(y))
        y = (y * np.exp(-1j * 2 * np.pi * cfo * n / spec.MAIN_RATE_HZ))
    LOGGER.info('[%s] cfo=%+.1f Hz; measuring subframe-1 pilots', name, cfo)

    snr = _timed(f'[{name}] pilot snr', pilot_snr_db, y, t0_sf1)
    return det.structure, snr


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', action='append', default=None,
                    help='directory of captures (repeatable); default ../out '
                         'and ../out/recapture')
    ap.add_argument('--rate', type=float, default=10e6,
                    help='capture sample rate in Hz (default 10e6)')
    ap.add_argument('--limit', type=int, default=0,
                    help='stop after this many files (0 = all)')
    ap.add_argument('--versions', default=None,
                    help='comma-separated bootstrap major.minor hypotheses, '
                         'e.g. 0.0 (RF33); default all 16')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='DEBUG logging (per-step timings)')
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='WARNING only')
    args = ap.parse_args(argv)

    level = logging.DEBUG if args.verbose else (
        logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=level, format='%(levelname)s %(message)s')

    base = os.path.join(os.path.dirname(__file__), '..', '..', 'out')
    src_dirs = args.src or [base, os.path.join(base, 'recapture')]
    paths = collect(src_dirs)
    if not paths:
        LOGGER.error('no captures found in: %s', ', '.join(src_dirs))
        return 1

    boot_span = int(round(spec.BOOTSTRAP_TOTAL_SAMPLES
                          * spec.MAIN_RATE_HZ / spec.BOOTSTRAP_RATE_HZ))
    pre = spec.PREAMBLE_STRUCTURE[PREAMBLE_STRUCT]
    t0_sf1 = (boot_span + (pre.fft + pre.gi)
              + (SF0_FFT + SF0_GI) * SF0_SYMBOLS)
    LOGGER.info('scanning %d capture(s); subframe 1 at frame sample %d, '
                'pattern %s', len(paths), t0_sf1, SF1_PATTERN)

    versions = None
    if args.versions:
        versions = [tuple(int(x) for x in v.split('.'))
                    for v in args.versions.split(',')]

    results = []
    for path in paths:
        try:
            structure, snr = analyse(path, args.rate, boot_span, t0_sf1,
                                     versions=versions)
            results.append((os.path.basename(path), structure, snr))
        except Exception as exc:  # noqa: BLE001
            if args.verbose:
                LOGGER.exception('[%s] failed', os.path.basename(path))
            else:
                LOGGER.warning('[%s] skipped: %s: %s',
                               os.path.basename(path), type(exc).__name__, exc)
        if args.limit and len(results) >= args.limit:
            break

    print('\n== subframe-1 pilot SNR summary ==')
    for name, structure, snr in sorted(results, key=lambda r: -r[2]):
        print(f'{name:28s} struct={structure:2d}  SNR {snr:5.1f} dB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
