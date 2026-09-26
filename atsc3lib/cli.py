"""Command-line interface for the ATSC 3.0 receiver.

``atsc3-decode`` runs the full validated signalling chain on a capture:

    capture -> bootstrap -> Preamble -> L1-Basic -> L1-Detail -> per-PLP config

``atsc3-capture`` (see :mod:`atsc3lib.capture`) records IQ from an SDR.
"""

import argparse
import logging
import sys

from . import spec
from .receiver import decode_capture, decode_plp_payload

logger = logging.getLogger(__name__)


def _print_result(result):
    if result.l1_basic is not None:
        lb = result.l1_basic
        print(f"  L1-Basic: version {lb.version}, CRC {'OK' if lb.crc_ok else 'FAIL'}")
        print(f"    subframes           : {lb.num_subframes + 1}")
        print(f"    preamble symbols    : {lb.preamble_num_symbols + 1}")
        print(f"    L1-Detail fec type  : {lb.l1_detail_fec_type} (mode {lb.l1_detail_fec_type + 1})")
        print(f"    L1-Detail size      : {lb.l1_detail_size_bytes} bytes")
        print(f"    L1-Detail total cells: {lb.l1_detail_total_cells}")
    if result.l1_detail is not None:
        ld = result.l1_detail
        print(f"  L1-Detail: version {ld.version}, BSID {ld.bsid}, "
              f"CRC {'OK' if ld.crc_ok else 'FAIL'}")
        fft = {0: '8K', 1: '16K', 2: '32K'}
        print(f"  Per-PLP configuration ({len(result.plps)} PLP(s)):")
        for sf_idx, plp in result.plps:
            print(f"    subframe {sf_idx}  PLP {plp.plp_id:<3d} "
                  f"layer={plp.layer} start={plp.start} size={plp.size} "
                  f"fec={plp.fec_type} mod={plp.modulation} cod={plp.code_rate} "
                  f"TI={plp.ti_mode}")


def decode_main(argv=None):
    """CLI for the full signalling decode chain."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="ATSC 3.0 receiver: decode L1 signalling and PLP config")
    parser.add_argument('file', help='IQ capture file path')
    parser.add_argument('--rate', type=float, required=True,
                        help='capture sample rate in Hz (e.g. 10e6)')
    parser.add_argument('--fmt', default='auto',
                        choices=['auto', 'cs8', 'cs16', 'cf32'],
                        help="sample format (default: cs8, HackRF int8)")
    parser.add_argument('--max-iterations', type=int, default=100,
                        help='LDPC iteration cap')
    parser.add_argument('--plp', type=int, default=None,
                        help='also decode this subframe-0 PLP id (default: '
                             'smallest subframe-0 PLP)')
    parser.add_argument('--no-payload', action='store_true',
                        help='decode signalling only, skip the PLP payload')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    result = decode_capture(args.file, args.rate, fmt=args.fmt,
                            max_iterations=args.max_iterations)

    print(f"Capture: {args.file} @ {args.rate/1e6:.3f} MHz")
    _print_result(result)
    if result.l1_basic is None:
        print(f"  FAILED: {result.error}")
        return 1
    if result.l1_detail is None:
        print(f"  L1-Detail FAILED: {result.error}")
        return 1

    if not args.no_payload:
        from .frontend import read_hackrf_iq
        iq = read_hackrf_iq(args.file)
        _, payload = decode_plp_payload(
            iq, args.rate, plp_id=args.plp,
            max_iterations=args.max_iterations, result=result)
        if payload is None:
            print("  Payload: no subframe-0 PLP decoded")
        else:
            print(f"  Payload: PLP {payload.plp_id}, "
                  f"{payload.n_converged}/{payload.n_fec} FEC blocks converged")
            for pkt in payload.baseband_packets[:4]:
                print(f"    Baseband Packet: {len(pkt)} bytes")
    return 0


def main(argv=None):
    return decode_main(argv)


if __name__ == '__main__':
    sys.exit(main())
