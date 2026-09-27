"""Command-line interface for the ATSC 3.0 receiver.

``atsc3-decode`` runs the full validated signalling chain on a capture:

    capture -> bootstrap -> Preamble -> L1-Basic -> L1-Detail -> per-PLP config

``atsc3-capture`` (see :mod:`atsc3lib.capture`) records IQ from an SDR.
"""

import argparse
import logging
import sys

from . import spec
from .receiver import decode_capture, decode_plp_payload, decode_cti_plp_streams
from .payload import TI_CTI

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
                        help='also decode this PLP id (default: smallest layer-0 '
                             'PLP of --subframe)')
    parser.add_argument('--subframe', type=int, default=0,
                        help='subframe whose PLP to decode (default 0)')
    parser.add_argument('--no-payload', action='store_true',
                        help='decode signalling only, skip the PLP payload')
    parser.add_argument('--frames', type=int, default=6,
                        help='frames to gather for a CTI-mode (TI mode 1) PLP')
    parser.add_argument('--no-fine-timing', action='store_true',
                        help='do not refine the FFT window off the scattered '
                             'pilots (A/322 8.1.3.1)')
    parser.add_argument('--no-cpe', action='store_true',
                        help='do not run the decision-directed per-symbol '
                             'common-phase correction')
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
        from .payload import decode_streams
        iq = read_hackrf_iq(args.file)
        sf = result.l1_detail.subframes[args.subframe]
        cti = any(p.ti_mode == TI_CTI for p in sf['plps'] if p.layer == 0)
        if cti:
            result, decoded, streams = decode_cti_plp_streams(
                iq, args.rate, plp_id=args.plp, n_frames=args.frames,
                max_iterations=args.max_iterations, result=result,
                fine_timing=not args.no_fine_timing,
                cpe=not args.no_cpe)
            if decoded is None:
                print(f"  Payload: no CTI PLP decoded in subframe {args.subframe}")
                return 0
            print(f"  Payload: PLP {decoded.payload.plp_id}, CTI Nrows "
                  f"{decoded.nrows}, C {decoded.c_offset}, "
                  f"{decoded.payload.n_converged}/{decoded.n_blocks} FEC "
                  f"blocks converged")
        else:
            _, payload = decode_plp_payload(
                iq, args.rate, plp_id=args.plp, subframe=args.subframe,
                max_iterations=args.max_iterations, result=result)
            if payload is None:
                print(f"  Payload: no PLP decoded in subframe {args.subframe}")
                return 0
            print(f"  Payload: PLP {payload.plp_id}, "
                  f"{payload.n_converged}/{payload.n_fec} FEC blocks converged")
            streams = decode_streams(payload)
        print(f"  Streams: {len(streams.packets)} ALP packet(s), "
              f"{len(streams.datagrams)} UDP datagram(s), "
              f"{len(streams.lls)} LLS table(s)")
        for t in streams.lls:
            print(f"    LLS table 0x{t.table_id:02x} ({t.name}): "
                  f"{len(t.data)} bytes")
        if streams.alp_stats.resync:
            print(f"    (ALP resyncs: {streams.alp_stats.resync})")
    return 0


def main(argv=None):
    return decode_main(argv)


if __name__ == '__main__':
    sys.exit(main())
