"""Capture IQ samples using an SDR device.

An ATSC 3.0 channel occupies 6 MHz, so the capture device must be wider than
that before resampling.  The SDRplay RSP1B (via the ``sdrbindings`` CPython
extension around the SoapySDR C API) qualifies; RTL-SDR does not (2.4 MHz) and
is not supported.  The HackRF and Airspy paths were removed: they are not
available on this setup, and the SDRplay delivers a wider dynamic range.
"""

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

#: Default gain elements for the SDRplay RSP1B (A/322 8.1: the capture must
#: deliver the full 6 MHz channel without clipping; these are the device's
#: IFGR/RFGR elements, tuned on the RF33 lighthouse).
SDRPLAY_IFGR_DEFAULT = 40
SDRPLAY_RFGR_DEFAULT = 4

#: The SoapySDR device driver name for the SDRplay.
SDRPLAY_DRIVER = 'sdrplay'

#: Baseband filter bandwidth for a 6 MHz ATSC 3.0 channel (A/322 Annex N).
SDRPLAY_BANDWIDTH_HZ = 8_000_000


@dataclass(frozen=True)
class DetectedSdr:
    """Which SDR front ends are usable on this host."""
    sdrplay: bool

    def __bool__(self) -> bool:
        return self.sdrplay


def _bindings():
    """Import and return the ``sdrbindings`` module, or None if unavailable."""
    try:
        import sdrbindings
    except ImportError:
        return None
    return sdrbindings


def capture(
    freq_hz: int,
    sample_rate: int,
    output_file: str,
    duration_sec: int = 10,
    gain_db: float = None,
    device_type: str = None,
    device_index: int = 0,
    rf_gain_db: float = None,
    cs8: bool = False,
):
    """
    Capture IQ samples using the SDRplay.

    Args:
        freq_hz: Center frequency in Hz
        sample_rate: Sample rate in Hz
        output_file: Output file path
        duration_sec: Capture duration in seconds
        gain_db: IF gain (IFGR); device default if None
        device_type: Only 'sdrplay' is accepted
        device_index: Device index (for multiple SDRplay units)
        rf_gain_db: RF gain (RFGR)
        cs8: Write int8 IQ instead of native int16

    Returns:
        str: Path to captured file
    """
    if not _detect_sdr_tools():
        raise RuntimeError(
            "No SDRplay found. Install the SDRplay driver (SoapySDR) and "
            "build/install the sdrbindings extension."
        )

    if device_type and device_type != SDRPLAY_DRIVER:
        raise RuntimeError(f"Unsupported device type: {device_type}")

    _capture_sdrplay(freq_hz, sample_rate, gain_db, duration_sec,
                     output_file, device_index, rf_gain_db, cs8)
    return output_file


def _detect_sdr_tools() -> DetectedSdr:
    """Detect the SDRplay front end (via sdrbindings or SoapySDRUtil)."""
    available = (_bindings() is not None) or bool(shutil.which('SoapySDRUtil'))
    return DetectedSdr(sdrplay=available)


def _capture_sdrplay(freq_hz, sample_rate, gain_db, duration_sec, output_file,
                     device_index=0, rf_gain_db=None, cs8=False):
    """Capture using the SDRplay through the ``sdrbindings`` extension."""
    bindings = _bindings()
    if bindings is None:
        raise RuntimeError(
            "sdrbindings is not installed; build it in ../sdrbindings "
            "(make) and install with `pip install .`")
    ifgr = int(gain_db) if gain_db is not None else SDRPLAY_IFGR_DEFAULT
    rfgr = int(rf_gain_db) if rf_gain_db is not None else SDRPLAY_RFGR_DEFAULT
    bindings.capture_iq(
        freq_hz,
        output_file,
        rate_hz=sample_rate,
        bandwidth_hz=SDRPLAY_BANDWIDTH_HZ,
        duration_sec=duration_sec,
        ifgr=ifgr,
        rfgr=rfgr,
        driver=SDRPLAY_DRIVER,
        index=device_index,
        cs8=cs8,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Capture IQ samples from the SDRplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atsc3-capture -f 587 -o out/capture.iq              # RSP1B with defaults
  atsc3-capture -f 587 -g 45 --rf-gain 3              # tuned on RF33
  atsc3-capture -f 605 --duration 30 --cs8            # int8 output

The SDRplay RSP1B is driven through sdrbindings (SoapySDR).  RTL-SDR
(2.4 MHz) is too narrow for the 6 MHz ATSC 3.0 channel and is not supported.
        """
    )

    parser.add_argument("-f", "--freq", type=float, required=True,
                        help="Frequency in MHz")
    parser.add_argument("-o", "--output", default="capture.iq",
                        help="Output file path")
    parser.add_argument("-d", "--duration", type=int, default=10,
                        help="Duration in seconds")
    parser.add_argument("-g", "--gain", type=float, default=None,
                        help="IF gain (IFGR); default if not specified")
    parser.add_argument("--rf-gain", type=float, default=None,
                        help="RF gain (RFGR) for the SDRplay")
    parser.add_argument("-i", "--device-index", type=int, default=0,
                        help="Device index (for multiple SDRplay units)")
    parser.add_argument("-s", "--sample-rate", type=int, default=None,
                        help="Sample rate in Hz (default 10 MS/s)")
    parser.add_argument("--cs8", action="store_true",
                        help="Write int8 IQ instead of native int16")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    freq_hz = int(args.freq * 1e6)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = args.sample_rate if args.sample_rate else 10_000_000

    try:
        capture(
            freq_hz=freq_hz,
            sample_rate=sample_rate,
            output_file=str(output),
            duration_sec=args.duration,
            gain_db=args.gain,
            device_index=args.device_index,
            rf_gain_db=args.rf_gain,
            cs8=args.cs8,
        )
        print(f"✓ Captured {args.duration}s at {args.freq} MHz → {output}")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
