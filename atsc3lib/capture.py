"""Capture IQ samples using an SDR device.

An ATSC 3.0 channel occupies 6 MHz, so the capture device must be wider than
that before resampling.  The SDRplay RSP1B (via the bundled ``soapy_capture``
helper) and the Airspy qualify; RTL-SDR does not (2.4 MHz) and is not
supported.  The HackRF path was removed: the device no longer works on this
setup, and the SDRplay delivers a wider dynamic range.
"""

import subprocess
import argparse
import shutil
import sys
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


def _soapy_capture_path() -> str:
    """Locate the bundled ``soapy_capture`` helper binary.

    Order: an existing build in the ``tools`` directory, then PATH.
    """
    candidate = Path(__file__).resolve().parent.parent / 'tools' / 'soapy_capture'
    if candidate.is_file():
        return str(candidate)
    which = shutil.which('soapy_capture')
    if which:
        return which
    return str(candidate)


def _ensure_soapy_capture_built(binary: str) -> str:
    """Build the helper if it is missing, using cc + pkg-config SoapySDR."""
    binary_path = Path(binary)
    if binary_path.is_file():
        return str(binary_path)
    cc = shutil.which('cc') or shutil.which('gcc')
    if not cc:
        raise RuntimeError(
            "soapy_capture helper is not built and no C compiler was found")
    src = Path(__file__).resolve().parent.parent / 'tools' / 'soapy_capture.c'
    if not src.is_file():
        raise RuntimeError(f"soapy_capture source not found at {src}")
    pkg = subprocess.run(
        ['pkg-config', '--cflags', '--libs', 'SoapySDR'],
        capture_output=True, text=True)
    flags = pkg.stdout.strip() if pkg.returncode == 0 else '-lSoapySDR'
    cache_dir = Path.home() / '.cache' / 'atsc3lib'
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / 'soapy_capture'
    cmd = [cc, '-O2', '-o', str(out), str(src)] + flags.split()
    subprocess.run(cmd, check=True)
    return str(out)


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
    Capture IQ samples using available SDR hardware.

    Args:
        freq_hz: Center frequency in Hz
        sample_rate: Sample rate in Hz
        output_file: Output file path
        duration_sec: Capture duration in seconds
        gain_db: RX gain (device-dependent default if None)
        device_type: Force specific device ('sdrplay', 'airspy')
        device_index: Device index (for multiple devices)
        rf_gain_db: RF gain for the SDRplay (RFGR element)
        cs8: Write down-converted int8 IQ instead of native int16 (SDRplay)

    Returns:
        str: Path to captured file
    """
    available_tools = _detect_sdr_tools()

    if not available_tools:
        raise RuntimeError(
            "No SDR tools found. Install an SDRplay (SoapySDR) or Airspy."
        )

    if device_type and device_type in available_tools:
        tool = device_type
    else:
        tool = list(available_tools.keys())[0]

    if tool == 'sdrplay':
        _capture_sdrplay(freq_hz, sample_rate, gain_db, duration_sec,
                         output_file, device_index, rf_gain_db, cs8)
    elif tool == 'airspy':
        _capture_airspy(freq_hz, sample_rate, gain_db, duration_sec, output_file)
    else:
        raise RuntimeError(f"Unsupported device type: {tool}")

    return output_file


def _detect_sdr_tools() -> dict:
    """Detect available SDR command-line tools (sdrplay preferred)."""
    tools = {}

    if shutil.which('SoapySDRUtil') or shutil.which('soapy_capture'):
        tools['sdrplay'] = 'SDRplay (SoapySDR)'

    if shutil.which('airspy_rx'):
        tools['airspy'] = 'Airspy'

    return tools


def _capture_sdrplay(freq_hz, sample_rate, gain_db, duration_sec, output_file,
                     device_index=0, rf_gain_db=None, cs8=False):
    """Capture using the SDRplay via the bundled SoapySDR helper."""
    ifgr = int(gain_db) if gain_db is not None else SDRPLAY_IFGR_DEFAULT
    rfgr = int(rf_gain_db) if rf_gain_db is not None else SDRPLAY_RFGR_DEFAULT
    binary = _ensure_soapy_capture_built(_soapy_capture_path())
    cmd = [
        binary,
        '--freq', str(int(freq_hz)),
        '--rate', str(int(sample_rate)),
        '--bw', str(SDRPLAY_BANDWIDTH_HZ),
        '--ifgr', str(ifgr),
        '--rfgr', str(rfgr),
        '--duration', str(duration_sec),
        '--out', output_file,
        '--index', str(device_index),
    ]
    if cs8:
        cmd.append('--cs8')
    subprocess.run(cmd, check=True)


def _capture_airspy(freq_hz, sample_rate, gain_db, duration_sec, output_file):
    """Capture using Airspy."""
    gain = int(gain_db) if gain_db else 10
    cmd = [
        "airspy_rx",
        "-r", output_file,
        "-f", str(freq_hz // 1_000_000),  # MHz
        "-s", str(sample_rate),
        "-g", str(gain),
        "-n", str(sample_rate * duration_sec),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Capture IQ samples from an SDR device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atsc3-capture -f 587 -o out/capture.iq            # Auto-detect device
  atsc3-capture -f 587 -t sdrplay -g 40 --rf-gain 4 # Force SDRplay + gains
  atsc3-capture -f 605 --duration 30                # 30 second capture

Supported devices (auto-detected):
  - SDRplay (SoapySDR; bundled soapy_capture helper)
  - Airspy (airspy_rx)

The device must be wider than the 6 MHz ATSC 3.0 channel; RTL-SDR (2.4 MHz)
is not wide enough and is not supported.
        """
    )

    parser.add_argument("-f", "--freq", type=float, required=True,
                        help="Frequency in MHz")
    parser.add_argument("-o", "--output", default="capture.iq",
                        help="Output file path")
    parser.add_argument("-d", "--duration", type=int, default=10,
                        help="Duration in seconds")
    parser.add_argument("-g", "--gain", type=float, default=None,
                        help="RX gain (IFGR for SDRplay; auto if not specified)")
    parser.add_argument("--rf-gain", type=float, default=None,
                        help="RF gain (RFGR) for the SDRplay")
    parser.add_argument("-t", "--type", choices=['sdrplay', 'airspy'],
                        help="Force device type (auto-detect if not specified)")
    parser.add_argument("-i", "--device-index", type=int, default=0,
                        help="Device index (for multiple devices)")
    parser.add_argument("-s", "--sample-rate", type=int, default=None,
                        help="Sample rate in Hz (device default if not specified)")
    parser.add_argument("--cs8", action="store_true",
                        help="Write down-converted int8 IQ (SDRplay) instead of native int16")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    freq_hz = int(args.freq * 1e6)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Default sample rates by device type
    sample_rate = args.sample_rate
    if sample_rate is None:
        sample_rate = 10_000_000  # SDRplay/Airspy: 10 MS/s covers the channel

    try:
        capture(
            freq_hz=freq_hz,
            sample_rate=sample_rate,
            output_file=str(output),
            duration_sec=args.duration,
            gain_db=args.gain,
            device_type=args.type,
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
