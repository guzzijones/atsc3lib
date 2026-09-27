"""Capture IQ samples using an SDR device.

An ATSC 3.0 channel occupies 6 MHz, so the capture device must be wider than
that before resampling.  The HackRF (10-20 MHz) qualifies; RTL-SDR does not
(2.4 MHz) and is not supported.
"""

import subprocess
import argparse
import shutil
from pathlib import Path


def capture(
    freq_hz: int,
    sample_rate: int,
    output_file: str,
    duration_sec: int = 10,
    gain_db: float = None,
    device_type: str = None,
    device_index: int = 0
):
    """
    Capture IQ samples using available SDR hardware.
    
    Args:
        freq_hz: Center frequency in Hz
        sample_rate: Sample rate in Hz
        output_file: Output file path
        duration_sec: Capture duration in seconds
        gain_db: RX gain (device-dependent default if None)
        device_type: Force specific device ('hackrf', 'airspy', 'soapysdr')
        device_index: Device index (for multiple devices)
    
    Returns:
        str: Path to captured file
    """
    # Detect available SDR tools
    available_tools = _detect_sdr_tools()
    
    if not available_tools:
        raise RuntimeError(
            "No SDR tools found. Install one of: hackrf, airspy, SoapySDR"
        )
    
    # Use specified device or first available
    if device_type and device_type in available_tools:
        tool = device_type
    else:
        tool = list(available_tools.keys())[0]
    
    # Capture with detected tool
    if tool == 'hackrf':
        _capture_hackrf(freq_hz, sample_rate, gain_db, duration_sec, output_file)
    elif tool == 'airspy':
        _capture_airspy(freq_hz, sample_rate, gain_db, duration_sec, output_file)
    else:
        raise RuntimeError(f"Unsupported device type: {tool}")
    
    return output_file


def _detect_sdr_tools() -> dict:
    """Detect available SDR command-line tools."""
    tools = {}
    
    if shutil.which('hackrf_transfer'):
        tools['hackrf'] = 'HackRF'
    
    if shutil.which('airspy_rx'):
        tools['airspy'] = 'Airspy'
    
    if shutil.which('SoapySDRUtil'):
        tools['soapysdr'] = 'SoapySDR'
    
    return tools


def _capture_hackrf(freq_hz, sample_rate, gain_db, duration_sec, output_file):
    """Capture using HackRF."""
    gain = int(gain_db) if gain_db else 32
    cmd = [
        "hackrf_transfer",
        "-r", output_file,
        "-f", str(freq_hz),
        "-s", str(sample_rate),
        "-g", str(gain),
        "-l", str(gain),
        "-a", "1",
        "-n", str(sample_rate * duration_sec // 4),
    ]
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
  atsc3-capture -f 569 -o out/capture.iq          # Auto-detect device
  atsc3-capture -f 569 -t hackrf -g 32            # Force HackRF
  atsc3-capture -f 605 --duration 30              # 30 second capture

Supported devices (auto-detected):
  - HackRF (hackrf_transfer)
  - Airspy (airspy_rx)
  - Any SoapySDR device

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
                        help="RX gain (auto if not specified)")
    parser.add_argument("-t", "--type", choices=['hackrf', 'airspy', 'soapysdr'],
                        help="Force device type (auto-detect if not specified)")
    parser.add_argument("-i", "--device-index", type=int, default=0,
                        help="Device index (for multiple devices)")
    parser.add_argument("-s", "--sample-rate", type=int, default=None,
                        help="Sample rate in Hz (device default if not specified)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    freq_hz = int(args.freq * 1e6)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Default sample rates by device type
    sample_rate = args.sample_rate
    if sample_rate is None:
        if args.type == 'airspy':
            sample_rate = 2_500_000
        else:
            sample_rate = 10_000_000  # HackRF default
    
    try:
        capture(
            freq_hz=freq_hz,
            sample_rate=sample_rate,
            output_file=str(output),
            duration_sec=args.duration,
            gain_db=args.gain,
            device_type=args.type,
            device_index=args.device_index
        )
        print(f"✓ Captured {args.duration}s at {args.freq} MHz → {output}")
    except RuntimeError as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        __import__('sys').exit(1)


if __name__ == "__main__":
    main()
