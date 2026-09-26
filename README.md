# atsc3lib

**Hardware-agnostic ATSC 3.0 Physical Layer Demodulation Library**

Works with any SDR: HackRF, RTL-SDR, Airspy, SDRplay, USRP, etc.

## Installation

```bash
cd atsc3lib
pip install -e .
```

## Capture IQ Samples

```bash
# Auto-detect available SDR
atsc3-capture -f 569 -o out/capture.iq

# Force specific device
atsc3-capture -f 569 -t hackrf -g 32
atsc3-capture -f 575 -t rtl -d 1
```

## Demodulate

```bash
atsc3-demod --file out/capture.iq --plot constellation
```

## Python API

```python
from atsc3lib import OFDMDemodulator, capture
import numpy as np

# Capture (auto-detects hardware)
capture(freq_hz=569_000_000, sample_rate=10_000_000, 
        output_file='capture.iq', duration_sec=10)

# Load and demodulate
iq_data = np.fromfile('capture.iq', dtype=np.int8)
iq_complex = iq_data[::2].astype(float) + 1j * iq_data[1::2].astype(float)

demod = OFDMDemodulator()
symbols = demod.demodulate(iq_complex / 128.0)  # Normalize int8
```

## Supported Hardware

- **HackRF** (via `hackrf_transfer`)
- **RTL-SDR** (via `rtl_sdr`)
- **Airspy** (via `airspy_rx`)
- **Any SoapySDR device**

The library is hardware-agnostic - it works with raw IQ samples from any source.
