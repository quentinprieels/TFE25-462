# ofdmlib

`ofdmlib` is a Python library that provides essential tools for implementing and simulating OFDM systems. It offers modules for frame construction, modulation, channel simulation, synchronization, demodulation and visualization to help you analyze and optimize OFDM system performance.

It is designed to be modular, easy to use and fit the requirement needed in this thesis. The library is built on top of the `radcomlib` library, which provides additional tools for signal processing and communication systems.

## Features

- Frame Construction: Build and manipulate OFDM frames easily.
- Modulation and Demodulation: Implement a basic modulation and demodulation scheme with cyclic prefix support.
- Channel Simulation: Simulate channel effects such as noise, sample timing offset, and multipath fading.
- Synchronization: Implement the Schmidl & Cox synchronization algorithm with and different variations.
- Visualization: Plot OFDM signals, constellation diagrams, and channel effects.

## Installation

To install the package in editable mode (ideal for development):

1. Clone the repository:

```bash
git clone https://github.com/yourusername/myofdmlib.git
cd myofdmlib
```

2. Install using pip in editable mode:

```bash
pip install -e .
```

Note: Editable mode allows you to modify the library source code and have those changes immediately available to your projects.

## Usage

After installation, you can import and use the package modules in your Python scripts or Jupyter notebooks. For example:

```python
import matplotlib.pyplot as plt

from ofdmlib.frame import Frame
from ofdmlib.modulation import Modulation
from ofdmlib.channel import Channel
from ofdmlib.synchronization import SchmidlAndCoxAvg
from ofdmlib.demodulation import Demodulation

# Frame creation
frame = Frame(K=512, CP=64, CP_preamble=64, M=1, N=4, preamble_mod="BPSK", payload_mod="QPSK", verbose=True)

# Modulation
mod = Modulation(frame)
mod.modulate()

# Channel
channel = Channel(frame, verbose=True)
channel.add_multipath([(1, 0.5), (2, 0.3), (3, 0.2)])
channel.add_sto(1000)
channel.add_noise(10)

# Schmidl and Cox timing synchronization
sync = SchmidlAndCoxAvg(frame, channel, verbose=True)
sync.run(threshold=0.5, width=64, averaged_metric="M", signal_metric="N", weight_metric="N")
# sync.plot()
print(f"Sync error: {sync.get_sync_error()}")
print(f"True sync point: {sync.get_sync_point()} - ({sync.get_sync_point() - channel.STO} without STO)")

# Demodulation
demod = Demodulation(frame, sync.get_sync_point(), channel, verbose=True)
demod.demodulate()
demod.plot_constellation("Constellation Diagram after demodulation")
print(f"BER before equalization: {demod.get_ber()}")

demod.equalize()
demod.plot_constellation("Constellation Diagram after equalization")

# BER
print(f"BER after equalization: {demod.get_ber()}")
plt.show()
```

Additional examples can be found in the [examples](examples) directory.

## Note

Adjust the example code to fit your simulation or analysis requirements.
Requirements

The package depends on the following libraries:

- numpy
- matplotlib
- seaborn
- radcomlib
- tqdm

These dependencies will be automatically installed when you install `ofdmlib` except for `radcomlib`, which you need to install manually:

```bash
pip install -e path/to/radcomlib
```
