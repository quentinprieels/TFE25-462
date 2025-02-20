import matplotlib.pyplot as plt
import numpy as np

from ofdmlib.frame import Frame
from ofdmlib.modulation import Modulation
from ofdmlib.channel import Channel
from ofdmlib.schmidlAndCox import SchmidlAndCoxAvg
from ofdmlib.demodulation import Demodulation


# Frame creation
frame = Frame(K=512, CP=64, CP_preamble=64, M=1, N=4, preamble_mod="BPSK", payload_mod="QPSK", verbose=True, random_seed=1882612092)

# Modulation
mod = Modulation(frame)
mod.modulate()

# Channel
channel = Channel(frame, verbose=True, random_seed=199317234)
#channel.add_multipath([(1, 0.5), (2, 0.3), (3, 0.2)])
channel.add_sto(1000)
channel.add_noise(np.inf)

# Schmidl and Cox timing synchronization
sync = SchmidlAndCoxAvg(frame, channel, verbose=True)
sync.run(threshold=0.5, min=64, width=65)
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