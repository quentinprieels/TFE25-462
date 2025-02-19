import matplotlib.pyplot as plt

from ofdmlib.frame import Frame
from ofdmlib.modulation import Modulation
from ofdmlib.channel import Channel

# Frame creation
frame = Frame(K=64, CP=16, CP_preamble=32, M=4, N=4, preamble_mod="BPSK", payload_mod="QPSK")

# Modulation
mod = Modulation(frame)
mod.modulate()

# Channel
channel = Channel(frame)
channel.add_multipath([(1, 0.5), (2, 0.3), (3, 0.2)])
channel.add_sto(500)
channel.add_noise(10)
channel.plot()

plt.show()