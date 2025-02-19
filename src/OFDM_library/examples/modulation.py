import matplotlib.pyplot as plt

from ofdmlib.frame import Frame
from ofdmlib.modulation import Modulation

frame = Frame(K=64, CP=16, CP_preamble=32, M=4, N=4, preamble_mod="BPSK", payload_mod="QPSK")
mod = Modulation(frame)
mod.modulate()
mod.plot()
plt.show()
