import numpy as np
import matplotlib.pyplot as plt

from ofdmlib.frame import Frame
from ofdmlib.modulation import Modulation
from ofdmlib.channel import Channel
from ofdmlib.schmidlAndCox import SchmidlAndCoxBasic, SchmidlAndCoxAvg, SchmidlAndCoxAvgR1, SchmidlAndCoxAvgR2


# Frame creation
frame = Frame(K=512, CP=64, CP_preamble=64, M=1, N=4, preamble_mod="BPSK", payload_mod="QPSK", verbose=True, random_seed=3572128029)

# Modulation
mod = Modulation(frame)
mod.modulate()

# Channel
channel = Channel(frame, verbose=True, random_seed=845060418)
# channel.add_multipath([(1, 0.5), (2, 0.3), (3, 0.2)])
channel.add_sto(1000)
channel.add_noise(np.inf)

# Schmidl and Cox timing synchronization
# sync = SchmidlAndCoxBasic(frame, channel, verbose=True)
# sync.run(threshold=0.5, min=64)

# sync = SchmidlAndCoxAvg(frame, channel, verbose=True)
# sync.run(threshold=0.5, min=64, width=65)

sync = SchmidlAndCoxAvgR1(frame, channel, verbose=True)
sync.run(threshold=0.5, min=64, width=65)

# sync = SchmidlAndCoxAvgR2(frame, channel, verbose=True)
# sync.run(threshold=0.5, min=64, width=65)


print(f"Sync error: {sync.get_sync_error()}")
print(f"True sync point: {sync.get_sync_point()} - ({sync.get_sync_point() - channel.STO} without STO) - ({sync.sync} without delay)")
sync.plot(limitate=True)
plt.show()
