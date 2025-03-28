import numpy as np
import matplotlib.pyplot as plt

from ofdmlib.frame import Frame
from ofdmlib.modulation import Modulation
from ofdmlib.channel import Channel
from ofdmlib.schmidlAndCox import SchmidlAndCoxBasic, SchmidlAndCoxAvg, SchmidlAndCoxAvgR1, SchmidlAndCoxAvgR2

# Frame creation
frame = Frame(K=1024, CP=128, CP_preamble=128, M=1, N=1, preamble_mod="BPSK", payload_mod="QPSK", verbose=True, random_seed=3572128029)

# Modulation
mod = Modulation(frame)
mod.modulate()
plt.show()

# Save
frame.save("sent.txt")

##########################
# PASS TO PHYSICAL CHANNEL
##########################

# # Load
# frame.load("sent.txt")

# # Run the schmidl and cox algorithm
# sync = SchmidlAndCoxAvgR1(frame, None, verbose=True)
# sync.run(threshold=0.5, min=128, width=129)

# # Print the sync error
# print(f"Sync error: {sync.get_sync_error()}")
# print(f"True sync point: {sync.get_sync_point()} - ({sync.get_sync_point() - frame.channel.STO} without STO) - ({sync.sync} without delay)")

# # Plot the sync
# sync.plot(limitate=True)
# plt.show()