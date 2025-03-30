import numpy as np
import matplotlib.pyplot as plt

from ofdmlib.frame import Frame
from ofdmlib.modulation import Modulation
from ofdmlib.channel import Channel
from ofdmlib.schmidlAndCox import SchmidlAndCoxBasic, SchmidlAndCoxAvg, SchmidlAndCoxAvgR1, SchmidlAndCoxAvgR2

# # Frame creation
frame = Frame(K=1024, CP=128, CP_preamble=128, M=1, N=1, preamble_mod="BPSK", payload_mod="QPSK", verbose=True, random_seed=3572128029)

# Modulation
mod = Modulation(frame)
mod.modulate()
# plt.show()

# # Save
# assert frame.len == len(frame.get_frame())
# frame.save("sent.txt")
# print("SIG_LEN: ", frame.len)
# print("2x SIG_LEN: ", 2 * frame.len)

##########################
# PASS TO PHYSICAL CHANNEL
##########################

# Fake channel
channel = Channel(frame, verbose=True, random_seed=845060418)
channel.add_sto(1)
channel.add_noise(np.inf)

# Load
frame.load("rx_sig_rx.dat")

# Resave the signal in the load format
frame.save("recv.txt")

# Run the schmidl and cox algorithm
sync = SchmidlAndCoxAvgR1(frame, channel, verbose=True)
sync.run(threshold=0.5, min=128, width=129)

# Print the sync error
# print(f"Sync error: {sync.get_sync_error()}")
# print(f"True sync point: {sync.get_sync_point()} - ({sync.get_sync_point() - frame.channel.STO} without STO) - ({sync.sync} without delay)")

# Plot the sync

sync.plot(limitate=False)
plt.show()