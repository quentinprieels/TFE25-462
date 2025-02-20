import matplotlib.pyplot as plt

from ofdmlib.frame import Frame

# Plot the pilots matrix
frame = Frame(K=16, CP=4, CP_preamble=4, M=1, N=8, preamble_mod="QPSK", payload_mod="QPSK", Nt=1, Nf=3)
frame.plot(bits=True)
plt.show()
