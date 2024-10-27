"""
The code is used to transmit an OFDM waveform with the SISO (single input,
single output) experimental setup at UCLouvain

-> How to use it:
    * Follow the steps defined in the Jupyter Notebook
    * Defined the tranmission parameters in section #2 and #3
    * Run this file and save the output file according to the Jupyter Notebook

Code from:
    * Jérôme Lafontaine
    * François De Saint Moulin
    * Martin Willame
"""

"""
#0 Packages import
"""
# The constellations
from radcomlib.comm_toolbox import BPSK_const, QPSK_const, QAM16_const

# Inverse mapping from symbols to bits
from radcomlib.comm_toolbox import inverse_mapping

# OFDM TX waveform
from radcomlib.ofdm import SISO_OFDM_DFRC_TX

# The usual imports
import matplotlib.pyplot as plt
import numpy as np

"""
#1. Subfunction definition
"""


def generate_tx(tx_sig, filename):
    """
    The function creates the <.txt> file for the transmission with the
    experimental setup. The file must be moved into RamDisk to be transmited as
    defined in the jupyter notebook "section 3: Transmission d'un signal"

    Parameters
    ----------
    tx_sig : numpy complex vector
        vector with the I/Q samples to transmits

    filename : string <filename.txt>
        name of the file with the I/Q samples to transmit

    Returns
    -------
    sig_len : int
        Number of complex samples to transmit. The value must be encoded in the
        execution command for the transmission (see section 3:
        Transmission d'un signal of the jupyter notebook)

    """
    # The number of complex symbols
    sig_len = len(tx_sig)

    # Interleaving between the real and the imaginary part
    split_sig = np.zeros((sig_len * 2))
    split_sig[::2] = np.real(tx_sig)
    split_sig[1::2] = np.imag(tx_sig)

    # Normalisation for the tranmission of the signal. We ensure here that the
    # norm of the samples to transmit is smaller than 1
    split_sig = split_sig / np.max(split_sig) * 0.7

    # Save the result
    np.savetxt(filename, split_sig)

    return sig_len


"""
#2. PARAMETERS for the OFDM transmission
"""
###############################################################################
# 2.1 These parameters define the tranmission and can be changed
###############################################################################

# Output folder for the transmission files
output_folder = "../../experiments/2048/"
# Check if the folder exists, and if not, create it
import os
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Transmission scenario name
scenario_name = "Setup_40MHz"
# Number of transmitted pulses (number of OFDM symbols)
P = 256  # default: P=256
# Bandwidth [Hz]: The maximum available bandwith is 100 MHz
B = 40e6  # default: B=40MHz
# Number of subcarriers
N_sc = 1024  # default: N_sc=1024
# Cyclic Prefix (CP) length
L_CP = 256  # default: N_sc=256
# Tx oversampling factor:
L = 5  # default: L=5

###############################################################################
# 2.2 Checks for potential errors
###############################################################################

# The maximum sampling rate of the setup is 200MHz for the transmission.
# Therefore the product of the bandwith and the oversampling factor must be
# smaller or equal to 200 MHz
assert L * B <= 200e6
# The CP length must be smaller than the number of subcarriers
assert L_CP < N_sc

###############################################################################
# 2.3 These values are directly defined from the above parameters
###############################################################################

# Subcarrier spacing [Hz]
Df = B / N_sc
# OFDM symbol duration [s]
T = 1 / Df
# Time between two samples in time domain if no oversampling factor [s]
Tc = 1 / B
# TX sampling frequency [Hz]
fs_tx = L / Tc
# Duration of the CP [s]
T_CP = L_CP * Tc
# Duration of the full OFDM symbol (with the CP included) [s]
T_PRI = T + T_CP
# Number of samples in the useful part of the OFDM symbols
pulse_width = N_sc * L
# Number of samples in the CP part of the OFDM symbols
Ng = L_CP * L

"""
#3. Parameters for the pilots and data symbols
"""
###############################################################################
# 3.1 These parameters define the data and pilot symbols and can be changed
###############################################################################

# Constellation ; either "BPSK", "QPSK", "16QAM"
const_type = "QPSK"  # Default: const_type="QPSK"
# Pilot spacing in the time domain (defines how often the OFDM symbol will
# countain pilots)
Nt = 1  # Default: Nt=1
# Pilot spacing in the frequency domain (defines the spacing in number of
# subcarriers  between two pilots subcarriers)
Nf = 1  # Default: Nf=1
# For example: P=4, N_sc = 8, Nt = 1 and Nf = 2:
# The subcarriers 0,2,4,6,7 will have pilots for all OFDM symbols 0,1,2,3. All
# the other subcarriers from the OFDM symbols are data symbols. Note that
# subcarrier 7 is included to enable interpolation and avoid extrapolation.

"""
#4. Definition of the preamble, pilots and data symbols
"""

## Loading of the constellations
if const_type == "BPSK":
    const = BPSK_const
elif const_type == "QPSK":
    const = QPSK_const
elif const_type == "16QAM":
    const = QAM16_const


## Generation of the pilots grid

# Note: the first and the last subcarrier is always included in the
# pilots. Same for the first and the last OFDM symbol. This ensure that we will
# not perform any extrapolation.
pilots_idx_f = np.concatenate((np.arange(0, N_sc - 1, Nf), [N_sc - 1]))
pilots_idx_t = np.concatenate((np.arange(0, P - 1, Nt), [P - 1]))
# Representation in two dimensions of the pilots in frequency and time domain
pilots_idx_t_mesh, pilots_idx_f_mesh = np.meshgrid(pilots_idx_t, pilots_idx_f)

## Random symbols generation for data and preamble

# Matrix of size: Number of OFDM symbols x Number of subcarriers (countains
# both the pilots and data symbols)
symbols = np.random.choice(const, (P, N_sc))
# Vector of size: Number of subcarriers (countains only the preamble symbols
# transmitted for the packet detection). In this case the preamble is not
# designed to have good autocorrelation properties
preamble = np.random.choice(const, (int(N_sc),))

## Get the corresponding pilots and bits

# Pilots symbols matrix of size: sizeof(pilots_idx_t) x sizeof(pilots_idx_f)
pilots = symbols[pilots_idx_t_mesh.T, pilots_idx_f_mesh.T]
# Corresponding bits for the data and pilot symbol of size:
# Number of OFDM symbols*Number of subcarriers
bits = inverse_mapping(symbols.reshape((P * N_sc,)), const_type)

"""
# 5. Generation of the samples of the OFDM signal to transmit
"""
## Generation of the signal
# If periodic_preamble is turned to True, the preamble should be of size N_sc/2
# as it will be periodised
tx_sig = SISO_OFDM_DFRC_TX(symbols, preamble, L_CP, L=L, periodic_preamble=False)


"""
# 6. Save, print and plot
"""
## Save everything

# Save the transmission samples in the format for the experimental setup and
# get the signal length
sig_len = generate_tx(tx_sig, output_folder + scenario_name + "_" + "tx_sig.txt")
# Save the transmitted signal the pilot and data symbols
np.save(output_folder + scenario_name + "_" + "signal.npy", tx_sig)
# Save the pilot and data symbols
np.save(output_folder + scenario_name + "_" + "symbols.npy", symbols)
# Save the preamble for the synchronization
np.save(output_folder + scenario_name + "_" + "preamble.npy", preamble)
# Save the transmission parameters
MyDic_param = {
    "P": P,
    "B": B,
    "N_sc": N_sc,
    "L_CP": L_CP,
    "L": L,
    "const_type": const_type,
    "Nt": Nt,
    "Nf": Nf,
}
# Check how to save the Dictionary
np.save(output_folder + scenario_name + "_" + "parameters.npy", MyDic_param)

## Print the sig_len for the execution in the experimental setup
print("SIG_LEN: ", sig_len)
print("2x SIG_LEN: ", 2 * sig_len)

## Plot the transmitted signal
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.real(tx_sig))
plt.subplot(2, 1, 2)
plt.plot(np.imag(tx_sig))
plt.title("TX signal")
plt.show()
