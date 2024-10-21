"""
The code is used to process the received OFDM waveform with the SISO (single 
input,single output) experimental setup at UCLouvain

-> How to use it:
    * Import the signal captured with the experimental setup

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
# OFDM RX Radar processing
from radcomlib.ofdm import SISO_OFDM_DFRC_RADAR_RX
# OFDM Rx Communication processing
from radcomlib.ofdm import OFDM_synchronisation,OFDM_demodulation,OFDM_channel_estimate,OFDM_channel_equalisation, OFDM_modulation
# Correlation function for the radar processing
from radcomlib.radar_toolbox import pulse_correlation

# The usual imports
#from scipy.signal import correlate as scipy_correlate
import matplotlib.pyplot as plt
import numpy as np

"""
#1. Subfunction definition
"""
def process_rx(filename):
    """
    The function opens the <.txt> file from the reception with the 
    experimental setup. The file must be moved from RamDisk to the current
    directory as defined in the jupyter notebook "section 4: Reception d'un 
    signal"
    
    Parameters
    ----------
    
    filename : string <filename.txt>
        name of the file with the I/Q samples received

    Returns
    -------
    tx_sig : numpy complex vector
        vector with the I/Q samples received

    """
    with open(filename, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32)
        data = data.astype(np.complex64)

        rx_sig = data[::2] + 1j*data[1::2]
        rx_sig.reshape(-1,1)
        rx_sig = np.squeeze(rx_sig)

    return rx_sig

"""
#2. PARAMETERS for the OFDM transmission
""" 
###############################################################################
# 2.1 These parameters were defined at the transmitter
###############################################################################

# Transmission scenario name
scenario_name = "Setup_40MHz"

# Load the parameters for the given scenario
MyDic_param = np.load(scenario_name+"_"+"parameters.npy", allow_pickle=True).item()
# Number of transmitted pulses (number of OFDM symbols) 
P = MyDic_param['P'] #default: same as Tx file
# Bandwidth [Hz]: The maximum available bandwith is 100 MHz
B = MyDic_param['B'] #default: same as Tx file
# Number of subcarriers
N_sc = MyDic_param['N_sc'] #default: same as Tx file
# Cyclic Prefix (CP) length 
L_CP = MyDic_param['L_CP'] #default: same as Tx file
# Rx oversampling factor: 
M = MyDic_param['L'] #default: same as Tx file
# Speed of light
c = 299792458 


###############################################################################
# 2.2 Checks for potential errors
###############################################################################

# The maximum sampling rate of the setup is 200MHz for the transmission. 
# Therefore the product of the bandwith and the oversampling factor must be 
# smaller or equal to 200 MHz
assert(M*B<=200e6)
# The CP length must be smaller than the number of subcarriers
assert(L_CP<N_sc)

###############################################################################
# 2.3 These values are directly defined from the above parameters 
###############################################################################

# Subcarrier spacing [Hz]
Df = B/N_sc 
# OFDM symbol duration [s]
T = 1/Df 
# Time between two samples in time domain if no oversampling factor [s] 
Tc = 1/B
# Rx sampling frequency [Hz]
fs_rx = M/Tc 
# Duration of the CP [s]
T_CP = L_CP*Tc
# Duration of the full OFDM symbol (with the CP included) [s]
T_PRI = T + T_CP
# Number of samples in the useful part of the OFDM symbols
pulse_width = N_sc*M 
# Number of samples in the CP part of the OFDM symbols
Ng = L_CP*M 

"""
#3. Parameters for the pilots and data symbols
"""
###############################################################################
# 3.1 These parameters define the data and pilot symbols. Defined by the Tx
###############################################################################

# Constellation ; either "BPSK", "QPSK", "16QAM"
const_type = MyDic_param['const_type'] # Default: same as Tx file
# Pilot spacing in the time domain (defines how often the OFDM symbol will 
# countain pilots)
Nt = MyDic_param['Nt'] # Default: same as Tx file
# Pilot spacing in the frequency domain (defines the spacing in number of 
# subcarriers  between two pilots subcarriers)
Nf = MyDic_param['Nf'] # Default: same as Tx file
# For example: P=4, N_sc = 8, Nt = 1 and Nf = 2:
# The subcarriers 0,2,4,6,7 will have pilots for all OFDM symbols 0,1,2,3. All 
# the other subcarriers from the OFDM symbols are data symbols. Note that
# subcarrier 7 is included to enable interpolation and avoid extrapolation.


"""
#4. Parameters for the Radar processing
"""

## zero-padding for radar processing
# If Zero padding is set to 1, then no zero padding is applied. If we set it to
# 2, then we add one point between each grid point
# Zero padding for the range
zeropad_N = 1 # Default: zeropad_N = 1
# Zero padding for the doppler
zeropad_P = 1 # Default: zeropad_P = 1


## RADAR grid spacing
dtau = 1/B / M / zeropad_N # delay
df_D = 1/(P*T_PRI) / zeropad_P # Doppler frequency


"""
#4. Loading of the preamble, pilots and data symbols
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
pilots_idx_f = np.concatenate((np.arange(0,N_sc-1,Nf),[N_sc-1])) 
pilots_idx_t = np.concatenate((np.arange(0,P-1,Nt),[P-1]))
# Representation in two dimensions of the pilots in frequency and time domain
pilots_idx_t_mesh,pilots_idx_f_mesh = np.meshgrid(pilots_idx_t,pilots_idx_f)

## Load the symbols generated for data and preamble 

# Matrix of size: Number of OFDM symbols x Number of subcarriers (countains 
# both the pilots and data symbols)
symbols = np.load(scenario_name+"_"+"symbols.npy")
# Vector of size: Number of subcarriers (countains only the preamble symbols
# transmitted for the packet detection). 
preamble = np.load(scenario_name+"_"+"preamble.npy")

## Get the corresponding pilots and bits

# Pilots symbols matrix of size: sizeof(pilots_idx_t) x sizeof(pilots_idx_f)
pilots = symbols[pilots_idx_t_mesh.T,pilots_idx_f_mesh.T] 
# Corresponding bits for the data and pilot symbol of size: 
# Number of OFDM symbols*Number of subcarriers
bits = inverse_mapping(symbols.reshape((P*N_sc,)), const_type)

## Load the received signal
rx_sig = process_rx("rx_sig.dat")

"""
#5. Communication processing
"""

###############################################################################
# 5.1 Synchronize the signal by correlating with the preamble 
###############################################################################

# Get the maximum of the correlation in the first half of the signal to 
# ensure that we have a full symbol afterwards
_,max_idx = pulse_correlation(rx_sig[:len(rx_sig)//2]) 
# Move the begin of the vector to the beginning of the preamble
r_sync = rx_sig[max_idx:]

###############################################################################
# 5.2 Demodulate the OFDM signal
###############################################################################

# We also demodulates the preamble hence the P+1
y_with_preamble = OFDM_demodulation(r_sync, P+1, N_sc, L_CP, M)
# Remove the first OFDM symbol since its the preamble
y = y_with_preamble[1:,:] 

###############################################################################
# 5.3 Perform the OFDM channel estimation
###############################################################################

channel_estimate_FD = OFDM_channel_estimate(y, pilots, P, N_sc, L_CP, Nt, Nf)    

###############################################################################
# 5.4 Estimate the data symbols after channel equalization
###############################################################################

symbols_hat = np.reshape(OFDM_channel_equalisation(y, channel_estimate_FD),(P*N_sc,))

###############################################################################
# 5.5 Extract the information bits from the detected symbols
###############################################################################

bits_hat = inverse_mapping(symbols_hat.reshape((P*N_sc,)), const_type) 

"""
#6. Radar processing
"""

###############################################################################
# 6.1 OFDM radar receiver (rx_sig is already synchronised at the start of the
# preamble)
###############################################################################
delay_doppler_map = SISO_OFDM_DFRC_RADAR_RX(rx_sig, symbols, L_CP, M,zeropad_N,zeropad_P)

"""
# 7. Print and plots
"""

###############################################################################
# 7.1 Plot the real and imaginary part of the received signal 
###############################################################################
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.real(rx_sig))
plt.subplot(2,1,2)
plt.plot(np.imag(rx_sig))
plt.title("RX signal")
plt.show()

###############################################################################
# 7.2 Print the Bit Error Rate (BER)
###############################################################################
# Note: Here the BER is defined with the data and pilot symbol. It should 
# normally only be computed with the data symbols.
print("BER: ",  np.mean(abs(bits-bits_hat)))  

###############################################################################
# 7.3 Plot the delay-Doppler maps
###############################################################################

# Axis definition for delay-Doppler map
delay_axis = np.arange(Ng*zeropad_N) * dtau
Doppler_axis = (np.arange(P*zeropad_P) - np.floor(P*zeropad_P/2)) * df_D

# delay-doppler map
plt.figure()
plt.pcolormesh(delay_axis,Doppler_axis,abs(delay_doppler_map),shading='nearest')
plt.ylabel("Doppler frequency [Hz]")
plt.title("Delay-Doppler map")

###############################################################################
# 7.4 Plot the received constellation before and after equalization
###############################################################################
plt.figure()
plt.plot(np.real(y), np.imag(y), '.b')
plt.title('Received constellation (before equalisation)')

plt.figure()
plt.plot(np.real(symbols_hat), np.imag(symbols_hat), '.b')
plt.plot(np.real(const), np.imag(const), '.r')
plt.title('Received constellation')

