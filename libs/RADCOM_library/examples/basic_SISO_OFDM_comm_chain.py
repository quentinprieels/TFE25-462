"""
Basic SISO communication chain over an AWGN channel using OFDM based on the RADCOM library. 
First part, a single transmission is executed and different signals are plotted. 
In the second part, the Bit Error Rate is obtained for different SNRs.
"""

# Import channel model from channel module
from radcomlib.channel import SISO_channel

# Import communication blocks from comm_toolbox
from radcomlib.comm_toolbox import BPSK_const, QPSK_const, QAM16_const, PSK16_const
from radcomlib.comm_toolbox import filter_energy
from radcomlib.comm_toolbox import symbol_mapping
from radcomlib.comm_toolbox import pulse_shaping
from radcomlib.comm_toolbox import matched_filter
from radcomlib.comm_toolbox import downsample
from radcomlib.comm_toolbox import inverse_mapping

from radcomlib.ofdm import OFDM_modulation
from radcomlib.ofdm import OFDM_demodulation

from numpy.random import choice

from numpy import arange,zeros,mean,sqrt,real,imag,array,Inf,sin,pi

from matplotlib.pyplot import semilogy,figure,plot,subplot,close,stem,xlabel,ylabel,title,tight_layout,show
from scipy.special import erfc

close('all')

print("--- BASIC TRANSMISSION CHAIN ---")

#### PARAMETERS
n_bits = 2000 # number of bits

const_type = "QPSK" # constellation ; either "BPSK", "QPSK" or "16QAM"

filter_type = "rect"

# AWGN channel: no complex attenuation, Doppler frequency and delay
paths_params = [[1,0,0]] 

fs_tx = 4e6 # TX sampling frequency
fs_rx = 4e6 # RX sampling frequency
L = 4       # TX oversampling factor
M = 4       # RX oversampling factor
# Symbol period = fs_tx/L = fs_rx/M !

# OFDM
N_OFDM = 10 # number of OFDM symbols
N_sc = 128  # number of subcarriers
L_CP = 8  # length of the cyclic prefix

#### SINGLE TRANSMISSION
print("Single transmission...")

SNR_dB = 30 # SNR [dB]
SNR = 10**(SNR_dB/10)

# computation of the Nyquist filter energy (maximum value)
Eu = filter_energy(M, filter_type)

# computation of the noise variance based on the SNR and assuming a normalised
# constellation: SNR = sigma_I^2/var_noise
var_noise = 1/SNR 

# Bits generation
bits = choice([0,1],(n_bits,)) 
# Symbol mapping
symbols = symbol_mapping(bits, const_type) 
N_symbols = symbols.size
symbols.resize((N_OFDM,N_sc),refcheck=False)
# OFDM modulation
tx_sig = OFDM_modulation(symbols,L_CP,L)
# AWGN channel
rx_sig = SISO_channel(tx_sig, paths_params, var_noise, fs_tx, fs_rx)
# OFDM demodulation
y = OFDM_demodulation(rx_sig,N_OFDM,N_sc,L_CP,M) 
symbols_hat = y.flatten()[:N_symbols]
# Inverse mapping
bits_hat = inverse_mapping(symbols_hat, const_type)

print("Transmitted bits (10 first): ",bits[:10])
print("Received bits (10 first): ",bits_hat[:10])
print("BER: ", mean(abs(bits-bits_hat)))

## PLOTS
# Signals for a single transmission
figure()
title("Single transmission")
subplot(2,1,1)
t = arange(len(tx_sig))/fs_tx
plot(t,real(tx_sig))
ylabel("TX signal")
subplot(2,1,2)
t = arange(len(rx_sig))/fs_rx
plot(t,real(rx_sig))
ylabel("RX signal")
xlabel("Time [s]")
tight_layout() 
show()

# Received constellation
figure()
if const_type == "QPSK":
    plot(real(QPSK_const),imag(QPSK_const),'.b')
plot(real(symbols_hat),imag(symbols_hat),'.r')
show()

print("Done.")

#### BER COMPUTATION  
print("BER Computations...")

n_iter = 1000 # number of iterations

SNR_dB_vec = arange(0,16,1) # SNR [dB]
SNR_vec = 10**(SNR_dB_vec/10)

Eu = filter_energy(M, filter_type)
    
BER = zeros((len(SNR_dB_vec),))
for SNR_idx, SNR in enumerate(SNR_vec): # for every SNR...
    # computation of the noise variance based on the SNR and assuming a normalised
    # constellation: SNR = sigma_I^2*Eu/var_noise
    var_noise = 1/SNR
    for iter_idx in range(n_iter): # at each iteration...
        # Bits generation
        bits = choice([0,1],(n_bits,)) 
        # Symbol mapping
        symbols = symbol_mapping(bits, const_type) 
        N_symbols = symbols.size
        symbols.resize((N_OFDM,N_sc),refcheck=False)
        # OFDM modulation
        tx_sig = OFDM_modulation(symbols,L_CP,L)
        # AWGN channel
        rx_sig = SISO_channel(tx_sig, paths_params, var_noise, fs_tx, fs_rx)
        # OFDM demodulation
        y = OFDM_demodulation(rx_sig,N_OFDM,N_sc,L_CP,M) 
        symbols_hat = y.flatten()[:N_symbols]
        # Inverse mapping
        bits_hat = inverse_mapping(symbols_hat, const_type)
        
        # BER computation
        BER[SNR_idx] += mean(abs(bits-bits_hat))            
    
    BER[SNR_idx] /= n_iter # averaging over iterations

## PLOTS
# BER
figure()
if const_type == "BPSK":
    semilogy(SNR_dB_vec,1/2*erfc(sqrt(SNR_vec)))
elif const_type == "QPSK":
    semilogy(SNR_dB_vec,1/2*erfc(sqrt(SNR_vec/2)))
elif const_type == "16QAM":
    semilogy(SNR_dB_vec,3/8*erfc(sqrt(SNR_vec/10)))
elif const_type == "16PSK":
    semilogy(SNR_dB_vec,1/4*erfc(sqrt(SNR_vec)*sin(pi/16)))
semilogy(SNR_dB_vec,BER,'.r')
show()
print("Done.")    
