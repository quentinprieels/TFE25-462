"""
Basic SISO communication chain over an AWGN channel based on the RADCOM library. 
First part, a single transmission is executed and different signals are plotted. 
In the second part, the Bit Error Rate is obtained for different SNRs.
"""

# Import channel model from channel module
from radcomlib.channel import SISO_channel

# Import communication blocks from comm_toolbox
from radcomlib.comm_toolbox import ideal_alignment
from radcomlib.comm_toolbox import filter_energy
from radcomlib.comm_toolbox import symbol_mapping
from radcomlib.comm_toolbox import pulse_shaping
from radcomlib.comm_toolbox import matched_filter
from radcomlib.comm_toolbox import downsample
from radcomlib.comm_toolbox import inverse_mapping

from radcomlib.convolutional_coding import poly2trellis
from radcomlib.convolutional_coding import conv_encoder
from radcomlib.convolutional_coding import viterbi_decoder

from numpy.random import choice

from numpy import arange,zeros,mean,sqrt,real,imag,array,Inf,sin,pi,log10

from matplotlib.pyplot import semilogy,figure,plot,subplot,close,stem,xlabel,ylabel,title,tight_layout

from scipy.special import erfc

close('all')

print("--- BASIC TRANSMISSION CHAIN ---")

#### PARAMETERS
n_bits = 500 # number of bits

const_type = "QPSK" 
mod_index = 2

filter_type = "rrc" # filter type ; either "rect", "rrc" or "rc"
alpha = 1         # roll-off factor for RC and RRC filters
symbols_length = 8 # symbol spanning for RC and RRC filters

# Convolutional code
Rc = 1/2
gn = array([1,1])
gd = array([1,1,1])

R1,R0,out_R1,out_R0 = poly2trellis(gn,gd)
symb_R1 = symbol_mapping(out_R1.reshape((out_R1.size,)),const=const_type)
symb_R0 = symbol_mapping(out_R0.reshape((out_R0.size,)),const=const_type)

# AWGN channel: no complex attenuation, Doppler frequency and delay
paths_params = [[1,0,0]] 

fs_tx = 4e6 # TX sampling frequency
fs_rx = 4e6 # RX sampling frequency
L = 4       # TX oversampling factor
M = 4       # RX oversampling factor
# Symbol period = fs_tx/L = fs_rx/M !

#### BER COMPUTATION  
print("BER Computations...")

n_iter = 1000 # number of iterations

EbN0_dB_vec = arange(0,10.2,0.2) 
EbN0_vec = 10**(EbN0_dB_vec/10)

SNR_vec = mod_index * EbN0_vec
SNR_dB_vec = 10*log10(SNR_vec)

SNR_coded_vec = mod_index * Rc * EbN0_vec
SNR_coded_dB_vec = 10*log10(SNR_coded_vec)

# computation of the delay induced by the pulse shaping and matched filter
delay = ideal_alignment(M, filter_type, symbols_length) 
# computation of the Nyquist filter energy (maximum value)
Eu = filter_energy(M, filter_type)
    
BER = zeros((len(EbN0_dB_vec),))
BER_coded = zeros((len(EbN0_dB_vec),))
for SNR_idx, (SNR,SNR_coded) in enumerate(zip(SNR_vec,SNR_coded_vec)): # for every SNR...
    # computation of the noise variance based on the SNR and assuming a normalised
    # constellation: SNR = sigma_I^2*Eu/var_noise
    var_noise = 1/SNR * Eu
    var_noise_coded = 1/SNR_coded * Eu
    
    for iter_idx in range(n_iter): # at each iteration...
        # Bits generation
        bits = choice([0,1],(n_bits,)) 
        _, bits_coded = conv_encoder(bits, R1, R0, out_R1, out_R0, len(bits))
        # Symbol mapping
        symbols = symbol_mapping(bits, const_type) 
        symbols_coded = symbol_mapping(array([bits,bits_coded]).T.reshape(int(n_bits/Rc),), const_type)
        # Pulse shaping
        tx_sig = pulse_shaping(symbols, L, filter_type, alpha, symbols_length) 
        tx_sig_coded = pulse_shaping(symbols_coded, L, filter_type, alpha, symbols_length) 
        # AWGN channel
        rx_sig = SISO_channel(tx_sig, paths_params, var_noise, fs_tx, fs_rx)
        rx_sig_coded = SISO_channel(tx_sig_coded, paths_params, var_noise_coded, fs_tx, fs_rx)
        # Matched filter
        y = matched_filter(rx_sig, M, filter_type, alpha, symbols_length)
        y_coded = matched_filter(rx_sig_coded, M, filter_type, alpha, symbols_length)
        # Downsampling and zero-forcing equalisation
        symbols_hat = downsample(y, M, delay)[:len(symbols)] / Eu
        symbols_hat_coded = downsample(y_coded, M, delay)[:len(symbols_coded)] / Eu
        # Inverse mapping
        bits_hat = inverse_mapping(symbols_hat, const_type)
        bits_hat_coded = viterbi_decoder(R1,R0,symb_R1,symb_R0,len(bits),symbols_hat_coded)
        
        # BER computation
        BER[SNR_idx] += mean(abs(bits-bits_hat))  
        BER_coded[SNR_idx] += mean(abs(bits-bits_hat_coded))          
    
    BER[SNR_idx] /= n_iter # averaging over iterations
    BER_coded[SNR_idx] /= n_iter # averaging over iterations

## PLOTS
# BER
figure()
if const_type == "BPSK":
    semilogy(EbN0_dB_vec,1/2*erfc(sqrt(SNR_vec)))
elif const_type == "QPSK":
    semilogy(EbN0_dB_vec,1/2*erfc(sqrt(SNR_vec/2)))
elif const_type == "16QAM":
    semilogy(EbN0_dB_vec,3/8*erfc(sqrt(SNR_vec/10)))
elif const_type == "16PSK":
    semilogy(EbN0_dB_vec,1/4*erfc(sqrt(SNR_vec)*sin(pi/16)))
semilogy(EbN0_dB_vec,BER,'.r')
semilogy(EbN0_dB_vec,BER_coded,'.g')
print("Done.")
    
    
    
    
    
