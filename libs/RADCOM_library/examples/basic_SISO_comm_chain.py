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

from numpy.random import choice

from numpy import arange,zeros,mean,sqrt,real,imag,array,Inf,sin,pi

from matplotlib.pyplot import semilogy,figure,plot,subplot,close,stem,xlabel,ylabel,title,tight_layout,show

from scipy.special import erfc

close('all')

print("--- BASIC TRANSMISSION CHAIN ---")

#### PARAMETERS
n_bits = 500 # number of bits

const_type = "QPSK" # constellation ; either "BPSK", "QPSK" or "16QAM"

filter_type = "rect" # filter type ; either "rect", "rrc" or "rc"
alpha = 1         # roll-off factor for RC and RRC filters
symbols_length = 8 # symbol spanning for RC and RRC filters

# AWGN channel: no complex attenuation, Doppler frequency and delay
paths_params = [[1,0,0]] 

fs_tx = 4e6 # TX sampling frequency
fs_rx = 4e6 # RX sampling frequency
L = 4       # TX oversampling factor
M = 4       # RX oversampling factor
# Symbol period = fs_tx/L = fs_rx/M !

#### SINGLE TRANSMISSION
print("Single transmission...")

SNR_dB = 30 # SNR [dB]
SNR = 10**(SNR_dB/10)

# computation of the delay induced by the pulse shaping and matched filter
delay = ideal_alignment(M, filter_type, symbols_length) 
# computation of the Nyquist filter energy (maximum value)
Eu = filter_energy(M, filter_type)

# computation of the noise variance based on the SNR and assuming a normalised
# constellation: SNR = sigma_I^2*Eu/var_noise
var_noise = 1/SNR * Eu

# Bits generation
bits = choice([0,1],(n_bits,)) 
# Symbol mapping
symbols = symbol_mapping(bits, const_type) 
# Pulse shaping
tx_sig = pulse_shaping(symbols, L, filter_type, alpha, symbols_length) 
# AWGN channel
rx_sig = SISO_channel(tx_sig, paths_params, var_noise, fs_tx, fs_rx)
# Matched filter
y = matched_filter(rx_sig, M, filter_type, alpha, symbols_length)   
# Downsampling and zero-forcing equalisation
symbols_hat = downsample(y, M, delay)[:len(symbols)] / Eu
# Inverse mapping
bits_hat = inverse_mapping(symbols_hat, const_type)

print("Transmitted bits (10 first): ",bits[:10])
print("Received bits (10 first): ",bits_hat[:10])
print("BER: ", mean(abs(bits-bits_hat)))

## PLOTS
# Pulse shaping and matched filters
PSF = pulse_shaping(array([1]),L,filter_type,alpha,symbols_length)[:L*symbols_length]
PSF_rec = SISO_channel(PSF, paths_params,0,fs_tx,fs_rx)
MF = matched_filter(array([1]),M,filter_type,alpha,symbols_length)
PSFxMF = matched_filter(PSF_rec,M,filter_type,alpha,symbols_length)
figure()
subplot(3,1,1)
stem(real(PSF_rec))
ylabel("Pulse shaping filter\n(resampled)")
subplot(3,1,2)
stem(real(MF))
ylabel("Matched filter")
subplot(3,1,3)
stem(real(PSFxMF))
ylabel("Nyquist filter")
show()

# Signals for a single transmission
figure()
title("Single transmission")
subplot(3,1,1)
t = arange(len(tx_sig))/fs_tx
plot(t,real(tx_sig))
ylabel("TX signal")
subplot(3,1,2)
t = arange(len(rx_sig))/fs_rx
plot(t,real(rx_sig))
ylabel("RX signal")
subplot(3,1,3)
t = arange(len(y))/fs_rx
t_s = arange(len(symbols_hat))*M/fs_rx + delay/fs_rx
plot(t,real(y))
stem(t_s,real(symbols_hat)*Eu,linefmt='r',markerfmt='.r')
ylabel("Matched filter output\n & downsampling")
xlabel("Time [s]")
tight_layout() 
show()

# Received constellation
figure()
plot(real(symbols),imag(symbols),'.b')
plot(real(symbols_hat),imag(symbols_hat),'.r')
show()

print("Done.")

#### BER COMPUTATION  
print("BER Computations...")

n_iter = 1000 # number of iterations

SNR_dB_vec = arange(0,16,1) # SNR [dB]
SNR_vec = 10**(SNR_dB_vec/10)


# computation of the delay induced by the pulse shaping and matched filter
delay = ideal_alignment(M, filter_type, symbols_length) 
# computation of the Nyquist filter energy (maximum value)
Eu = filter_energy(M, filter_type)
    
BER = zeros((len(SNR_dB_vec),))
for SNR_idx, SNR in enumerate(SNR_vec): # for every SNR...
    # computation of the noise variance based on the SNR and assuming a normalised
    # constellation: SNR = sigma_I^2*Eu/var_noise
    var_noise = 1/SNR * Eu
    for iter_idx in range(n_iter): # at each iteration...
        # Bits generation
        bits = choice([0,1],(n_bits,)) 
        # Symbol mapping
        symbols = symbol_mapping(bits, const_type) 
        # Pulse shaping
        tx_sig = pulse_shaping(symbols, L, filter_type, alpha, symbols_length) 
        # AWGN channel
        rx_sig = SISO_channel(tx_sig, paths_params, var_noise, fs_tx, fs_rx)
        # Matched filter
        y = matched_filter(rx_sig, M, filter_type, alpha, symbols_length)   
        # Downsampling and zero-forcing equalisation
        symbols_hat = downsample(y, M, delay)[:len(symbols)] / Eu
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
