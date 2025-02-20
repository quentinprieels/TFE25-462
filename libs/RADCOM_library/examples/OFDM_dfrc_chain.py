"""
Basic SISO RADAR chain over an AWGN channel based on the RADCOM library. First 
part, a single transmission is executed and different signals are plotted. In 
the second part, the false alarm probability and detection probability are
obtained for different SNRs.
"""
from radcomlib.channel import SISO_channel
from radcomlib.channel import add_signals

from radcomlib.comm_toolbox import BPSK_const, QPSK_const, QAM16_const
from radcomlib.comm_toolbox import filter_energy
from radcomlib.comm_toolbox import inverse_mapping

from radcomlib.radar_toolbox import CFAR_detector
from radcomlib.radar_toolbox import binary_map_processing
from radcomlib.radar_toolbox import targets_extraction
from radcomlib.radar_toolbox import compare_targets

from radcomlib.ofdm import SISO_OFDM_DFRC_TX
from radcomlib.ofdm import SISO_OFDM_DFRC_RADAR_RX
from radcomlib.ofdm import SISO_OFDM_DFRC_COMM_RX

from numpy import arange, ones, array
from numpy import concatenate,meshgrid
from numpy import floor,sqrt,exp,real,imag,log10,abs,angle,mean
from numpy import pi,Inf
from numpy.matlib import repmat
from numpy.random import choice,rayleigh,rand
from numpy.fft import fft,ifft,ifftshift

from matplotlib.pyplot import figure,plot,subplot,close,xlabel,ylabel,title,tight_layout,pcolormesh,text,show

import matplotlib.pyplot as plt
import numpy as np

%matplotlib qt

close('all')

print("--- OFDM DFRC CHAIN ---")

#### PARAMETERS
c = 299792458 # speed of light

P = 50 # number of transmitted pulses

B = 100e6 # bandwidth
N_sc = 5000
L_CP = 25

Df = B/N_sc
T = 1/Df
Tc = 1/B

L = 4
M = 4
fs_tx = L/Tc # TX sampling frequency
fs_rx = M/Tc # RX sampling frequency

zeropad_N = 1
zeropad_P = 4

T_CP = L_CP*Tc

T_PRI = T + T_CP

## COMM synchronisation
perfect_pulse_sync = False
perfect_CSI = False

Nt = 2
Nf = 2

pilots_idx_f = concatenate((arange(0,N_sc-1,Nf),[N_sc-1]))
pilots_idx_t = concatenate((arange(0,P-1,Nt),[P-1]))

pilots_idx_t_mesh,pilots_idx_f_mesh = meshgrid(pilots_idx_t,pilots_idx_f)

const_type = "QPSK" # constellation ; either "BPSK", "QPSK" or "16QAM"
# recovery of constellation points
if const_type == "BPSK":
    const = BPSK_const
elif const_type == "QPSK":
    const = QPSK_const
else:
    const = QAM16_const


## CFAR detector parameters
P_FA = 1e-4 # targeted false alarm probability
# kind of detector, respectively "CA" or "OS" for cell-averaging or ordered 
# statistic CFAR detectors
kind = "OS" 
order = 0.75 # Order of the OS-CFAR 
# CFAR geometry : |------| Nl | Gl | CUT | Gr | Nr |------|, specified for each dimension
Nl = 5 * ones((2,)) # test cells at the left of the CUT
Nr = 5 * ones((2,)) # test cells at the right of the CUT
Gl = 3 * ones((2,)) # guard cells at the left of the CUT
Gr = 3 * ones((2,)) # guard cells at the right of the CUT

# RADAR resolutions
dtau = 1/B / M / zeropad_N # delay
df_D = 1/(P*T_PRI) / zeropad_P # Doppler frequency
dx = [df_D,dtau] # resolution vector (amplitude does not matter => Inf)
centered = [1,0] # centered = 1 for parameters ranges centered around 0

## Random targets generation
N_max = 5 # maximum number of targets
f_D_max = 1/T_PRI - df_D # maximum Doppler frequency (in absolute value)
tau_max = T_CP - dtau # maximum delay

# Target generation function
def generate_targets(N_max,f_D_max,tau_max,random_amplitude=True):
    """
    Generates a random number of targets with random parameters following the 
    specifications given in argument. The targets amplitudes are either Rayleigh 
    distributed if <random_amplitude> is True, or equal to 1 otherwise.
    """
    paths_params = []
    
    N_targets = choice(arange(N_max)+1) # random number of targets
    for target_idx in range(N_targets):
        if random_amplitude: # complex normal amplitude
            alpha = rayleigh(sqrt(1/2))*exp(-1j*rand()*2*pi) 
        else: # amplitude equal to 1
            alpha = exp(-1j*rand()*2*pi) 
        f_D = (rand()-1/2)*f_D_max # random uniformly distributed Doppler frequency
        tau = rand()*tau_max # random delay
        paths_params.append([alpha,f_D,tau])
    
    return paths_params            

# Computation of the filter energy
Eu = filter_energy(M)

# COMM SNR
SNR_dB_C = Inf # [dB]
SNR_C = 10**(SNR_dB_C/10)

# RADAR SNR 
SNR_R = SNR_C * M*N_sc*P
SNR_R_dB = 10*log10(SNR_R)

print("RADAR SNR: {}dB".format(SNR_R_dB))

#### SINGLE TRANSMISSION
print("Single transmission...")

pulse_width = N_sc*M # pulse width
Ng = L_CP*M # guard interval width

# Computation of the noise variance: SNR = N*P/var_noise
var_noise = 1/SNR_C # * np.abs(0.0003605130522449762+0.00023168767043800912j)**2

# Symbols and preambles generation for RADAR and COMM transmitters    
symbols_R = choice(const,(P,N_sc)) 
preamble_R = choice(const,(int(N_sc/2),))
symbols_C = choice(const,(P,N_sc))
preamble_C = choice(const,(int(N_sc/2),))
doppler_pilots = concatenate(([preamble_C[0]],symbols_C[:,0]))

bits = inverse_mapping(symbols_C.reshape((P*N_sc,)), const_type)

# Random RADAR targets generation
targets_params = generate_targets(N_max,f_D_max,tau_max,random_amplitude=False) 

# Comm channel parameters
#               alpha,          f_D,  tau
# path_params = [[1*exp(1j*1*pi/4), 1000, 5/fs_rx],[0.5*exp(1j*0*pi/4), -500, 6/fs_rx],[0.1*exp(1j*-1*pi/4), 2000, 2/fs_rx]]
# tau_sync = path_params[0][2]
path_params = [[1*exp(1j*1*pi/4), 1000, 2.5e-08]] # (0.0003605130522449762+0.00023168767043800912j)
tau_sync = path_params[0][2]

# OFDM DFRC signal generation
tx_sig_R = SISO_OFDM_DFRC_TX(symbols_R, preamble_R, L_CP, L)
tx_sig_C = SISO_OFDM_DFRC_TX(symbols_C, preamble_C, L_CP, L)

# SISO Multipath channel
rx_sig_R = SISO_channel(tx_sig_R, targets_params, var_noise, fs_tx, fs_rx)
rx_sig_C = SISO_channel(tx_sig_C, path_params, var_noise, fs_tx, fs_rx)



# If signals interfere with each other, one single received signal should be considered (<!> noise addition <!>)
# rx_sig_C = SISO_channel(tx_sig_C, path_params, 0, fs_tx, fs_rx)
# rx_sig = add_signals(rx_sig_R, rx_sig_C)


## COMM PROCESSING
# OFDM COMM receiver
symbols_hat,path_params_hat = SISO_OFDM_DFRC_COMM_RX(rx_sig_C, preamble_C, symbols_C[pilots_idx_t_mesh.T,pilots_idx_f_mesh.T], N_sc, L_CP, P, Nt, Nf, fs_rx, M, perfect_pulse_sync, tau_sync, perfect_CSI, path_params,zeropad_doppler=1,zeropad_delay=1)

# Inverse mapping
bits_hat = inverse_mapping(symbols_hat, const_type)

## RADAR PROCESSING
# OFDM RADAR receiver
delay_doppler_map = SISO_OFDM_DFRC_RADAR_RX(rx_sig_R, symbols_R, L_CP, M,zeropad_N,zeropad_P)

# CFAR Detector
thresh_map, binary_map = CFAR_detector(delay_doppler_map,P_FA,Nl,Nr,Gl,Gr,kind=kind,order=order,save=True,verbose=False)

# Binary map processing => neighbour cells which are above the threshold of the 
# detector are merged around the highest value 
targets_map = binary_map_processing(delay_doppler_map, binary_map)

# Targets parameters extractions
targets_params_hat = targets_extraction(delay_doppler_map, targets_map, dx, centered)


# Comparison between estimation and true targets parameters to compute the number
# of detections and false alarms and their indices
detections,false_alarms = compare_targets(targets_params,targets_params_hat,dx)



## Result
print("Delay resolution: {}s".format(dtau))
print("Doppler resolution: {}Hz".format(df_D))
print("--------------RADAR---------------")
for i,target_params in enumerate(targets_params):
    if detections[i]: # detection
        print("Target {}: alpha={:.3f}<{:.3f}°, f_D={:.3e}Hz, tau={:.3e}s => D".format(i,abs(target_params[0]),angle(target_params[0])*180/pi,
                                                                                  target_params[1],target_params[2]))
    else: # misdetection
        print("Target {}: alpha={:.3f}<{:.3f}°, f_D={:.3e}Hz, tau={:.3e}s => M".format(i,abs(target_params[0]),angle(target_params[0])*180/pi,
                                                                                  target_params[1],target_params[2]))
print("-----")      
for i,target_params_hat_corrected in enumerate(targets_params_hat):
    if false_alarms[i]: # false alarm
        print("Estimated target {}: LH={:.3f}, f_D={:.3e}Hz, tau={:.3e}s => FA".format(i,abs(target_params_hat_corrected[0]),
                                                                                 target_params_hat_corrected[1],target_params_hat_corrected[2]))
    else: #  detection
        print("Estimated target {}: LH={:.3f}, f_D={:.3e}Hz, tau={:.3e}s => D".format(i,abs(target_params_hat_corrected[0]),
                                                                                 target_params_hat_corrected[1],target_params_hat_corrected[2]))

print("--------------COMM----------------") 
print("Path parameters     : alpha={:.3f}<{:.3f}°, f_D={:.3e}Hz, tau={:.3e}s".format(abs(path_params[0][0]),angle(path_params[0][0])*180/pi,path_params[0][1],path_params[0][2]))
print("Estimated parameters: alpha={:.3f}<{:.3f}°, f_D={:.3e}Hz, tau={:.3e}s".format(abs(path_params_hat[0]),angle(path_params_hat[0])*180/pi, path_params_hat[1],path_params_hat[2]))
print("BER: ",  mean(abs(bits-bits_hat)))  

## PLOTS
# Axis definition for delay-Doppler map
delay_axis = arange(Ng*zeropad_N) * dtau
Doppler_axis = (arange(P*zeropad_P) - floor(P*zeropad_P/2)) * df_D

figure(figsize=(10, 9))
subplot(2,2,1)
pcolormesh(delay_axis,Doppler_axis,abs(delay_doppler_map),shading='nearest')
for i,(_,f_D,tau) in enumerate(targets_params):
    if detections[i] == 1:
        plot(tau,f_D,'.g')
    else:
        plot(tau,f_D,'.r')
        
for i,(_,f_D_hat,tau_hat) in enumerate(targets_params_hat):
    if false_alarms[i] == 0:
        plot(tau_hat,f_D_hat,'xg')
    else:
        plot(tau_hat,f_D_hat,'xr')
ylabel("Doppler frequency [Hz]")
title("Delay-Doppler map")
text(min(delay_axis),min(Doppler_axis),"green = D, red = M/FA\ncircle=target, cross=estimation",color="white")
subplot(2,2,2)
pcolormesh(delay_axis,Doppler_axis,thresh_map,shading='nearest')
title("Threshold map (CFAR detector)")
subplot(2,2,3)
pcolormesh(delay_axis,Doppler_axis,binary_map,shading='nearest')
xlabel("Delay [s]")
ylabel("Doppler frequency [Hz]")
title("Binary map")
subplot(2,2,4)
pcolormesh(delay_axis,Doppler_axis,targets_map,shading='nearest')
tight_layout()
xlabel("Delay [s]")
title("Target map")

# constellation
figure()
plot(real(symbols_hat), imag(symbols_hat), '.b')
plot(real(const), imag(const), '.r')
title('Received constellation')

print("Done.")