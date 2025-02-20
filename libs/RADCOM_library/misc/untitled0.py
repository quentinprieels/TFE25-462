from radcomlib.channel import SISO_channel

from radcomlib.radar_toolbox import time2pulse
from radcomlib.radar_toolbox import generate_pulse_train
from radcomlib.radar_toolbox import pulse_correlation
from radcomlib.radar_toolbox import CFAR_detector
from radcomlib.radar_toolbox import binary_map_processing
from radcomlib.radar_toolbox import targets_extraction
from radcomlib.radar_toolbox import compare_targets

from numpy import ones,zeros,arange,floor,sqrt,exp,pi,angle,real,argwhere,abs,sum,Inf
from numpy.random import choice,rayleigh,rand,normal
from numpy.fft import fft,fftshift

from matplotlib.pyplot import figure,plot,subplot,close,xlabel,ylabel,title,tight_layout,pcolormesh,text,legend

close('all')

print("--- CFAR Validation ---")

#### PARAMETERS
c = 299792458 # speed of light

P = 50 # number of transmitted pulses
T = 0.5e-6 # rectangular pulse duration
T_PRI = 10e-6 # pulse repetition interval duration

B = 2/T # rectangular signal bandwidth

fs_tx = B # TX sampling frequency
fs_rx = B # RX sampling frequency

## CFAR detector parameters
P_FA = 1e-3 # targeted false alarm probability
kind = "OS" # kind of detector, "CA" or "OS"
order = 0.5 # Order of the OS-CFAR 
# CFAR geometry : |------| Nl | Gl | CUT | Gr | Nr |------|, specified for each dimension
Nl = 5 * ones((2,)) # test cells at the left of the CUT
Nr = 5 * ones((2,)) # test cells at the right of the CUT
Gl = 2 * ones((2,)) # guard cells at the left of the CUT
Gr = 2 * ones((2,)) # guard cells at the right of the CUT

## Random targets generation
N_max = 5 # maximum number of targets
f_D_max = 0.9/T_PRI # maximum Doppler frequency (in absolute value)
tau_max = T_PRI - T # maximum delay

# RADAR resolutions
dtau = 1/fs_rx # delay
df_D = 1/(P*T_PRI) # Doppler frequency
dx = [df_D,dtau] # resolution vector (amplitude does not matter => Inf)
centered = [1,0] # centered = 1 for parameters ranges centered around 0

SNR_dB = 30 # SNR [dB]
SNR = 10**(SNR_dB/10)


pulse_width = int(T*fs_tx) # pulse width
N = int(T_PRI*fs_rx) # PRI width

# Computation of the noise variance: SNR = pulse_width*P/var_noise
var_noise = 1/SNR * pulse_width*P 

n_iter = 1000 # number of iterations

SNR_dB = 0
SNR = 10**(SNR_dB/10)

pulse_width = int(T*fs_tx) # pulse width
N = int(T_PRI*fs_rx) # PRI width

P_FA_MC = 0

var_noise = 1/SNR
n_targets = 0
for iter_idx in range(n_iter): # at each iteration...        
    delay_doppler_map = normal(scale=sqrt(var_noise/2),size=(P,N)) + 1j*normal(scale=sqrt(var_noise/2),size=(P,N)) 
        
    # CFAR Detector
    thresh_map, binary_map = CFAR_detector(delay_doppler_map,P_FA,Nl,Nr,Gl,Gr,kind=kind,
                                           order=order,save=True,force_manual_solver=True,verbose=False)        
    
    P_FA_MC += sum(binary_map)

P_FA_MC /= n_iter * N * P
    
print("P_FA = ", P_FA_MC)

print("Done.")