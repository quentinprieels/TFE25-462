"""
Basic SISO RADAR chain over an AWGN channel based on the RADCOM library. First 
part, a single transmission is executed and different signals are plotted. In 
the second part, the false alarm probability and detection probability are
obtained for different SNRs.
"""
from radcomlib.channel import SISO_channel

from radcomlib.radar_toolbox import time2pulse
from radcomlib.radar_toolbox import generate_pulse_train
from radcomlib.radar_toolbox import pulse_correlation
from radcomlib.radar_toolbox import CFAR_detector
from radcomlib.radar_toolbox import binary_map_processing
from radcomlib.radar_toolbox import targets_extraction
from radcomlib.radar_toolbox import compare_targets

from radcomlib.fmcw import generate_chirp_waveform

from numpy import ones,zeros,arange,floor,sqrt,exp,pi,angle,real,abs,sum,log
from numpy.random import choice,rayleigh,rand
from numpy.fft import fft,fftshift

from scipy.stats import ncx2

from matplotlib.pyplot import figure,plot,subplot,close,xlabel,ylabel,title,tight_layout,pcolormesh,text

%matplotlib qt 

close('all')

print("--- BASIC RADAR CHAIN ---")

#### PARAMETERS
c = 299792458 # speed of light

P = 51 # number of transmitted pulses
T = 0.5e-6 # rectangular pulse duration
T_PRI = 10e-6 # pulse repetition interval duration

B = 2/T # rectangular signal bandwidth

fs_tx = B # TX sampling frequency
fs_rx = B # RX sampling frequency

## CFAR detector parameters
P_FA = 1e-4 # targeted false alarm probability
kind = "OS" # kind of detector, "CA" or "OS"
order = 0.75 # Order of the OS-CFAR 
# CFAR geometry : |------| Nl | Gl | CUT | Gr | Nr |------|, specified for each dimension
Nl = 5 * ones((2,)) # test cells at the left of the CUT
Nr = 5 * ones((2,)) # test cells at the right of the CUT
Gl = 3 * ones((2,)) # guard cells at the left of the CUT
Gr = 3 * ones((2,)) # guard cells at the right of the CUT

## Random targets generation
N_max = 5 # maximum number of targets
f_D_max = 0.9/T_PRI # maximum Doppler frequency (in absolute value)
tau_max = T_PRI - T # maximum delay

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

# RADAR resolutions
dtau = 1/fs_rx # delay
df_D = 1/(P*T_PRI) # Doppler frequency
dx = [df_D,dtau] # resolution vector (amplitude does not matter => Inf)
centered = [1,0] # centered = 1 for parameters ranges centered around 0

SNR_dB = 30 # SNR [dB]
SNR = 10**(SNR_dB/10)

#### SINGLE TRANSMISSION
print("Single transmission...")

pulse_width = int(T*fs_tx) # pulse width
N = int(T_PRI*fs_rx) # PRI width

# Computation of the noise variance: SNR = pulse_width*P/var_noise
var_noise = 1/SNR * pulse_width*P 

# Random targets generation
targets_params = generate_targets(N_max,f_D_max,tau_max,random_amplitude=True) 

# Rectangular waveform generation
waveform = ones((pulse_width,))
# Chirp waveform generation (<!> parameters design <!>)
# waveform = generate_chirp_waveform(B,T,fs_tx)

# Creation of the pulse train
tx_sig,_ = generate_pulse_train(waveform,P,T_PRI,fs_tx) 
# SISO Multipath channel
rx_sig = SISO_channel(tx_sig, targets_params, var_noise, fs_tx, fs_rx)
# Time to pulse conversion
received_pulses = time2pulse(rx_sig,P,T_PRI,fs_rx)

# Correlation in the slow time (delay), and (centered) FFT in the fast time (Doppler)
delay_doppler_map = zeros(received_pulses.shape,dtype=received_pulses.dtype)
for i,received_pulse in enumerate(received_pulses): # Correlation per pulse
    delay_doppler_map[i,:],_ = pulse_correlation(received_pulse, waveform)
delay_doppler_map = fftshift(fft(delay_doppler_map,axis=0),axes=0) # FFT

# CFAR Detector
thresh_map, binary_map = CFAR_detector(delay_doppler_map,P_FA,Nl,Nr,Gl,Gr,kind=kind, order=order,save=True,verbose=False)

# Binary map processing => neighbour cells which are above the threshold of the 
# detector are merged around the highest value 
targets_map = binary_map_processing(delay_doppler_map, binary_map)

# Targets parameters extractions
targets_params_hat = targets_extraction(delay_doppler_map, targets_map, dx, centered)
    
# Comparison between estimation and true targets parameters to compute the number
# of detections and false alarms and their indices
detections,false_alarms = compare_targets(targets_params,targets_params_hat,dx)

# Result
print("Delay resolution: {}s".format(dtau))
print("Doppler resolution: {}Hz".format(df_D))
print("----------------------------------")
for i,target_params in enumerate(targets_params):
    if detections[i]: # detection
        print("Target {}: alpha={:.3f}<{:.3f}°, f_D={:.3e}Hz, tau={:.3e}s => D".format(i,abs(target_params[0]),angle(target_params[0])*180/pi,
                                                                                  target_params[1],target_params[2]))
    else: # misdetection
        print("Target {}: alpha={:.3f}<{:.3f}°, f_D={:.3e}Hz, tau={:.3e}s => M".format(i,abs(target_params[0]),angle(target_params[0])*180/pi,
                                                                                  target_params[1],target_params[2]))
print("----------------------------------")      
for i,target_params_hat in enumerate(targets_params_hat):
    if false_alarms[i]: # false alarm
        print("Estimated target {}: LH={:.3f}, f_D={:.3e}Hz, tau={:.3e}s => FA".format(i,abs(target_params_hat[0]),
                                                                                 target_params_hat[1],target_params_hat[2]))
    else: #  detection
        print("Estimated target {}: LH={:.3f}, f_D={:.3e}Hz, tau={:.3e}s => D".format(i,abs(target_params_hat[0]),
                                                                                 target_params_hat[1],target_params_hat[2]))
    
## PLOTS
# Axis definition for delay-Doppler map
delay_axis = arange(N) * dtau 
Doppler_axis = (arange(P) - floor(P/2)) * df_D

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

print("Done.")

### DETECTION AND FALSE ALARM PROBABILITIES COMPUTATION
compute_detection_probability = False # Set to True to evaluate a detection probability

if compute_detection_probability:
    print("Detection and false alarm probabilities Computations...")
    
    ### ADDITIONAL PARAMETERS
    n_iter = 200 # number of iterations
    
    SNR_dB_vec = arange(0,21,1) # SNR [dB]
    SNR_vec = 10**(SNR_dB_vec/10)
    ###☺
    
    pulse_width = int(T*fs_tx) # pulse width
    N = int(T_PRI*fs_rx) # PRI width
    
    P_D_MC = zeros((len(SNR_dB_vec),))
    P_FA_MC = zeros((len(SNR_dB_vec),))
    
    for SNR_idx, SNR in enumerate(SNR_vec):
        # Computation of the noise variance: SNR = pulse_width*P/var_noise
        var_noise = 1/SNR * pulse_width*P 
        n_targets = 0
        for iter_idx in range(n_iter): # at each iteration...
            # Random targets generation
            targets_params = generate_targets(N_max,f_D_max,tau_max,random_amplitude=False) 
            
            # Rectangular waveform generation
            waveform = ones((pulse_width,))
            # Creation of the pulse train
            tx_sig,_ = generate_pulse_train(waveform,P,T_PRI,fs_tx) 
            # SISO Multipath channel
            rx_sig = SISO_channel(tx_sig, targets_params, var_noise, fs_tx, fs_rx)
            # Time to pulse conversion
            received_pulses = time2pulse(rx_sig,P,T_PRI,fs_rx)
            
            # Correlation in the slow time (delay), and (centered) FFT in the fast time (Doppler)
            delay_doppler_map = zeros(received_pulses.shape,dtype=received_pulses.dtype)
            for i,received_pulse in enumerate(received_pulses): # Correlation per pulse
                delay_doppler_map[i,:],_ = pulse_correlation(received_pulse, waveform)
            delay_doppler_map = fftshift(fft(delay_doppler_map,n=P,axis=0),axes=0) # FFT
            
            # CFAR Detector
            thresh_map, binary_map = CFAR_detector(delay_doppler_map,P_FA,Nl,Nr,Gl,Gr,kind=kind,
                                                   order=order,save=True,force_manual_solver=True,verbose=False)
            
            # Binary map processing => neighbour cells which are above the threshold of the 
            # detector are merged around the highest value 
            targets_map = binary_map_processing(delay_doppler_map, binary_map)
            
            # Targets parameters extractions
            targets_params_hat = targets_extraction(delay_doppler_map, targets_map, dx, centered)
                
            # Comparison between estimation and true targets parameters to compute the number
            # of detections and false alarms and their indices
            detections,false_alarms = compare_targets(targets_params,targets_params_hat,dx)
            
            n_targets += len(targets_params)
            P_D_MC[SNR_idx] += sum(detections)
            P_FA_MC[SNR_idx] += sum(false_alarms)
        # averaging
        P_D_MC[SNR_idx] /= n_targets
        P_FA_MC[SNR_idx] /= (n_iter * N * P - n_targets)
        
    P_D_th = 1 - ncx2.cdf(-2*log(P_FA),2,2*SNR_vec)
    
    ### PLOT
    figure()
    title("Detection probability")
    plot(SNR_dB_vec,P_D_th,'k')
    plot(SNR_dB_vec,P_D_MC)
    
    print("Done.")