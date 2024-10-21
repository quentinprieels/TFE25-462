"""
Basic SISO RADAR chain over an AWGN channel based on the RADCOM library. First 
part, a single transmission is executed and different signals are plotted. In 
the second part, the false alarm probability and detection probability are
obtained for different SNRs.
"""
from radcomlib.channel import SISO_channel

from radcomlib.radar_toolbox import precompute_CFAR_thresholds
from radcomlib.radar_toolbox import CFAR_detector, CFAR_detector_2D
from radcomlib.radar_toolbox import binary_map_processing
from radcomlib.radar_toolbox import targets_extraction
from radcomlib.radar_toolbox import compare_targets

from radcomlib.fmcw import SISO_FMCW_RADAR_TX
from radcomlib.fmcw import SISO_FMCW_RADAR_RX
from radcomlib.fmcw import delay_Doppler_decoupling, delay_Doppler_decoupling_meshgrid

from numpy import ones,zeros,arange,floor,sqrt,exp,pi,angle,real,abs,sum,log,meshgrid,mean
from numpy.random import choice,rayleigh,rand

from scipy.stats import ncx2

from matplotlib.pyplot import figure,plot,subplot,close,xlabel,ylabel,title,tight_layout,pcolormesh,text

%matplotlib qt

close('all')

print("--- FMCW RADAR CHAIN ---")

#### PARAMETERS
c = 299792458 # speed of light

P = 50 # number of transmitted pulses
T = 9e-6 # chirp duration
T_PRI = 10e-6 # pulse repetition interval duration
Tg = T_PRI - T # guard interval

B = 20e6 # chirp bandwidth
up = True # Up-chirp if True, otherwise down-chirp
centered = True # Centered around 0Hz if True
fs_tx = B # TX sampling frequency
fs_rx = B # RX sampling frequency

## CFAR detector parameters
P_FA = 1e-4 # targeted false alarm probability
kind = "OS" # kind of detector, "CA" or "OS"
order = 0.75 # Order of the OS-CFAR 
# CFAR geometry : |------| Nl | Gl | CUT | Gr | Nr |------|, specified for each dimension
Nl = 5 * ones((2,),dtype=int) # test cells at the left of the CUT
Nr = 5 * ones((2,),dtype=int) # test cells at the right of the CUT
Gl = 3 * ones((2,),dtype=int) # guard cells at the left of the CUT
Gr = 3 * ones((2,),dtype=int) # guard cells at the right of the CUT
N_max = (Nl[0]+Nr[0])*(Nl[1]+Gl[1]+1+Gr[1]+Nr[1]) + (Nl[1]+Nr[1])*(Gl[0]+1+Gr[0]) # upper bound for the number of cells



# RADAR resolutions
dtau = 1/B # delay
df_D = 1/(P*T_PRI) # Doppler frequency
dx = [df_D,dtau] # resolution vector (amplitude does not matter => Inf)
centered = [1,0] # centered = 1 for parameters ranges centered around 0

## Random targets generation
N_targets_max = 5 # maximum number of targets
f_D_max = 1/T_PRI - df_D # maximum Doppler frequency (in absolute value)
tau_max = Tg - dtau # maximum delay

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

SNR_dB = 30 # SNR [dB]
SNR = 10**(SNR_dB/10)

#### SINGLE TRANSMISSION
print("Single transmission...")

pulse_width = int(T*fs_rx) # pulse width
Ng = int(Tg*fs_rx) # guard interval width

# Computation of the noise variance: SNR = N*P/var_noise
var_noise = 1/SNR * pulse_width*P 

# Random targets generation
# targets_params = [[1,0,11/fs_rx]] # single target centered
targets_params = generate_targets(N_targets_max,f_D_max,tau_max,random_amplitude=True) 

# FMCW RADAR signal generation
tx_sig = SISO_FMCW_RADAR_TX(B, T, P, T_PRI, fs_tx, up, centered)
# SISO Multipath channel
rx_sig = SISO_channel(tx_sig, targets_params, var_noise, fs_tx, fs_rx)
# FMCW RADAR receiver
delay_doppler_map = SISO_FMCW_RADAR_RX(rx_sig, B, T, P, T_PRI, fs_rx, up, centered)
                                    
# CFAR Detector
# thresh_map, binary_map = CFAR_detector(delay_doppler_map,P_FA,Nl,Nr,Gl,Gr,kind=kind,order=order,save=True,verbose=False)
CFAR_coefs_array = precompute_CFAR_thresholds(P_FA,arange(N_max)+1,kind=kind,order=order) # precomputing potential CFAR thresholds values
thresh_map, binary_map = CFAR_detector_2D(delay_doppler_map,Nl,Nr,Gl,Gr,CFAR_coefs_array,kind=kind,order=order)

# Binary map processing => neighbour cells which are above the threshold of the 
# detector are merged around the highest value 
targets_map = binary_map_processing(delay_doppler_map, binary_map)

# Targets parameters extractions
targets_params_hat = targets_extraction(delay_doppler_map, targets_map, dx, centered)

# Delay Doppler decoupling
targets_params_hat_corrected = delay_Doppler_decoupling(targets_params_hat,B,T,up)

# Comparison between estimation and true targets parameters to compute the number
# of detections and false alarms and their indices
detections,false_alarms = compare_targets(targets_params,targets_params_hat_corrected,dx)

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
for i,target_params_hat_corrected in enumerate(targets_params_hat_corrected):
    if false_alarms[i]: # false alarm
        print("Estimated target {}: LH={:.3f}, f_D={:.3e}Hz, tau={:.3e}s => FA".format(i,abs(target_params_hat_corrected[0]),
                                                                                 target_params_hat_corrected[1],target_params_hat_corrected[2]))
    else: #  detection
        print("Estimated target {}: LH={:.3f}, f_D={:.3e}Hz, tau={:.3e}s => D".format(i,abs(target_params_hat_corrected[0]),
                                                                                 target_params_hat_corrected[1],target_params_hat_corrected[2]))
    
## PLOTS
# Axis definition for delay-Doppler map
delay_axis = arange(Ng) * dtau 
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

doppler_mesh,delay_mesh = meshgrid(Doppler_axis,delay_axis,indexing='ij')
delay_mesh_corrected = delay_Doppler_decoupling_meshgrid(delay_mesh,doppler_mesh,B,T,up)

figure(figsize=(10, 9))
subplot(2,2,1)
pcolormesh(delay_mesh_corrected,doppler_mesh,abs(delay_doppler_map),shading='nearest')
for i,(_,f_D,tau) in enumerate(targets_params):
    if detections[i] == 1:
        plot(tau,f_D,'.g')
    else:
        plot(tau,f_D,'.r')
        
for i,(_,f_D_hat,tau_hat) in enumerate(targets_params_hat_corrected):
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

#### DETECTION AND FALSE ALARM PROBABILITIES COMPUTATION
compute_detection_probability = True # Set to True to evaluate a detection probability

if compute_detection_probability: 
    print("Detection and false alarm probabilities computations...")
    
    ### ADDITIONAL PARAMETERS
    n_iter = 100 # number of iterations
    
    SNR_dB_vec = arange(0,31,1) # SNR [dB]
    SNR_vec = 10**(SNR_dB_vec/10)
    ### 
    
    pulse_width = int(T*fs_rx) # pulse width
    Ng = int((T_PRI - T)*fs_rx) # guard interval width
    
    P_D_MC = zeros((len(SNR_dB_vec),))
    P_FA_MC = zeros((len(SNR_dB_vec),))
    
    CFAR_coefs_array = precompute_CFAR_thresholds(P_FA,arange(N_max)+1,kind=kind,order=order) # precomputing potential CFAR thresholds values   
    
    for SNR_idx, SNR in enumerate(SNR_vec):
        # Computation of the noise variance: SNR = pulse_width*P/var_noise
        var_noise = 1/SNR * pulse_width*P 
        n_targets = 0
        for iter_idx in range(n_iter): # at each iteration...
            # Random targets generation
            targets_params = [[1,0,11/fs_rx]] # single target centered
            # targets_params = generate_targets(N_max,f_D_max,tau_max,random_amplitude=False) 
            
            # FMCW RADAR signal generation
            tx_sig = SISO_FMCW_RADAR_TX(B, T, P, T_PRI, fs_tx)
            # SISO Multipath channel
            rx_sig = SISO_channel(tx_sig, targets_params, var_noise, fs_tx, fs_rx)
            # FMCW RADAR receiver
            delay_doppler_map = SISO_FMCW_RADAR_RX(rx_sig, B, T, P, T_PRI, fs_rx)    
            
            # CFAR Detector
            thresh_map, binary_map = CFAR_detector_2D(delay_doppler_map,Nl,Nr,Gl,Gr,CFAR_coefs_array,kind=kind,order=order)

            # Binary map processing => neighbour cells which are above the threshold of the 
            # detector are merged around the highest value 
            targets_map = binary_map_processing(delay_doppler_map, binary_map)
            
            # Targets parameters extractions
            targets_params_hat = targets_extraction(delay_doppler_map, targets_map, dx, centered)
            
            # Delay-Doppler decoupling
            targets_params_hat_corrected = delay_Doppler_decoupling(targets_params_hat,B,T,up)
            
            # Comparison between estimation and true targets parameters to compute the number
            # of detections and false alarms and their indices
            detections,false_alarms = compare_targets(targets_params,targets_params_hat_corrected,dx)
            
            n_targets += len(targets_params)
            P_D_MC[SNR_idx] += sum(detections)
            P_FA_MC[SNR_idx] += sum(false_alarms)
        # averaging
        P_D_MC[SNR_idx] /= n_targets
        P_FA_MC[SNR_idx] /= (n_iter * Ng * P - n_targets)
        
    P_D_th = 1 - ncx2.cdf(-2*log(P_FA),2,2*SNR_vec)
    
    ### PLOT
    figure()
    title("Detection probability")
    plot(SNR_dB_vec,P_D_th,'k')
    plot(SNR_dB_vec,P_D_MC)
    
    print("Mean achieved false alarm probability: {:.2e} (target: {:.2e})".format(mean(P_FA_MC),P_FA))
    print("Done.")