"""
Basic SISO RADAR chain over an AWGN channel based on the RADCOM library. First 
part, a single transmission is executed and different signals are plotted. In 
the second part, the false alarm probability and detection probability are
obtained for different SNRs.
"""
from radcomlib.channel import SISO_channel

from radcomlib.comm_toolbox import BPSK_const, QPSK_const, PSK16_const, QAM16_const
from radcomlib.comm_toolbox import filter_energy
from radcomlib.comm_toolbox import inverse_mapping

from radcomlib.radar_toolbox import precompute_CFAR_thresholds, CFAR_detector_2D
from radcomlib.radar_toolbox import binary_map_processing
from radcomlib.radar_toolbox import targets_extraction
from radcomlib.radar_toolbox import compare_targets

from radcomlib.fmcw import SISO_PC_FMCW_DFRC_TX
from radcomlib.fmcw import SISO_PC_FMCW_DFRC_RADAR_RX
from radcomlib.fmcw import SISO_PC_FMCW_DFRC_COMM_RX
from radcomlib.fmcw import delay_Doppler_decoupling, delay_Doppler_decoupling_meshgrid

from numpy import arange, ones, zeros
from numpy import concatenate,meshgrid
from numpy import floor,sqrt,exp,real,imag,log10,abs,angle,mean,log,sin
from numpy import pi,Inf
from numpy.matlib import repmat
from numpy.random import choice,rayleigh,rand

from matplotlib.pyplot import figure,plot,subplot,close,xlabel,ylabel,title,tight_layout,pcolormesh,text,semilogy

from scipy.special import erfc
from scipy.stats import ncx2

%matplotlib qt

close('all')

print("--- FMCW DFRC CHAIN ---")

#### PARAMETERS
c = 299792458 # speed of light

P = 50 # number of transmitted pulses
T = 99e-6 # chirp duration
T_PRI = 100e-6 # pulse repetition interval duration
Tg = T_PRI - T # guard interval

Lc = 100 # number of symbols per pulse
Tc = T/Lc # symbol duration

## COMM synchronisation
# Carrier frequency estimation method, either "fft" or "moose" respectively for
# FFT-based method or Moose algorithm
CFO_estimation_method = "fft"
perfect_alpha_estimation = False # perfect estimation of complex attenuation if True
perfect_CFO_estimation = False # perfect estimation of CFO if True
perfect_pulse_detection = False # perfect pulse detection if True
perfect_sync_vec = [perfect_alpha_estimation,perfect_CFO_estimation,perfect_pulse_detection]

const_type = "QPSK" # constellation ; either "BPSK", "QPSK" or "16QAM"
# recovery of constellation points
if const_type == "BPSK":
    const = BPSK_const
    mod_index = 1
elif const_type == "QPSK":
    const = QPSK_const
    mod_index = 2
elif const_type == "16PSK":
    const = PSK16_const
    mod_index = 4
else:
    const = QAM16_const
    mod_index = 4

B = 20e6 # chirp bandwidth
up = True # Up-chirp if True, otherwise down-chirp
centered = True # Centered around 0Hz if True

L = int(2*B*T/Lc) # upsampling factor at TX
M = L # downsampling factor at RX
fs_tx = L/Tc # TX sampling frequency
fs_rx = M/Tc # RX sampling frequency

## CFAR detector parameters
P_FA = 1e-4 # targeted false alarm probability
# kind of detector, respectively "CA" or "OS" for cell-averaging or ordered 
# statistic CFAR detectors
kind = "OS" 
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

# Computation of the filter energy
Eu = filter_energy(M)

# COMM SNR
SNR_dB_C = Inf # [dB]
SNR_C = 10**(SNR_dB_C/10)

# RADAR SNR 
SNR_R = SNR_C/Eu * M*Lc*P
SNR_R_dB = 10*log10(SNR_R)

print("RADAR SNR: {}dB".format(SNR_R_dB))

#### SINGLE TRANSMISSION
print("Single transmission...")

pulse_width = int(T*fs_rx) # pulse width
Ng = int(Tg*B) # guard interval width (after downsampling)

# Computation of the noise variance: SNR = N*P/var_noise
var_noise = 1/SNR_C * Eu

# Symbols and preambles generation for RADAR and COMM transmitters    
symbols_R = choice(const,(P,Lc)) 
preamble_R = repmat(choice(const,(int(Lc/2),)),1,2)[0,:]
symbols_C = choice(const,(P,Lc))
preamble_C = repmat(choice(const,(int(Lc/2),)),1,2)[0,:]
doppler_pilots = concatenate(([preamble_C[0]],symbols_C[:,0]))

bits = inverse_mapping(symbols_C.reshape((P*Lc,)), const_type)

# Random RADAR targets generation
targets_params = generate_targets(N_targets_max,f_D_max,tau_max,random_amplitude=False) 
# Flat fading channel parameters
#               alpha,          f_D,  tau
path_params = [[1*exp(1j*pi/4), 1000, 10.1/fs_rx]]

# PC-FMCW DFRC signal generation
tx_sig_R = SISO_PC_FMCW_DFRC_TX(symbols_R, preamble_R, L, B, T, T_PRI, up, centered)
tx_sig_C = SISO_PC_FMCW_DFRC_TX(symbols_C, preamble_C, L, B, T, T_PRI, up, centered)

# SISO Multipath channel
rx_sig_R = SISO_channel(tx_sig_R, targets_params, var_noise, fs_tx, fs_rx)
rx_sig_C = SISO_channel(tx_sig_C, path_params, var_noise, fs_tx, fs_rx)

# If signals interfere with each other, one single received signal should be considered (<!> noise addition <!>)
# rx_sig_C = SISO_channel(tx_sig_C, path_params, 0, fs_tx, fs_rx)
# rx_sig = add_signals(rx_sig_R, rx_sig_C)

## RADAR PROCESSING
# FMCW RADAR receiver
delay_doppler_map = SISO_PC_FMCW_DFRC_RADAR_RX(rx_sig_R, symbols_R, preamble_R, M, B, T, T_PRI, up, centered)

# CFAR Detector
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

## COMM PROCESSING
# FMCW COMM receiver
symbols_hat, path_params_hat = SISO_PC_FMCW_DFRC_COMM_RX(rx_sig_C, preamble_C,M,Lc,B,T,P,T_PRI,up,centered,CFO_estimation_method,doppler_pilots,perfect_sync_vec,path_params[0])
            
# Inverse mapping
bits_hat = inverse_mapping(symbols_hat, const_type)

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
for i,target_params_hat_corrected in enumerate(targets_params_hat_corrected):
    if false_alarms[i]: # false alarm
        print("Estimated target {}: LH={:.3f}, f_D={:.3e}Hz, tau={:.3e}s => FA".format(i,abs(target_params_hat_corrected[0]),
                                                                                 target_params_hat_corrected[1],target_params_hat_corrected[2]))
    else: #  detection
        print("Estimated target {}: LH={:.3f}, f_D={:.3e}Hz, tau={:.3e}s => D".format(i,abs(target_params_hat_corrected[0]),
                                                                                 target_params_hat_corrected[1],target_params_hat_corrected[2]))

print("--------------COMM----------------") 
print("Path parameters     : alpha={:.3f}<{:.3f}°, f_D={:.3e}Hz, tau={:.3e}s".format(abs(path_params[0][0]),angle(path_params[0][0])*180/pi,
                                                                                  path_params[0][1],path_params[0][2]))
print("Estimated parameters: alpha={:.3f}<{:.3f}°, f_D={:.3e}Hz, tau={:.3e}s".format(abs(path_params_hat[0]),angle(path_params_hat[0])*180/pi,
                                                                                  path_params_hat[1],path_params_hat[2]))
print("BER: ",  mean(abs(bits-bits_hat)))  

## PLOTS
# Axis definition for delay-Doppler map
delay_axis = arange(Ng) * dtau 
Doppler_axis = (arange(P) - floor(P/2)) * df_D

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

# constellation
figure()
plot(real(symbols_hat), imag(symbols_hat), '.b')
plot(real(const), imag(const), '.r')
title('Received constellation')

print("Done.")

compute_metrics = False

if compute_metrics:
    
    n_iter = 100
    
    EbN0_dB_vec = arange(-30,-9,2)
    EbN0_vec = 10**(EbN0_dB_vec/10)
    
    SNR_C_vec = mod_index * EbN0_vec
    SNR_C_dB_vec = 10*log10(SNR_C_vec)
    
    SNR_R_vec = pulse_width*P/Eu * mod_index * EbN0_vec
    SNR_R_dB_vec = 10*log10(SNR_R_vec)
    
    BER_MC = zeros((len(EbN0_dB_vec,)))
    P_D_MC = zeros((len(EbN0_dB_vec,)))
    P_FA_MC = zeros((len(EbN0_dB_vec,))) 
    
    for SNR_idx, (SNR_R,SNR_C) in enumerate(zip(SNR_R_vec,SNR_C_vec)):

        var_noise = Eu / SNR_C
        
        targets_params = [[1,0,10*dtau]]

        n_targets = 0
        n_bits = 0
        for iter_idx in range(n_iter):
    
            symbols_R = ones((P,Lc))#symbols_R = choice(const,(P,Lc)) 
            preamble_R = ones((Lc,))#repmat(choice(const,(int(Lc/2),)),1,2)[0,:]
            symbols_C = choice(const,(P,Lc))
            preamble_C = repmat(choice(const,(int(Lc/2),)),1,2)[0,:]
            doppler_pilots = concatenate(([preamble_C[0]],symbols_C[:,0]))
    
            bits = inverse_mapping(symbols_C.reshape((P*Lc,)), const_type)
    
            # PC-FMCW DFRC signal generation
            tx_sig_R = SISO_PC_FMCW_DFRC_TX(symbols_R, preamble_R, L, B, T, T_PRI, up, centered)
            tx_sig_C = SISO_PC_FMCW_DFRC_TX(symbols_C, preamble_C, L, B, T, T_PRI, up, centered)
    
            # SISO Multipath channel
            rx_sig_R = SISO_channel(tx_sig_R, targets_params, var_noise, fs_tx, fs_rx)
            rx_sig_C = SISO_channel(tx_sig_C, path_params, var_noise, fs_tx, fs_rx)
    
            ## RADAR PROCESSING
            # FMCW RADAR receiver
            delay_doppler_map = SISO_PC_FMCW_DFRC_RADAR_RX(rx_sig_R, symbols_R, preamble_R, M, B, T, T_PRI, up, centered)
    
            # CFAR Detector
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
            
            ## COMM PROCESSING
            # FMCW COMM receiver
            symbols_hat, path_params_hat = SISO_PC_FMCW_DFRC_COMM_RX(rx_sig_C, preamble_C,M,Lc,B,T,P,T_PRI,up,centered,CFO_estimation_method,doppler_pilots,perfect_sync_vec,path_params[0])
                        
            # Inverse mapping
            bits_hat = inverse_mapping(symbols_hat, const_type)
        
            n_targets += len(targets_params)
            n_bits += len(bits)
            P_D_MC[SNR_idx] += sum(detections)
            P_FA_MC[SNR_idx] += sum(false_alarms)
            BER_MC[SNR_idx] += sum(abs(bits-bits_hat)/2)
        # averaging
        P_D_MC[SNR_idx] /= n_targets
        P_FA_MC[SNR_idx] /= (n_iter * Ng * P - n_targets)
        BER_MC[SNR_idx] /= n_bits
        
    P_D_th = 1 - ncx2.cdf(-2*log(P_FA),2,2*SNR_R_vec)
    if const_type == "BPSK":
        BER_th = 1/2*erfc(sqrt(SNR_C_vec))
    elif const_type == "QPSK":
        BER_th = 1/2*erfc(sqrt(SNR_C_vec/2))
    elif const_type == "16QAM":
        BER_th = 3/8*erfc(sqrt(SNR_C_vec/10))
    elif const_type == "16PSK":
        BER_th = 1/4*erfc(sqrt(SNR_C_vec)*sin(pi/16))
        
    figure()
    title("Bit Error Rate")
    semilogy(EbN0_dB_vec, BER_th, 'k')
    semilogy(EbN0_dB_vec, BER_MC)
    
    figure()
    title("Detection probability")
    plot(EbN0_dB_vec, P_D_th, 'k')
    plot(EbN0_dB_vec, P_D_MC)
    
    figure()
    title("False alarm probability")
    semilogy(EbN0_dB_vec,P_FA_MC)