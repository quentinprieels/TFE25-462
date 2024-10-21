"""
Blocks needed to setup a Phase-Coded FMCW DFRC system.
"""

__all__ = ["chirp_modulation", \
           "chirp_demodulation", \
           "generate_chirp_waveform", \
           "LPF", \
           "GDF", \
           "delay_Doppler_decoupling", \
           "delay_Doppler_decoupling_meshgrid", \
           "SISO_FMCW_RADAR_TX", \
           "SISO_FMCW_RADAR_RX", \
           "SISO_PC_FMCW_DFRC_TX", \
           "SISO_PC_FMCW_DFRC_RADAR_RX", \
           "SISO_PC_FMCW_DFRC_COMM_RX"]

### Imports
from numpy import arange,zeros,ones
from numpy import argmax
from numpy import reshape,concatenate
from numpy import exp,abs,sum,round,ceil,conj,angle
from numpy import pi, complex64

from numpy.fft import fft,fft2,fftshift,ifft,ifftshift

from .radar_toolbox import time2pulse, pulse2time
from .radar_toolbox import generate_pulse_train
from .radar_toolbox import pulse_correlation

from .comm_toolbox import filter_energy
from .comm_toolbox import pulse_shaping, matched_filter
from .comm_toolbox import downsample
from .comm_toolbox import ideal_alignment
from .comm_toolbox import timing_synchronisation
from .comm_toolbox import estimate_frequency_offset
from .comm_toolbox import correct_frequency_offset

### Exceptions
class InputError(Exception):
    pass

### Functions
def chirp_modulation(x,B,T,fs,up=True,centered=True):
    """
    Modulates input signal x with a chirp of duration T. The chirp is increasing
    or decreasing in frequency depending on <up>.
    The chirp rate is equal to <B>/<T>, meaning that the signal is modulated in
    baseband around a carrier frequency
    | -<B>/2 + <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |          <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for an increasing chirp, and
    |  <B>/2 - <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |        - <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for a decreasing chirp.
    If len(x)/<fs> > <T>, additional elements are set to zero.
    
    Parameters
    ----------
    x : Numpy 1D array
        Input signal.
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    fs : Float
        Input sample rate.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
    centered : Boolean, optional
        If set to <True>, the total bandwidth ranged by the chirp is centered around 0. 
        The default is True.

    Returns
    -------
    Numpy complex 1D array
        Output signal.
    """
    
    n = arange(0,len(x))

    if centered:
        chirp = exp(-1j*pi*B*n/fs) * exp(1j*pi*B/T*(n/fs)**2)
    else:
        chirp =                      exp(1j*pi*B/T*(n/fs)**2)
        
    out = x * chirp if up else x * conj(chirp)
    out[int(fs*T):] = 0
    return out

def chirp_demodulation(x,B,T,fs,up=True,centered=True):
    """
    Demodulates input signal x with a chirp of duration T. The chirp is increasing
    or decreasing in frequency depending on <up>.
    The chirp rate is equal to <B>/<T>, meaning that the signal to demodulate is 
    located in baseband around a carrier frequency
    | -<B>/2 + <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |          <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for an increasing chirp, and
    |  <B>/2 - <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |        - <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for a decreasing chirp.
    If len(x)/<fs> > <T>, additional elements are set to zero.

    Parameters
    ----------
    x : Numpy 1D array
        Input signal.
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    fs : Float
        Input sample rate.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
    centered : Boolean, optional
        If set to <True>, the total bandwidth ranged by the chirp is centered around 0. 
        The default is True.

    Returns
    -------
    Numpy complex 1D array
        Output signal.
    """
    
    n = arange(0,len(x))
    if up:
        chirp = exp(1j*pi*B*n/fs) * exp(-1j*pi*B/T*(n/fs)**2)
    else:
        chirp = exp(-1j*pi*B*n/fs) * exp(1j*pi*B/T*(n/fs)**2)
    out = x * chirp
    out[int(fs*T):] = 0
    return out

def generate_chirp_waveform(B,T,fs,up=True,centered=True):
    """
    Generates a chirp waveform of duration <T> over a bandwidth <B>. The chirp 
    is increasing or decreasing in frequency depending on <up>. The chirp rate 
    is equal to <B>/<T>, meaning that the signal is modulated in baseband 
    around a carrier frequency
    | -<B>/2 + <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |          <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for an increasing chirp, and
    |  <B>/2 - <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |        - <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for a decreasing chirp.

    Parameters
    ----------
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    fs : Float
        Sampling frequency.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
    centered : Boolean, optional
        If set to <True>, the total bandwidth ranged by the chirp is centered around 0. 
        The default is True.

    Returns
    -------
    waveform : Numpy 1D array
        Chirp waveform.
    """
    
    pulse_width = int(round(T*fs))
    out = ones((pulse_width,))
    waveform = chirp_modulation(out,B,T,fs,up=up,centered=centered)
    return waveform 

def LPF(x,fc,fs):
    """
    Transfers the input signal <x> through a low-pass filter with cut-off frequency
    <fc>. The implemented filter is a rectangular filter in frequency, which deletes 
    every frequency component higher or equal than <fc> in the frequency domain.

    Parameters
    ----------
    x : Numpy 1D array
        Input signal.
    fc : Float
        Cut-off frequency.
    fs : Float
        Sampling frequency.

    Returns
    -------
    Numpy 1D array
        Filtered signal.
    """
    
    fft_x = fftshift(fft(x))
    f = arange(len(x))/len(x)*fs - ceil(fs/2)
    fft_x[abs(f) >= fc] = 0
    return ifft(ifftshift(fft_x))
    
def GDF(x,B,T,fs):
    """
    Transfers the input signal <x> through a group-delay filter, with group delay
    tau_gd(f) = <T>/<B>*f and phase delay tau_ph(f) = 1/2*<T>/<B>*f.
    
    Parameters
    ----------
    x : Numpy 1D array
        Input signal.
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    fs : Float
        Sampling frequency.

    Returns
    -------
    Numpy 1D array
        Filtered signal.
    """
    
    fft_x = fft(x)
    f = arange(len(x))/len(x)*fs
    phi = -pi*T/B*f**2
    return ifft(fft_x * exp(1j*phi))

def delay_Doppler_decoupling(targets_params,B,T,up=True):
    """
    Modifies the estimated targets parameters <targets_params> to decouple the 
    impact of the Doppler frequencies on the estimated delays.

    Parameters
    ----------
    targets_params : List of list
        Estimated targets parameters. For example, with a SISO channel and two
        targets, the list is
        
                [[alpha_0,fD_0,tau_0], [alpha_1,fD_1,tau_1]]
                
        with <fD_i> and <tau_i> are respectively the Doppler frequencies and 
        delays of the two paths. 
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
        
    Returns
    -------
    List of list
    Corrected estimated targets parameters, of the same format as <targets_params>.
    """
    
    targets_params_corrected = targets_params.copy()
    for i in range(len(targets_params)):
        if up:
            targets_params_corrected[i][2] = targets_params[i][2] + T/B*targets_params[i][1]
        else:
            targets_params_corrected[i][2] = targets_params[i][2] - T/B*targets_params[i][1]
    
    return targets_params_corrected

def delay_Doppler_decoupling_meshgrid(tau_mesh,fD_mesh,B,T,up=True):
    """
    Modifies the meshgrid of delays <tau_mesh> to decouple the impact of the 
    Doppler frequencies on the delays.
    
    Parameters
    ----------
    tau_mesh : Numpy 2D array
        Meshgrid of delays.
    fD_mesh : Numpy 2D array
        Meshgrid of Doppler frequencies.
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
        
    Returns
    -------
    Numpy 2D array
    Corrected meshgrid of delays.
    """
    
    if up:
        tau_mesh_corrected = tau_mesh + T/B*fD_mesh
    else:
        tau_mesh_corrected = tau_mesh - T/B*fD_mesh

    return tau_mesh_corrected

def SISO_FMCW_RADAR_TX(B,T,P,T_PRI,fs,up=True,centered=True):
    """
    SISO FMCW RADAR Transmitter.

    Generates the transmitted signal, a train of <P> chirps of duration <T> and
    bandwidth <B> with Pulse Repetition Interval (PRI) duration <T_PRI>.
    
    The chirps are increasing or decreasing in frequency depending on <up>. The 
    chirp rate is equal to <B>/<T>, meaning that each chirp is modulated in
    baseband around a carrier frequency
    | -<B>/2 + <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |          <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for an increasing chirp, and
    |  <B>/2 - <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |        - <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for a decreasing chirp.    

    Parameters
    ----------
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    P : Integer
        Number of transmitted pulses.
    T_PRI : Float
        PRI duration.
    fs : Float
        Sampling frequency.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
    centered : Boolean, optional
        If set to <True>, the total bandwidth ranged by the chirp is centered around 0. 
        The default is True.

    Returns
    -------
        Numpy 1D array
        TX signal.
    """
    
    waveform = generate_chirp_waveform(B,T,fs,up,centered)
    tx_sig,_ = generate_pulse_train(waveform,P,T_PRI,fs)    
    return tx_sig

def SISO_FMCW_RADAR_RX(rx_sig,B,T,P,T_PRI,fs,up=True,centered=True):
    """
    SISO FMCW RADAR Receiver.

    Process the received SISO FMCW RADAR signal <rx_sig> to generate a 
    delay-Doppler map.

    Parameters
    ----------
    rx_sig : Numpy 1D array
        Input signal.
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    P : Integer
        Number of transmitted pulses.
    T_PRI : Float
        PRI duration.
    fs : Float
        Sampling frequency.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
    centered : Boolean, optional
        If set to <True>, the total bandwidth ranged by the chirp is centered around 0. 
        The default is True.

    Returns
    -------
        Numpy 2D array
        Output delay-Doppler map.
    """
    
    Tg = T_PRI - T
    Ng = int(Tg*fs)
    Nt = int(T*fs)
    
    f_cut = B/T * Tg
    
    received_pulses = time2pulse(rx_sig,P,T_PRI,fs)[:,:Nt]
    
    processed_pulses = zeros(received_pulses.shape,dtype=received_pulses.dtype)
    for i,pulse in enumerate(received_pulses):
        dechirped_pulse = chirp_demodulation(pulse,B,T,fs,up,centered)
        processed_pulses[i,:] = LPF(dechirped_pulse,f_cut,fs)    
    
    delay_doppler_map = fftshift(fft2(processed_pulses),axes=0)
    out = delay_doppler_map[:,arange(0,-Ng,-1)] if up else delay_doppler_map[:,arange(Ng)]
    return out
            
def SISO_PC_FMCW_DFRC_TX(symbols,preamble,L,B,T,T_PRI,up=True,centered=True):
    """
    SISO PC-FMCW DFRC Transmitter.
    
    Generates the transmitted signal. The signal is composed of a train of chirps
    of duration <T> and bandwidth <B>, with Pulse Repetition Interval (PRI) 
    duration <T_PRI>. Each chirp is modulated by phase-coded symbols with rectangular 
    pulse-shaping.
    
    The chirps are increasing or decreasing in frequency depending on <up>. The 
    chirp rate is equal to <B>/<T>, meaning that each chirp is modulated in
    baseband around a carrier frequency
    | -<B>/2 + <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |          <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for an increasing chirp, and
    |  <B>/2 - <B>/<T>*t for t in [0,T] if <centered> is set to True;
    |        - <B>/<T>*t for t in [0,T] if <centered> is set to False,
    for a decreasing chirp.    

    The number of pulses <P> (preamble excluded) and the number of symbols per 
    pulse <Lc> are given by the dimension of <symbols>. The sampling frequency 
    is computed as <fs> = <L>*<Lc>/<T>. A warning is raised if this sampling 
    frequency is low compared to the theoretical signal bandwidth <B> + 2*<Lc>/<T>.

    Parameters
    ----------
    symbols : Numpy 2D array
        Array of phase-coded symbols. The row number corresponds to the number
        of pulses (preamble excluded), while the column number gives the number 
        of symbols in one pulse.
    preamble: Numpy 1D array
        Preamble symbols. The length of the sequence must be equal to the number 
        of symbols in one pulse (number of columns of <symbols>).
    L : Integer
        Oversampling factor for communication symbols.
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    T_PRI : Float
        PRI duration.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
    centered : Boolean, optional
        If set to <True>, the total bandwidth ranged by the chirp is centered around 0. 
        The default is True.

    Returns
    -------
        Numpy 1D array
        TX signal.

    """
    P,Lc = symbols.shape
    Nt = L*Lc
    Tc = T/Lc
    fs = L/Tc
    
    if fs <= B + 2/Tc or fs <= B + 2/Tc:
       print("WARNING: low sampling rate at TX!")
    
    waveform = generate_chirp_waveform(B,T,fs,up,centered)
    tx_sig,pulses = generate_pulse_train(waveform,P+1,T_PRI,fs)    
    
    symbols_extended = concatenate(([preamble],symbols))
    
    for p in range(P+1):
        shaped_symbols = pulse_shaping(symbols_extended[p,:], L)[:Nt]
        pulses[p,:Nt] *= shaped_symbols
    
    tx_sig = pulse2time(pulses,fs)
    return tx_sig

def SISO_PC_FMCW_DFRC_RADAR_RX(rx_sig,symbols,preamble,M,B,T,T_PRI,up=True,centered=True,zeropad_delay=1,zeropad_doppler=1):
    """
    SISO PC-FMCW DFRC RADAR Receiver.

    Processes the received SISO PC-FMCW DFRC signal <rx_sig> to generate a 
    delay-Doppler map.

    The sampling frequency is computed as <fs> = <M>*<Lc>/<T>. A warning is 
    raised if this sampling frequency is low compared to the theoretical signal 
    bandwidth <B> + 2*<Lc>/<T>.

    Parameters
    ----------
    rx_sig : Numpy 1D array
        Input signal.
    symbols : Numpy 2D array
        Array of phase-coded symbols. The row number corresponds to the number
        of pulses, while the column number gives the number of symbols in one pulse.
    preamble: Numpy 1D array
        Preamble symbols. The length of the sequence must be equal to the number 
        of symbols in one pulse (number of columns of <symbols>).
    M : Integer
        Oversampling factor.
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    T_PRI : Float
        PRI duration.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
    centered : Boolean, optional
        If set to <True>, the total bandwidth ranged by the chirp is centered around 0. 
        The default is True.
    zeropad_delay : Integer, optional
        Zero-padding factor for delay axis. The default is 1.
    zeropad_doppler : Integer, optional
        Zero-padding factor for Doppler axis. The default is 1.

    Returns
    -------
        Numpy 2D array
        Output delay-Doppler map.

    """
    P,Lc = symbols.shape
    Tg = T_PRI - T
    Nt = M*Lc
    Tc = T/Lc
    fs = M/Tc
    Ng = int(Tg*B)
    f_cut = B/T*Tg
    
    if fs <= B + 2/Tc or fs <= B + 2/Tc:
       print("WARNING: low sampling rate at RX!")
    
    received_pulses = time2pulse(rx_sig,P+1,T_PRI,fs)[1:,:Nt]
    
    processed_pulses = zeros(received_pulses.shape,dtype=received_pulses.dtype)
    for p,pulse in enumerate(received_pulses):
        shaped_symbols = pulse_shaping(symbols[p,:], M)[:Nt]
        
        dechirped_pulse = chirp_demodulation(pulse,B,T,fs,up,centered)
        delayed_pulse = GDF(dechirped_pulse,B,T,fs)
        radar_pulse = delayed_pulse * conj(shaped_symbols)
        processed_pulses[p,:] = LPF(radar_pulse,f_cut,fs)    
    
    N_doppler,N_delay = processed_pulses.shape
    delay_doppler_map = fftshift(fft2(processed_pulses,(N_doppler*zeropad_doppler,N_delay*zeropad_delay)),axes=0)
    out = delay_doppler_map[:,arange(0,-Ng*zeropad_delay,-1)] if up else delay_doppler_map[:,arange(Ng*zeropad_delay)]
    return out

def SISO_PC_FMCW_DFRC_COMM_RX(rx_sig,preamble,M,Lc,B,T,P,T_PRI,up=True,centered=True,CFO_estimation_method="fft",pulse_pilots=[],perfect_sync_vec=[False,False,False],path_params=[]):
    """
    SISO PC-FMCW DFRC RADAR Receiver.

    Process the received SISO PC-FMCW DFRC signal <rx_sig> to recover the transmitted
    symbols.
    
    The sampling frequency is computed as <fs> = <M>*<Lc>/<T>. A warning is 
    raised if this sampling frequency is low compared to the theoretical signal 
    bandwidth <B> + 2*<Lc>/<T>.

    Parameters
    ----------
    rx_sig : Numpy 1D array
        Input signal.
    preamble : Numpy 1D array
        Preamble symbols. This corresponds to the symbols inserted in the first pulse.
    M : Integer
        Oversampling factor.
    Lc : Integer
        Number of symbols per pulse.
    B : Float
        Chirp bandwidth.
    T : Float
        Chirp duration.
    P : Integer
        Number of pulses (preamble excluded).
    T_PRI : Float
        PRI duration.
    up : Boolean, optional
        If set to <True>, the chirp frequency is increasing. Otherwise, it is decreasing.
        The default is True.
    centered : Boolean, optional
        If set to <True>, the total bandwidth ranged by the chirp is centered around 0. 
        The default is True.
    CFO_estimation_method : String, optional
        CFO estimation method :
            - "moose" = Moose algorithm;
            - "fft" = FFT-based estimation.
        The default is "moose".
    pulse_pilots : Numpy 1D array, optional
        First symbol of each pulse. It should be specified only if the selected
        method is "fft". The default is [].
    perfect_sync_vec : List of three booleans, optional
        Perfect synchronisation vector. For each value set to True, a perfect 
        synchronisation is performed based on the given <path_params>.
        1- Complex attenuation estimation;
        2- Frequency offset estimation;
        3- Pulse synchronisation.
        The default is [False,False,False].
    path_params : List of 3 elements, optional
        List of 3 elements containing the path parameters:
            
        <path_params> = [alpha, f_D, tau]
         
        The default is [].

    Returns
    -------
    symbols_hat : Numpy 1D array
        Estimated symbols.
    path_params_hat : Tuple of estimated path parameters (complex attenuation,
        Doppler frequency and delay).

    """
    Nt = M*Lc
    Tc = T/Lc
    fs = M/Tc
    
    if fs <= B + 2/Tc or fs <= B + 2/Tc:
       print("WARNING: low sampling rate at RX!")
    
    fs_out = Lc/T
    
    delay = ideal_alignment(M)
    Eu = filter_energy(M)

    shaped_preamble = pulse_shaping(preamble, M)[:Nt]
    preamble_pulse = generate_chirp_waveform(B,T,fs,up,centered)
    preamble_pulse *= shaped_preamble
    
    corr,max_idx = pulse_correlation(rx_sig, preamble_pulse)
    tau_hat = max_idx/fs
    
    max_idx_sel = int(round(path_params[2]*fs)) if perfect_sync_vec[2] else max_idx
    rx_sig_sync = rx_sig[max_idx_sel:]
    
    pulses = time2pulse(rx_sig_sync,P+1,T_PRI,fs)
    
    processed_pulses = zeros((P+1,Lc),dtype=complex64)
    for p,pulse in enumerate(pulses):
        dechirped_pulse = chirp_demodulation(pulse, B, T, fs, up, centered)
        matched_filter_output = matched_filter(dechirped_pulse,M)
        # processed_pulses[p,:] = timing_synchronisation(matched_filter_output, M, 1)[:Lc]
        processed_pulses[p,:] = downsample(matched_filter_output, M, delay)[:Lc]
    
    f_D_hat = estimate_frequency_offset(processed_pulses, T, T_PRI, fs, CFO_estimation_method, pulse_pilots, zeropad=99)
    f_D_hat_sel = path_params[1] if perfect_sync_vec[1] else f_D_hat
    sync_pulses = correct_frequency_offset(processed_pulses, T, T_PRI, fs_out, f_D_hat_sel)
    
    alpha_hat_tilde = sync_pulses[0,:] @ conj(preamble) / (preamble @ conj(preamble))
    alpha_hat = alpha_hat_tilde / Eu * exp(1j*2*pi*f_D_hat*max_idx/fs)   
    
    alpha_hat_tilde_sel = path_params[0] * Eu * exp(-1j*2*pi*f_D_hat_sel*max_idx_sel/fs) if perfect_sync_vec[0] else alpha_hat_tilde
    
    symbols_hat = reshape(conj(alpha_hat_tilde_sel)/abs(alpha_hat_tilde_sel)**2 * sync_pulses[1:,:] , (P*Lc,))
    
    path_params_hat = [alpha_hat, f_D_hat, tau_hat]
    return symbols_hat,path_params_hat
    
    
    
    