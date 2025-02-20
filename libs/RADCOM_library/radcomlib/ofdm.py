"""
Blocks needed to setup an OFDM DFRC system.
"""

__all__ = ["OFDM_modulation",
           "OFDM_demodulation", 
           "OFDM_synchronisation",
           "OFDM_channel_estimate",
           "OFDM_channel_equalisation",
           "OFDM_channel_parameters_estimation",
           "SISO_OFDM_DFRC_TX",
           "SISO_OFDM_DFRC_RADAR_RX",
           "SISO_OFDM_DFRC_COMM_RX"]

### Imports
from numpy import array,arange,zeros,ones
from numpy import concatenate,reshape,meshgrid,unravel_index
from numpy import convolve, argmax
from numpy import sqrt,sum,abs,angle,conj,real,imag,exp,sin,floor,round
from numpy import pi,complex64
from numpy import inf as Inf # Correct the 'Inf' import from numpy

from numpy.fft import fft,ifft,ifftshift

from scipy.interpolate import interp2d

from .comm_toolbox import upsample
from .comm_toolbox import estimate_frequency_offset
from .comm_toolbox import correct_frequency_offset

from .radar_toolbox import pulse_correlation

### Exceptions

### Functions
def OFDM_modulation(I,L_CP,L=1):
    """
    Modulates input symbols <I> using Cyclic Prefix (CP) - Orthogonal Frequency 
    Division Multiplexing (OFDM) modulation.
    
    Parameters
    ----------
    I : Numpy 2D array
        Input symbols. The number of lines corresponds to the number of generated 
        OFDM symbols, while the number of columns corresponds to the number of
        subcarriers.
    L_CP : Integer
        Cyclic prefix length.
    L : Integer, optional
        Oversampling factor. For discrete CP-OFDM implementation with pulse shaping
        and matched filter, set <L> to 1 and modify the oversampling factor
        of the pulse shaping and matched filter blocks. For continuous time 
        implementation with windowing, modify <L> here.
        The default is 1.

    Returns
    -------
    Numpy 1D array
        Output sample stream of the modulated symbols
    """
    
    N_OFDM,N_sc = I.shape
    
    ifft_I = sqrt(N_sc*L)*ifft(I,L*N_sc)
    
    out_blk = concatenate((ifft_I[:,-L*L_CP:],ifft_I),axis=1)
    
    return reshape(out_blk,(out_blk.size,))

def OFDM_demodulation(r,N_OFDM,N_sc,L_CP,M=1):
    """
    Demodulates the input sample stream <r>, obtained after modulating symbols
    using Cyclic Prefix (CP) - Orthogonal Frequency Division Multiplexing (OFDM) 
    modulation.

    Parameters
    ----------
    r : Numpy 1D array
        Input sample stream.
    N_OFDM : Integer
        Number of OFDM symbols.
    N_sc : Integer
        Number of subcarriers.
    L_CP : Integer
        Cyclic prefix length.
    M : Integer, optional
        Oversampling factor. For discrete CP-OFDM implementation with pulse shaping
        and matched filter, set <M> to 1 and modify the oversampling factor
        of the pulse shaping and matched filter blocks. For continuous time 
        implementation with windowing, modify <M> here.
        The default is 1.

    Returns
    -------
    Numpy 2D array
        Demodulated symbols.
    """
    
    Nr = N_OFDM*M*(L_CP+N_sc)
    r_trunc = r[:Nr]
    
    N = len(r_trunc)
    if N < Nr:
        zeropad = zeros((Nr-N,))    
        r_blk = reshape(concatenate((r_trunc,zeropad)),(N_OFDM,M*(L_CP+N_sc)))[:,M*L_CP:]
    else:
        r_blk = reshape(r_trunc,(N_OFDM,M*(L_CP+N_sc)))[:,M*L_CP:]

    r_fft = 1/sqrt(N_sc*M)*fft(r_blk)
    
    return r_fft[:,:N_sc]

def OFDM_synchronisation(r,N_sc,T,M=1):
    """
    Estimates the delay and frequency offset affecting the OFDM payload based
    on the two-periodic OFDM training symbol at the start of the payload, using
    the Schmidl & Cox algorithm. Note that the delay does not correspond exactly
    at the start of the OFDM payload, but it is comprised in the duration of the
    CP of the OFDM training symbol.

    Parameters
    ----------
    r : Numpy 1D array
        Input sample stream.
    N_sc : Integer
        Number of subcarriers.
    T : Float
        OFDM symbol duration (CP excluded).
    M : Integer, optional
        Oversampling factor. The default is 1.

    Returns
    -------
    max_idx_hat : Integer
        Estimate of the index of the start of the payload. This index does not 
        correspond exactly to the start of the OFDM payload, but it is comprised
        in the duration of the CP of the OFDM training symbol.
    CFO_hat : Float
        Frequency offset estimation.
    """
    
    Nh = int(N_sc/2)*M
    autocorr = convolve(conj(r[0:len(r)-Nh:]) * r[Nh::],ones((Nh,)),mode='valid')
    autocorr_power_1 = convolve(abs(r[0:len(r)-Nh:])**2,ones((Nh,)),mode='valid')
    autocorr_power_2 = convolve(abs(r[Nh::])**2        ,ones((Nh,)),mode='valid')
    
    sync_metric = abs(autocorr)**2/(autocorr_power_1*autocorr_power_2)
    
    max_idx = argmax(sync_metric)
    sync_metric[sync_metric > 0.98*sync_metric[max_idx]] = -1
    
    first_left  = sum(sync_metric[:max_idx+1] != -1)
    first_right = max_idx + sum(sync_metric[max_idx:] == -1) - 1
    max_idx_hat = int((first_left + first_right)/2)

    CFO_hat = angle(autocorr[max_idx_hat]) / (pi*T)
                
    return max_idx_hat,CFO_hat
           
def OFDM_channel_estimate(r_fft,pilots,N_OFDM,N_sc,L_CP,Nt,Nf):
    """
    Computes an estimation of the channel in the frequency domain based on the 
    received symbols <r_fft> and pilots repartited in time and frequency. The
    structure of the OFDM payload is the following 
    
                    Nf = 4
                <------------->
          ^  | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
  Nt = 2  |  |---|---|---|---|---|---|---|---|---| ... |---|---|---|---|---|
          v  | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
             |---|---|---|---|---|---|---|---|---| ... |---|---|---|---|---|
             | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
             |---|---|---|---|---|---|---|---|---| ... |---|---|---|---|---|
             | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
             | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
         
    where crosses represent pilots positions, each line is an OFDM symbol (without
    the CP) while each column is a frequency.
    
    Parameters
    ----------
    r_fft : Numpy 2D array
        Received symbols.
    pilots : Numpy 2D array
        Pilot symbols.
    N_OFDM : Integer
        Number of OFDM symbols.
    N_sc : Integer
        Number of subcarriers.
    L_CP : Integer
        Number of cylic prefixes.
    Nt : Integer
        Pilot spacing between OFDM symbols.
    Nf : Integer
        Pilot spacing between subcarriers.

    Returns
    -------
    Numpy 2D
        Output channel estimate in the frequency domain.
    """
    
    def interp2d_complex(x,y,z,x_new,y_new,kind='linear'):
        fr = interp2d(x,y,real(z.T),kind=kind)
        fi = interp2d(x,y,imag(z.T),kind=kind)
        return fr(x_new,y_new).T + 1j*fi(x_new,y_new).T

    pilots_idx_f = concatenate((arange(0,N_sc-1,Nf),[N_sc-1]))
    pilots_idx_t = concatenate((arange(0,N_OFDM-1,Nt),[N_OFDM-1]))
        
    (pilots_t_mesh,pilots_f_mesh) = meshgrid(pilots_idx_t,pilots_idx_f)
    
    r_pilots = r_fft[pilots_t_mesh.T,pilots_f_mesh.T]
    H_pilots = r_pilots / pilots
    
    H_interp = interp2d_complex(pilots_idx_t,pilots_idx_f,H_pilots,arange(N_OFDM),arange(N_sc),kind="linear") 

    return H_interp

def OFDM_channel_equalisation(OFDM_symbols,channel_estimate_FD,kind='ZF',SNR=Inf):
    """
    Equalises the received OFDM symbols <OFDM_symbols> based on a given channel
    estimate in the frequency domain.

    Parameters
    ----------
    OFDM_symbols : Numpy 2D array
        Input OFDM symbols.
    channel_estimate_FD : Numpy 2D array
        Channel estimate in the frequency domain.
    kind : String, optional
        Equalisation method:
            - "ZF"/"LSE" = zero-forcing or maximum likelihood;
            - "MMSE"/"MAP" = minimum mean square error or maximum a postiori.
        If <kind>="MMSE" or "MAP", an estimation of the SNR must be provided.
        The default is 'ZF'.
    SNR : Float, optional
        Estimation of the SNR. It must be provided only if the selected
        equalisation method is the minimum mean square error or the maximum
        a postiori.
        The default is Inf.

    Returns
    -------
    Numpy 2D array
        Equalised OFDM symbols.
    """
    
    if kind == "MMSE" or kind == "MAP":
        output_const = OFDM_symbols * conj(channel_estimate_FD) / (abs(channel_estimate_FD)**2 + 1/SNR)
    else:
        output_const = OFDM_symbols / channel_estimate_FD
    return output_const

def OFDM_channel_parameters_estimation(rx_sig,channel_estimate_FD,preamble,tau_offset,L_CP,fs,M,dx,centered,zeropad_doppler,zeropad_delay):
    N_OFDM,N_sc = channel_estimate_FD.shape

    delay_doppler_map_comm = ifftshift(ifft(fft(channel_estimate_FD,axis=0,n=N_OFDM*zeropad_doppler),axis=1,n=N_sc*M*zeropad_delay),axes=0)

    max_idx = unravel_index(abs(delay_doppler_map_comm).argmax(),delay_doppler_map_comm.shape)

    fD,tau_tilde = (max_idx - centered*floor(array(delay_doppler_map_comm.shape)/2)) * dx / array([zeropad_doppler,zeropad_delay])
    tau = tau_offset + tau_tilde - (N_sc*M + L_CP*M)/fs
    start_idx = int(round(tau*fs))

    rx_preamble = rx_sig[start_idx:start_idx + L_CP*M + N_sc*M] * exp(-1j*2*pi*fD*(tau + arange(L_CP*M+N_sc*M)/fs))

    # rx_preamble_y = OFDM_demodulation(rx_preamble, 1, N_sc, L_CP, M)

    alpha_tilde = rx_preamble @ conj(preamble) / (preamble @ conj(preamble))
    alpha = alpha_tilde * exp(1j*2*pi*fD*tau) 

    path_params_hat = [alpha, fD, tau]

    return path_params_hat

def SISO_OFDM_DFRC_TX(symbols, preamble, L_CP, L=1, periodic_preamble=True):
    """
    SISO OFDM DFRC Transmitter.
    
    Generates the transmitted signal, a train of <P>+1 OFDM symbols of duration
    (<N>+<L_CP>)*<L> samples, where <P> and <N> are the dimensions of <symbols>.
    The first OFDM symbol is a 2-periodic sequence generated from the preamble.

    Parameters
    ----------
    symbols : Numpy 2D array
        Input symbols. The row number corresponds to the number of pulses (preamble
        excluded), while the column number gives the number of symbols in one pulse.
    preamble : Numpy 1D array
        Preamble symbols. The length of the sequence must be equal to half the number
        of symbols in one pulse (number of columns of <symbols>) if <periodic_preamble> is True, or equal to the number of symbols if it is False.
    L_CP : Integer
        Cyclic prefix length.
    L : Integer, optional
        Oversampling factor. Note that the exponential basis functions are oversampled,
        and there is no pulse shaping.
        The default is 1., 
    periodic_preamble : Boolean, optional
        If True, the preamble is upsampled to be repeated twice in the time domain. The default is True.

    Returns
    -------
    out : Numpy 1D array
        TX signal.
    """
    
    if periodic_preamble:
        preamble_upsampled = upsample(preamble,2)
    else:
        preamble_upsampled = preamble
    symbols_extended = concatenate(([preamble_upsampled], symbols))
    
    out = OFDM_modulation(symbols_extended, L_CP, L)
    return out

def SISO_OFDM_RADAR_RX(rx_sig, symbols, L_CP, M=1):

    P,N_sc = symbols.shape
    
    N = (N_sc + L_CP)*M
    
    demod_output = OFDM_demodulation(rx_sig, P, N_sc, L_CP, M)
    
    H_hat = demod_output / symbols
    
    delay_doppler_map = ifftshift(ifft(fft(H_hat,axis=0),axis=1)[:,:M*L_CP],axes=0)
    
    return delay_doppler_map
    

def SISO_OFDM_DFRC_RADAR_RX(rx_sig, symbols, L_CP, M=1, zeropad_delay=1, zeropad_doppler=1):
    """
    SISO OFDM DFRC RADAR Receiver.
    
    Processes the received SISO OFDM DFRC signal <rx_sig> to generate a delay-Doppler
    map.

    Parameters
    ----------
    rx_sig : Numpy 1D array
        Input signal.
    symbols : Numpy 2D array
        Transmitted symbols. The row number corresponds to the number of pulses
        (preamble excluded) while the number of columns gives the number of symbols
        in one pulse.
    L_CP : Integer
        Cyclic prefix length..
    M : Integer, optional
        Oversampling factor. Note that the exponential basis functions are oversampled,
        and there is no pulse shaping. 
        The default is 1.
    zeropad_delay : Integer, optional
        Zero-padding on the delay axis; the number of zeros inserted is equal to
        <zeropad_delay>-1 times the number of transmitted symbols.
        The default is 1.
    zeropad_doppler : Integer, optional
        Zero-padding on the Doppler axis; the number of zeros inserted is equal to
        <zeropad_doppler>-1 times the number of transmitted pulses.
        The default is 1.

    Returns
    -------
    delay_doppler_map : Numpy 2D array
        Output delay-Doppler map.
    """
    
    P,N_sc = symbols.shape
    
    N = (N_sc + L_CP)*M
    
    demod_output = OFDM_demodulation(rx_sig[N:], P, N_sc, L_CP, M)
    
    H_hat = demod_output / symbols
    
    delay_doppler_map = ifftshift(ifft(fft(H_hat,axis=0,n=P*zeropad_doppler),axis=1,n=N_sc*M*zeropad_delay)[:,:M*L_CP*zeropad_delay],axes=0)
    
    return delay_doppler_map

def SISO_OFDM_DFRC_COMM_RX(rx_sig, preamble, pilots, N_sc, L_CP, N_OFDM, Nt, Nf, fs, M=1, perfect_pulse_sync=False, tau_sync=0, perfect_CSI=False, path_params=[],zeropad_delay=1,zeropad_doppler=1,ofdm_synchronisation=True,doppler_pilots=[]):
    """
    SISO OFDM DFRC Communication Receiver.
    
    Processes the received SISO OFDM DFRC signal <rx_sig> to recover the transmitted
    symbols. The structure of the OFDM payload is the following 
    
                    Nf = 4
                <------------->
          ^  | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
  Nt = 2  |  |---|---|---|---|---|---|---|---|---| ... |---|---|---|---|---|
          v  | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
             |---|---|---|---|---|---|---|---|---| ... |---|---|---|---|---|
             | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
             |---|---|---|---|---|---|---|---|---| ... |---|---|---|---|---|
             | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
             | X |---|---|---| X |---|---|---| X | ... |---| X |---|---| X |
         
    where crosses represent pilots positions, each line is an OFDM symbol (without
    the CP) while each column is a frequency.
    
    ! The pilots symbols are not deleted from the output symbol stream !
    
    Parameters
    ----------
    rx_sig : Numpy 1D array
        Input signal.
    preamble : Numpy 1D array
        Preamble symbols in the frequency domain.
    pilots : Numpy 2D array
        Matrix of pilot symbols. The rows correspond to the time dimension while 
        the columns correspond to the frequency dimension.
    N_sc : Integer
        Number of subcarriers.
    L_CP : Integer
        Cyclic prefix length.
    N_OFDM : Integer
        Number of OFDM symbols.
    Nt : Integer
        Pilot spacing between OFDM symbols.
    Nf : Integer
        Pilot spacing between subcarriers.
    fs : Float
        Sampling frequency.
    M : Oversampling factor, optional
        Oversampling factor. Note that the exponential basis functions are oversampled,
        and there is no pulse shaping. 
        The default is 1.
    perfect_pulse_sync : Boolean, optional
        If set to True, the start of the payload is supposed to be located at 
        <tau_sync>. 
        The default is False.
    tau_sync : Float, optional
        Payload delay. Used if <perfect_pulse_sync> is set to True.
        The default is 0.
    perfect_CSI : Boolean, optional
        If set to True, the channel state informations given by <path_params> are
        used for equalisation. 
        The default is False.
    path_params : List, optional
        List of K elements containing the paths parameters. Each element is
        another list containing the complex attenuations, Doppler frequencies 
        and delays of the paths. Here is an example for a three-path channel:
            
        [[alpha_0,f_D_0,tau_0], [alpha_1,f_D_1,tau_1], [alpha_2,f_D_2,tau_2]]. 
        
        The default is [].
    zeropad_delay : Integer, optional
        Zero-padding on the delay axis; the number of zeros inserted is equal to
        <zeropad_delay>-1 times the number of transmitted symbols.
        The default is 1.
    zeropad_doppler : Integer, optional
        Zero-padding on the Doppler axis; the number of zeros inserted is equal to
        <zeropad_doppler>-1 times the number of transmitted pulses.
        The default is 1.
    ofdm_synchronisation : Boolean, optional
        If set to True, the OFDM synchronisation algorithm is used to estimate
        the delay.

    Returns
    -------
    symbols_hat : Numpy 2D array
        Output symbols (pilots included).
    channel_estimate_FD : Tuple of 2 elements
        Contains the delay computed with the timing synchronisation algorithm,
        and the channel estimate in the frequency domain.
    path_params_hat : Tuple of estimated path parameters (complex attenuation, Doppler frequency and delay), assuming a flat fading channel.
        
    """
    
    T = M*N_sc/fs
    T_CP = M*L_CP/fs
    T_PRI = T + T_CP
    
    # timing sync
    if ofdm_synchronisation:
        max_idx_hat,_ = OFDM_synchronisation(rx_sig, N_sc, T, M)
        tau_hat = (max_idx_hat + N_sc*M)/fs
        if perfect_pulse_sync:
            max_idx_hat = int(floor(tau_sync*fs)) 
            r_sync = rx_sig[max_idx_hat + N_sc*M + L_CP*M:]
        else: 
            r_sync = rx_sig[max_idx_hat + N_sc*M:]
        
        y = OFDM_demodulation(r_sync, N_OFDM, N_sc, L_CP, M)
    else:
        preamble_pulse = OFDM_modulation(array([preamble]), L_CP, M)
        _,max_idx_hat = pulse_correlation(rx_sig, preamble_pulse)
        tau_hat = max_idx_hat/fs

        max_idx_sel = int(floor(tau_sync*fs)) if perfect_pulse_sync else max_idx_hat
        r_sync = rx_sig[max_idx_sel:]

        y_with_preamble = OFDM_demodulation(r_sync, N_OFDM+1, N_sc, L_CP, M)
        y = y_with_preamble[1:,:]
    
    # channel estimation and equalisation
    if perfect_CSI: #wrong
        t_mesh, f_mesh = meshgrid(arange(N_OFDM),arange(N_sc))
        channel_estimate_FD = zeros((N_OFDM,N_sc),dtype=complex64)
        
        gamma_func = lambda eps: 1 if eps == 0 else 1/(M*N_sc) * sin(pi*eps)/sin(pi/N_sc*eps) * exp(1j*pi*(N_sc*M-1)/(N_sc*M)*eps)
        
        for alpha, f_D, tau in path_params:
            dtau = max_idx_hat - tau*fs
            channel_estimate_FD += alpha * exp(1j*2*pi*f_D*(dtau+T_PRI))  * exp(1j*2*pi*f_D*T_PRI*t_mesh.T) \
                * exp(-1j*2*pi*(dtau/T + L_CP/N_sc)*f_mesh.T) * gamma_func(f_D*T)

    else:
        channel_estimate_FD = OFDM_channel_estimate(y, pilots, N_OFDM, N_sc, L_CP, Nt, Nf)    

        symbols_hat = reshape(OFDM_channel_equalisation(y, channel_estimate_FD),(N_OFDM*N_sc,))

    # path parameters estimation
    if ofdm_synchronisation:
        preamble_upsampled = upsample(preamble,2)
        preamble_t = OFDM_modulation(preamble_upsampled.reshape((1,preamble_upsampled.size)), L_CP, M)
        path_params_hat = OFDM_channel_parameters_estimation(rx_sig,channel_estimate_FD,preamble_t,(max_idx_hat+N_sc*M)/fs,L_CP,fs,M,array([1/(N_OFDM*T_PRI),1/fs]),array([1,0]),zeropad_doppler,zeropad_delay)
    else:    
        f_D_hat = estimate_frequency_offset(y_with_preamble, T, T_PRI, fs, "fft", doppler_pilots, zeropad=99)
        sync_pulses = correct_frequency_offset(y_with_preamble, T, T_PRI, fs/M, f_D_hat)
        
        alpha_hat_tilde = sync_pulses[0,:] @ conj(preamble) / (preamble @ conj(preamble))
        alpha_hat = alpha_hat_tilde * exp(1j*2*pi*f_D_hat*max_idx_hat/fs)   
        
        path_params_hat = [alpha_hat, f_D_hat, tau_hat]
    
    # pilots_idx_f = concatenate((arange(0,N_sc-1,Nf),[N_sc-1]))
    # pilots_idx_t = concatenate((arange(0,P-1,Nt),[P-1]))
    # pilots_idx_t_mesh,pilots_idx_f_mesh = meshgrid(pilots_idx_t,pilots_idx_f)
    
    # symbols_hat[pilots_idx_t_mesh,pilots_idx_f_mesh] = Inf    
    # symbols_hat = delete(symbols_hat,where(symbols_hat.flatten() == Inf))  
    
    return symbols_hat, path_params_hat