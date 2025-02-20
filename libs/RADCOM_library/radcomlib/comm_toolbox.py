"""
Useful communication blocks for basic transmission chains.
"""

__all__ = ["symbol_mapping",
           "inverse_mapping",
           "upsample",
           "downsample",
           "pulse_shaping",
           "matched_filter",
           "ideal_alignment", 
           "timing_synchronisation", 
           "estimate_frequency_offset", 
           "correct_frequency_offset", ]

__all__ += ["BPSK_const",
            "QPSK_const",
            "QAM16_const",
            "PSK16_const"]

### Imports
from numpy import array,arange,zeros,ones
from numpy import reshape,transpose
from numpy import convolve
from numpy import sqrt,real,imag,conj,sign,abs,ceil,exp,angle,argmax
from numpy import complex64,pi
from numpy.fft import fft,fftshift

from commpy.filters import rcosfilter,rrcosfilter

### Exceptions
class InputError(Exception):
    pass

### Functions
def symbol_mapping(bits, const="BPSK"):
    """
    Performs symbol mapping on bit stream <bits>, using the constellation defined by <const>.

    Parameters
    ----------
    bits : List or numpy integer 1D array
        Input bit stream, containing only zeros and ones.
    const : String
        Constellation :
            - "BPSK"  : BPSK constellation;
            - "QPSK"  : QPSK constellation;
            - "16QAM" : 16-QAM constellation;
            - "16PSK" : 16-PSK constellation.

    Raises
    ------
    InputError
        Raised when an incorrect constellation is specified.

    Returns
    -------
    Numpy complex64 1D array
        Output symbol stream.
    """

    if ((array(bits) != 0)*(array(bits) != 1)).any():
        raise InputError("Input bit stream contains incorrect elements !")
    
    if const == "BPSK":
        out = zeros((len(bits),),dtype=complex64)
        out = -2*array(bits) + 1
    elif const == "QPSK":
        out = zeros((int(len(bits)/2),),dtype=complex64)
        bits_I = array(bits[0::2])
        bits_Q = array(bits[1::2])
        out = (-2*bits_I + 1) /sqrt(2) + 1j * (-2*bits_Q + 1) /sqrt(2)
    elif const == "16QAM":
        MSB0 = array(bits[::4])
        MSB1 = array(bits[1::4])
        LSB0 = array(bits[2::4])
        LSB1 = array(bits[3::4])        
        out = (2*LSB0-1) * (3 - 2*LSB1) / sqrt(10) + 1j * (2*MSB0-1) * (3 - 2*MSB1) / sqrt(10)
    elif const == "16PSK":
        MSB0 = array(bits[::4])
        MSB1 = array(bits[1::4])
        LSB0 = array(bits[2::4])
        LSB1 = array(bits[3::4])    
        out = exp(1j * ((2*MSB0-1)*pi/2 + (2*MSB0-1)*(2*MSB1-1)*pi/4 - (2*MSB0-1)*(2*MSB1-1)*(2*LSB0-1)*pi/8 + (2*MSB0-1)*(2*MSB1-1)*(2*LSB0-1)*(2*LSB1-1)*pi/16 + pi/16))
    else:
        raise InputError("Unknown constellation specified : " + const)
    return out.astype(complex64)

def inverse_mapping(symb, const="BPSK"):
    """
    Performs decision on the symbol stream <symb>, using the constellation defined by <const>.
    Decision is performed by selecting the symbol minimising the Euclidian distance with
    the symbols of the constellation (maximum likelihood).

    Parameters
    ----------
    symb : Numpy complex64 1D array
        Input symbol stream.
    const : String
        Constellation :
            - "BPSK"  : BPSK constellation;
            - "QPSK"  : QPSK constellation;
            - "16QAM" : 16-QAM constellation;
            - "16PSK" : 16-PSK constellation.

    Raises
    ------
    InputError
        Raised when an incorrect constellation is specified.

    Returns
    -------
    Numpy integer 1D array
        Output bit stream.
    """
    
    if const == "BPSK":
        out = zeros((len(symb),),dtype=int)
        out = real(symb) < 0
    elif const == "QPSK":
        out = zeros((len(symb)*2,),dtype=int)
        bits_r = -sign(real(symb))
        bits_i = -sign(imag(symb))
        out[0::2] = (bits_r + 1)/2
        out[1::2] = (bits_i + 1)/2
    elif const == "16QAM":
        MSB0 = sign(imag(symb)) > 0
        MSB1 = abs(imag(symb)) < 2/sqrt(10)
        LSB0 = sign(real(symb)) > 0
        LSB1 = abs(real(symb)) < 2/sqrt(10)
        arr = array([MSB0,MSB1,LSB0,LSB1])
        out = reshape(arr.T,(arr.size,))
    elif const == "16PSK":
        phi = angle(symb*exp(-1j*pi/16))
        MSB0 = sign(phi) > 0
        phi -=  (2*MSB0-1)*pi/2
        MSB1 =  ((2*MSB0-1) * sign(phi) + 1)/2
        phi -=  (2*MSB0-1)*(2*MSB1-1)*pi/4
        LSB0 = (-(2*MSB0-1)*(2*MSB1-1) * sign(phi) + 1)/2
        phi +=  (2*MSB0-1)*(2*MSB1-1)*(2*LSB0-1)*pi/8
        LSB1 =  ((2*MSB0-1)*(2*MSB1-1)*(2*LSB0-1) * sign(phi) + 1)/2
        arr = array([MSB0,MSB1,LSB0,LSB1])
        out = reshape(arr.T,(arr.size,))
    else:
        raise InputError("Unknown constellation specified : " + const)
    return out.astype(int)

def upsample(x,L):
    """
    Upsample the input signal <x> by inserting <L>-1 zeros between every sample.

    Parameters
    ----------
    x : Numpy 1D array
        Input signal.
    L : Integer
        Upsampling factor.

    Returns
    -------
    Numpy 1D array
        Upsampled signal.
    """
    zeros_array = zeros((L,len(x)),dtype=complex64)
    zeros_array[0,:] = x
    return reshape(transpose(zeros_array),(L*len(x),))

def downsample(x,M,offset=0):
    """
    Downsample the input signal <x> by taking one sample every <M> samples, starting
    at sample <offset>.

    Parameters
    ----------
    x : Numpy 1D array
        Input signal.
    M : Integer
        Downsampling factor.
    offset : Integer, optional
        Starting index for downsampling. The default is 0.

    Returns
    -------
    Numpy 1D array
        Downsampled signal.
    """
    return x[offset::M]

def pulse_shaping(x,L,filter_type="rect",alpha=0,symbols_length=8):
    """
    Apply a pulse shaping on the input signal <x>. The signal <x> is first upsampled
    by inserting <L>-1 zeros and then convolved with the shaping filter.

    Parameters
    ----------
    x : Numpy 1D array
        Input signal.
    L : Integer
        Oversampling factor.
    filter_type : String, optional
        Pulse shaping filter :
            - "rect" = rectangular
            - "rc" = raised cosine;
            - "rrc" = root raised cosine.
            The default is "rect".
    alpha : Float, optional
        Roll-off factor, comprised between 0 and 1. It should be specified only
        if <filter_type> = "rc" or "rrc".
        The default is 0.
    symbols_length : Integer, optional
        Number of overlapped symbols. The filter length is equal to
        <symbol_length>*<L>. It should be specified only if <filter_type> = "rc" 
        or "rrc". 
        The default is 8.

    Raises
    ------
    InputError
        Raised when an incorrect filter type is specified.

    Returns
    -------
    Numpy 1D array
        Output sample stream.
    """
    
    x_up = upsample(x,L)
    if filter_type=='rect':
        h = ones((L,))
    elif filter_type=='rrc':
        h = rrcosfilter(L*symbols_length,alpha,1,L)[1]
    elif filter_type=='rc':
        h = rcosfilter(L*symbols_length,alpha,1,L)[1] 
    else:
        raise InputError("Invalid filter_type specified : " + filter_type)
    return convolve(x_up,h,mode='full')

def matched_filter(x,M,filter_type='rect',alpha=0,symbols_length=8):
    """
    Apply a matched filter on the input signal <x>.
    
    Parameters
    ----------
    x : Numpy 1D array
        Input signal.
    M : Integer
        Oversampling factor.
    filter_type : String, optional
        Matched filter :
            - "rect" = rectangular
            - "rc" = raised cosine;
            - "rrc" = root raised cosine.
            The default is "rect".
    alpha : Float, optional
        Roll-off factor, comprised between 0 and 1. It should be specified only
        if <filter_type> = "rc" or "rrc".
        The default is 0.
    symbols_length : Integer, optional
        Number of overlapped symbols. The filter length is equal to
        <symbol_length>*<L>. It should be specified only if <filter_type> = "rc" 
        or "rrc". 
        The default is 8.

    Raises
    ------
    InputError
        Raised when an incorrect filter type is specified.

    Returns
    -------
    Numpy 1D array
        Output sample stream.
    """
    
    if filter_type=='rect':
        h = ones((M,))
    elif filter_type=='rrc':
        h = rrcosfilter(M*symbols_length,alpha,1,M)[1]
    elif filter_type=='rc':
        h = 1
    else:
        raise InputError("Invalid filter_type specified : " + filter_type)
    return convolve(x,h,mode='full')

def ideal_alignment(M,filter_type='rect',symbols_length=8):
    """
    Computes the discrete delay induced by the pulse shaping and matched filtering
    depending on the selected filter type <filter_type>, filter parameters and 
    oversampling factor.

    Parameters
    ----------
    M : Integer
        Oversampling factor.
    filter_type : String, optional
        Pulse shaping & matched filter :
            - "rect" = rectangular
            - "rc" = raised cosine;
            - "rrc" = root raised cosine.
            The default is "rect".
    symbols_length : Integer, optional
        Number of overlapped symbols. The filter length is equal to
        <symbol_length>*<L>. It should be specified only if <filter_type> = "rc" 
        or "rrc". 
        The default is 8.

    Raises
    ------
    InputError
        Raised when an incorrect filter type is specified.

    Returns
    -------
    Integer
        Discrete delay induced by the filters.
    """
    
    if filter_type == 'rect':
        delay = M - 1
    elif filter_type == 'rrc':
        delay = symbols_length * M
    elif filter_type == 'rc':
        delay = int(ceil(symbols_length * M/2))
    else:
        raise InputError("Invalid filter_type specified : " + filter_type)
    return delay

def filter_energy(M,filter_type='rect'):
    """
    Computes the amplitude of the maximum of the convolution between the pulse 
    shaping and matched filtering depending on the selected filter type <filter_type>
    and the oversampling factor.

    Parameters
    ----------
    M : Integer
        Oversampling factor.
    filter_type : String, optional
        Pulse shaping & matched filter :
            - "rect" = rectangular
            - "rc" = raised cosine;
            - "rrc" = root raised cosine.
            The default is "rect".
            
    Raises
    ------
    InputError
        Raised when an incorrect filter type is specified.

    Returns
    -------
    Float
        Amplitude of the maximum of the convolution between the pulse shaping 
        and matched filtering.
    """
    
    if filter_type == 'rect':
        Eu = M
    elif filter_type == 'rrc':
        Eu = M
    elif filter_type == 'rc':
        Eu = 1
    else:
        raise InputError("Invalid filter_type specified : " + filter_type)
    return Eu

def estimate_frequency_offset(pulses,T,T_PRI,fs,method="moose",training_symbs=[],zeropad=0):
    """
    Estimates the frequency offset affecting the input signal <pulses> in pulse
    notation, using either the Moose algorithm or an FFT performed over the pulses.
    In the first case, the first pulse must be a 2-periodic training sequence with
    good auto-correlation properties, which should not be specified. In the second 
    case, the first symbol of each pulse is considered to be known, and this sequence
    should be specified as <training_symbs>. 

    Parameters
    ----------
    pulses : Numpy 2D array
        Input signal in pulse notation. The first axis refers to the pulse index,
        while the second axis refers to the time index.
    T : Float
        Chirp duration.
    T_PRI : Float
        Pulse Repetition Interval duration.
    fs : Float
        Sampling frequency.
    method : String, optional
        Method to estimate the frequency offset :
            - "moose" = Moose algorithm;
            - "fft" = FFT-based estimation.
        The default is "moose".
    training_symbs : Numpy 1D array, optional
        First symbol of each pulse. It should be specified only if the selected
        method is "fft". The default is [].
    zeropad : Integer, optional
        Zero-padding for the FFT estimation of the frequency offset. It should 
        be specified only if the selected method is "fft". The default is 0.

    Returns
    -------
    Float
        Estimate of the frequency offset.
    """
    
    (P,N) = pulses.shape
    
    fD_hat = 0
    if method == "fft":
        fft_len = (zeropad + 1)*P
        fft_pulses = fftshift(fft(pulses[:,0] * conj(training_symbs)/(abs(training_symbs)**2),n=fft_len))
        f = (arange(fft_len) - ceil(fft_len/2)) / (fft_len*T_PRI)
        max_idx = argmax(abs(fft_pulses))
        fD_hat = f[max_idx]
    else:
        fD_hat = angle(sum(pulses[0,int(round(N/2)):N] * conj(pulses[0,:int(round(N/2))]))) / (pi*T)
        
    return fD_hat

def correct_frequency_offset(pulses,T,T_PRI,fs,fD_hat):
    """
    Compensates the frequency offset affecting the input signal <pulses> in pulse
    notation.

    Parameters
    ----------
    pulses : Numpy 2D array
        Input signal in pulse notation. The first axis refers to the pulse index,
        while the second axis refers to the time index.
    T : Float
        Chirp duration.
    T_PRI : Float
        Pulse Repetition Interval duration.
    fs : Float
        Sampling frequency.
    fD_hat : Float
        Estimate of the frequency offset.

    Returns
    -------
    Numpy 1D array
        Corrected signal in pulse notation. The first axis refers to the pulse index,
        while the second axis refers to the time index.
    """
    
    (P,N) = pulses.shape
    
    pulses_corr = pulses.copy()
    for p in range(P):
        pulses_corr[p,:] *= exp(-1j*2*pi*fD_hat*arange(N)/fs) * exp(-1j*2*pi*fD_hat*T_PRI*p)
    
    return pulses_corr

def timing_synchronisation(r,M_in,M_out):
    """
    Performs a timing synchronisation on the input signal <r> based on the 
    maximum energy method.

    <!> TO OPTIMISE <!>
    Parameters
    ----------
    r : Numpy 1D array
        Input signal.
    M_in : Integer
        Input oversampling factor.
    M_out : Integer
        Output oversampling factor. It should be smaller than <M_in> and <M_in>/<M_out>
        should be an integer.

    Raises
    ------
    InputError
        Raised when M_in and M_out are not selected accordingly to the conditions
        listed above.

    Returns
    -------
    Numpy 1D array
        Output downsampled signal.
    """
    
    if M_in/M_out % 1 > 1e-8 or M_in < M_out:
        raise InputError("Invalid combination of M_in ({}) and M_out ({})!".format(M_in,M_out))
        
    N_in = len(r)
    N_out = int(N_in*M_out/M_in)
    
    J = zeros((M_in,),dtype=complex64)
    for m in range(M_in):
        J[m] = sum(abs(r[m:N_in:M_in])**2)
    
    max_idx = argmax(J)
    r_sync = downsample(r,int(M_in/M_out),max_idx)[:N_out]
        
    return r_sync

### Variables

bits = [0,1]
BPSK_const = symbol_mapping(bits,"BPSK")
bits = [0,0,0,1,1,0,1,1]
QPSK_const = symbol_mapping(bits,"QPSK")
bits = [0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1, \
        1,0,0,0,1,0,0,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1]
QAM16_const = symbol_mapping(bits,"16QAM")
PSK16_const = symbol_mapping(bits,"16PSK")