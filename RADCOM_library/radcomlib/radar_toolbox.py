"""
Useful RADAR processing blocks for basic RADAR chains.
"""

__all__ = ["time2pulse", 
           "pulse2time", 
           "generate_pulse_train",
           "pulse_correlation", 
           "compute_CFAR_threshold", 
           "CFAR_detector_2D",
           "CFAR_detector", 
           "binary_map_processing",
           "targets_extraction",
           "compare_targets"]

### Imports
from numpy import zeros,ones,arange,array
from numpy import reshape,concatenate,ix_,pad
from numpy import correlate,argmax,ravel_multi_index,unravel_index,ndenumerate,delete,sort,argwhere
from numpy import abs,sum,min,floor,round
# from numpy import int

from numpy.matlib import repmat

from scipy.special import beta
from scipy.ndimage import binary_dilation
from scipy.signal import correlate as scipy_correlate

from math import comb

from numba import jit

from decimal import Decimal

### Exceptions
class InputError(Exception):
    pass

class CFARCoefficientMissingError(Exception):
    pass

### Functions

def time2pulse(x,P,T_PRI,fs):
    """
    Decomposes the input signal <x> to generate a signal in pulse notation:
    y(t,p) = x(t+p*(<T_PRI>)) for p=0,...,<P>-1 and t in [0,<T_PRI>], where
    <T_PRI> is the Pulse Repetition (PRI) Interval duration. 

    Parameters
    ----------
    x : Numpy 1D array
        Input signal.
    P : Integer
        Number of transmitted pulses.
    T_PRI : Float
        PRI duration.
    fs : Float
        Sampling frequency.

    Returns
    -------
    Numpy 2D array
        Output signal in pulse notation. The first axis refers to the pulse index,
        while the second axis refers to the time index.
    """
    
    N = int(fs*T_PRI)
    
    if len(x) > N*P:
        pulses = reshape(x[:P*N],(P,N))
    else:
        pulses = reshape(pad(x,(0,P*N-len(x)),mode='constant',constant_values=0),(P,N))

    return pulses

def pulse2time(pulses,fs):
    """
    Flattens the input <pulses> to generate a signal in time notation:
    y(t+p*<T_PRI>) = x(t,p) for p=0,...,<P>-1 and t in [0,<T_PRI>], where
    <T_PRI> is the Pulse Repetition Interval duration.

    Parameters
    ----------
    pulses : Numpy 2D array
        Input signal in pulse notation. The first axis refers to the pulse index,
        while the second axis refers to the time index.
    fs : Float
        Sampling frequency.

    Returns
    -------
    Numpy 1D array
        Output signal in time notation.
    """
    
    (P,N) = pulses.shape

    return reshape(pulses,(N*P,))

def generate_pulse_train(waveform,P,T_PRI,fs):
    """
    Generates a pulse train from the given waveform <waveform>. The pulse train
    consists of <P> repetitions of duration <T_PRI>, where each repetition starts
    with <waveforms> and is filled by zeros.

    Parameters
    ----------
    waveform : Numpy 1D array
        Input waveform.
    P : Integer
        Number of transmitted pulses.
    T_PRI : Float
        Pulse Repetition Interval (PRI) duration.
    fs : Float
        Sampling frequency.

    Raises
    ------
    InputError
        Raised when the waveform duration is longer than the PRI duration.

    Returns
    -------
    pulse_train : Numpy 1D array
        Output pulse train.
    pulses : Numpy 2D array
        Output pulse train in pulse notation. The first axis refers to the pulse
        index, while the second axis refers to the time index.
    """
    
    pulse_width = len(waveform)
    Ng = int(fs*T_PRI) - pulse_width
    if Ng < 0:
        raise InputError("Invalid PRI: waveform duration is longer than PRI duration. ({} vs {})".format(pulse_width/fs,T_PRI))
    
    waveforms = repmat(waveform,P,1)
    guard_samples = zeros((P,Ng))
    pulses = concatenate((waveforms,guard_samples),axis=1)
    pulse_train = pulse2time(pulses,fs)
    return pulse_train,pulses

def pulse_correlation(r,pulse):
    """
    Performs the correlation between the input signal <x> and a pulse <pulse>.

    Parameters
    ----------
    r : Numpy 1D array
        Input signal.
    pulse : Numpy 1D array
        Pulse for correlation.

    Returns
    -------
    pulse_corr : Numpy 1D array
        Correlation output.
    max_idx : Integer
        Index of the maximum correlation (absolute value).
    """
    
    N = len(pulse)
    # pulse_corr = correlate(r,pulse,mode='full')[N-1:len(r)+N-1]
    pulse_corr = scipy_correlate(r,pulse,mode='full')[N-1:len(r)+N-1]
    
    max_idx = argmax(abs(pulse_corr))
    return pulse_corr,max_idx

def compute_CFAR_threshold(P_FA,N,kind="OS",order=0.75,save=True,verbose=False):
    """
    Computes the CFAR threshold to use in order to obtain a given false alarm 
    probability (supposing that samples are affected by an complex normal noise),
    for cell-averaging or ordered-statistic CFAR.

    Parameters
    ----------
    P_FA : Float
        Targeted false alarm probability.
    N : Integer
        Total number of cells.
    kind : String, optional
        CFAR algorithm:
            - "CA" = Cell Averaging;
            - "OS" = Ordered Statistic.
        The default is "OS".
    order : Order, optional
        Order of the OS-CFAR detector. In the detector, the noise variance is 
        computed by taking the int(<order>*<N_cells>)-th greatest value in the 
        test cells around the cell under test, with <N_cells> being the number 
        of test cells. 
        The default is 0.75.
    save : Boolean, optional
        If set to True, the computed CFAR threshold is saved in a file in order
        to avoid to compute it again if the same parameters are forwarded. 
        The default is True.
    verbose : Boolean, optional
        If set to True, informations about the operations done are printed in the 
        console. The default is False.

    Returns
    -------
    Float
        Output CFAR threshold.
    """
    
    def verbose_print(verbose,msg):
        if verbose:
            print(msg)
            
    def find_root(f,eps0=1e-10,n_iter_max=10000,verbose=False):
        a0 = 0
        b0 = 2
        while f(b0) > 0:
            b0 *= 2
        
        a = a0
        b = b0
        eps = eps0
        
        n_iter = 0
        while b - a >= eps:
            if n_iter > n_iter_max:
                eps *= 10
                a = a0
                b = b0                
                n_iter = 0
                # print("WARNING - CFAR coefficient computation : Maximum number of iterations reached ({}). Tolerance modified to {:e}...".format(n_iter_max,eps))
            
            x = (a + b)/2
            if f(x) > 0:
                a = x
            elif f(x) < 0:
                b = x
            else:
                return x
            
            n_iter += 1
        
        return (a + b)/2
    
    verbose_print(verbose,"--- CFAR Threshold computation (P_FA={}, N={}, kind={})---".format(P_FA,N,kind))
    verbose_print(verbose,"Reading saved threshold...")
    try:
        file = open(r"CFAR_coefficients.dat","r")
        entry = file.readline().split()
        while entry:
            if ((str(entry[0]) == kind and kind == "OS" and float(entry[3]) == order) or str(entry[0]) == kind and kind == "CA") \
            and float(entry[1]) == P_FA and float(entry[2]) == N:
                file.close()
                verbose_print(verbose,"Successful !")
                return float(entry[4])
            else:            
                entry = file.readline().split()
        file.close()
        verbose_print(verbose,"No corresponding data found. Starting computation...")
    except:
        verbose_print(verbose,"No save found. Starting computation...")
    
    if kind == "CA":
        out = N * (P_FA**(-1/N) - 1)
    else:
        k = int(round(order * N))
        f = lambda x : beta(x+N-k+1,k) * comb(N,k) * k - P_FA # float(Decimal(beta(x+N-k+1,k)) * Decimal(comb(N,k)) * Decimal(k)) - P_FA
        verbose_print(verbose,"Computation with custom method...")
        out = find_root(f,eps0=1e-10,n_iter_max=10000,verbose=verbose)
        verbose_print(verbose,"Custom method root: {}".format(out))
    verbose_print(verbose,"Computation done.")
    
    if save:
        verbose_print(verbose,"Saving...")
        file = open(r"CFAR_coefficients.dat","a")
        file.write(kind + " " + str(P_FA) + " " + str(N) + " " + str(order) + " " + str(out) + "\n")
        file.close()
        verbose_print(verbose,"Done.")
        
    return out

def precompute_CFAR_thresholds(P_FA,N_values,kind="OS",order=0.75):    
    """
    This function generates a matrix containing the CFAR coefficients associated
    with the different number of cells given in <N_values> using the function
    <compute_CFAR_coefs>, in order to use the function <CFAR_detector_2D>. More
    informations are available on the descriptions of these functions.

    Parameters
    ----------
    P_FA : Float
        Targeted false alarm probability.
    N_values : Numpy 1D array
        Total numbers of cells for which the CFAR coefficients should be computed.
    kind : String, optional
        CFAR algorithm:
            - "CA" = Cell Averaging;
            - "OS" = Ordered Statistic.
        The default is "OS".
    order : Order, optional
        Order of the OS-CFAR detector. In the detector, the noise variance is 
        computed by taking the int(<order>*<N_cells>)-th greatest value in the 
        test cells around the cell under test, with <N_cells> being the number 
        of test cells. 
        The default is 0.75.
    save : Boolean, optional
        If set to True, the computed CFAR threshold is saved in a file in order
        to avoid to compute it again if the same parameters are forwarded. 
        The default is True.
    verbose : Boolean, optional
        If set to True, informations about the operations done are printed in the 
        console. The default is False.

    Returns
    -------
    CFAR_coefs_array : Numpy 2D array
        Array containing the number of cells in the first column and the associated
        CFAR coefficient in the second column.
    """
    
    CFAR_coefs_array = zeros((len(N_values),2))
    for i,N in enumerate(N_values):
        CFAR_coefs_array[i,0] = N
        CFAR_coefs_array[i,1] = compute_CFAR_threshold(P_FA,N,kind,order) 
    return CFAR_coefs_array        

@jit(nopython=True,error_model="numpy")
def CFAR_detector_2D(z,Nl,Nr,Gl,Gr,CFAR_coefs_array,kind="CA",order=0.75,zeropad=array([1,1])):  
    """
    This function is a numba-accelerated version of the function <CFAR_detector>
    for 2D arrays. More informations are available on its description. 
    
    In order to use this function, the CFAR coefficients matrix should be 
    pre-computed using the function <precompute_CFAR_thresholds>. Assuming that 
    the dimensions of <z> are large compared to the window size, the number of 
    cells <N> is bounded following these equations:
        
        Nxm = min(Nl[0],Nr[0])
        Nym = min(Nl[1],Nr[1])
        Ngxm = min(Gl[0] + Gr[0])
        Ngym = min(Gl[1] + Gr[1])
        
        Nmin = Nxm*(Ngym + 1) + Nym*(Ngxm + 1) + Nxm*Nym
        
        Nx = Nl[0] + Nr[0]
        Ny = Nl[1] + Nr[1]
        Ngx = Gl[0] + Gr[0]
        Ngy = Gl[1] + Gr[1]
        
        Nmax = Nx*(Ngy + 1) + Ny*(Ngx + 1) + Nx*Ny

        Nmin <= N <= Nmax

    Parameters
    ----------
    z : Numpy 2D array
        Input complex data array.
    Nl : Numpy 1D array
        Lengths of the window at the left of the CUT for every dimension. The
        size of <Nl> should be equal to 2.
    Nr : Numpy 1D array
        Lengths of the window at the right of the CUT for every dimension. The
        size of <Nr> should be equal to 2.
    Gl : Numpy 1D array
        Lengths of the guard interval at the left of the CUT for every dimension.
        The size of <Gl> should be equal to 2.
    Gr : Numpy 1D array
        Lengths of the guard interval at the right of the CUT for every dimension.
        The size of <Gr> should be equal to 2.
    CFAR_coefs_array : Numpy 2D array
        CFAR coefficients for every possible size of N
    kind : String, optional
        CFAR algorithm:
            - "CA" = Cell Averaging;
            - "OS" = Ordered Statistic.
        The default is "OS".
    order : Order, optional
        Order of the OS-CFAR detector. In the detector, the noise variance is 
        computed by taking the int(<order>*<N_cells>)-th greatest value in the 
        test cells around the cell under test, with <N_cells> being the number 
        of test cells. 
        The default is 0.75.
    zeropad : Numpy 1D array, optional
        Number of cells to insert between the cells of the window to extend the grid when zeropading is applied. 
        The size of <zeropad> should be equal to 2. The default is [1,1].
        
    Raises
    ------
    CFARCoefficientMissingError
        Raised when there is no CFAR coefficient precomputed for the needed value
        of <N>.
        
    Returns
    -------
    thresh_map : Numpy array
        Threshold map of the detector, with the same dimensions as <z>. If the 
        values of |<z>|^2 are higher than the threshold map for some indices, 
        targets are detected at these indices.
    binary_map : Numpy array
        Binary map of the detector, with the same dimensions as <z>. The values
        1 in the binary map corresponds to the indices where targets has been
        detected.
    """
    thresh_map = zeros(z.shape)
    binary_map = zeros(z.shape)
    
    (Nz,Mz) = z.shape
        
    for n in range(Nz):
        for m in range(Mz):
            idx_full_x = arange(n-(Nl[0]+Gl[0])*zeropad[0],n+(Nr[0]+Gr[0])*zeropad[0]+1,zeropad[0])
            idx_guard_x = arange(n-Gl[0]*zeropad[0],n+Gr[0]*zeropad[0]+1,zeropad[0])
            idx_full_y = arange(m-(Nl[1]+Gl[1])*zeropad[1],m+(Nr[1]+Gr[1])*zeropad[1]+1,zeropad[1])
            idx_guard_y = arange(m-Gl[1]*zeropad[1],m+Gr[1]*zeropad[1]+1,zeropad[1])
    
            idx_full_x = idx_full_x[(idx_full_x >= 0)*(idx_full_x < Nz)]    
            idx_guard_x = idx_guard_x[(idx_guard_x >= 0)*(idx_guard_x < Nz)]    
            idx_full_y = idx_full_y[(idx_full_y >= 0)*(idx_full_y < Mz)]    
            idx_guard_y = idx_guard_y[(idx_guard_y >= 0)*(idx_guard_y < Mz)]
            
            N = len(idx_full_x)*len(idx_full_y) - len(idx_guard_x)*len(idx_guard_y)

            cells = zeros((N,))
            idx_cnt = 0
            for idx_x in idx_full_x:
                for idx_y in idx_full_y:
                    if not ((idx_x in idx_guard_x) and (idx_y in idx_guard_y)):
                        cells[idx_cnt] = abs(z[idx_x,idx_y])**2
                        idx_cnt += 1
            
            CFAR_coef = -1
            for N_v,CFAR_coef_v in CFAR_coefs_array:
                if N_v == N:
                    CFAR_coef = CFAR_coef_v
                    break
            
            if CFAR_coef == -1:
                raise CFARCoefficientMissingError
            
            if kind == "CA":
                thresh_map[n,m] = CFAR_coef * sum(cells)/N
            else:
                k = int(round(order * N))
                thresh_map[n,m] = CFAR_coef * sort(cells)[k-1]
            
            binary_map[n,m] = abs(z[n,m])**2 >= thresh_map[n,m]
    
    return thresh_map,binary_map

def CFAR_detector(z,P_FA,Nl,Nr,Gl,Gr,kind="CA",order=0.75,save=True,verbose=False):
    """
    Compute a threshold map and a binary map associated to the input data <z> for 
    a given false alarm probability <P_FA>, based on the Constant False Alarm Rate
    (CFAR) algorithm with square-law detection. An estimation of the noise+interference 
    power is computed based on the samples in a window as shown here (in 1D):
        
          ... |-|-| Nl | Gl |CUT| Gr | Nr |-|-|-|...    
          
    where <Nl> and <Nr> are the lengths of the window at the left and right of 
    the Cell Under Test (CUT), and <Gl> and <Gr> are the lengths of the guard
    intervals at the left and right of the CUT. For example, with <Nl>=3, <Gl>=2,
    <Gr>=3, <Nr>=2, the window looks as follows:
          
          ... |-|-|o|o|o|x|x|CUT|x|x|x|o|o|-|-|-|...

    where "o" represent window samples, "x" represent guard interval samples, "-"
    represents other samples.

    Parameters
    ----------
    z : Numpy array
        Input complex data array.
    P_FA : Float
        Targeted false alarm probability.
    Nl : Numpy 1D array
        Lengths of the window at the left of the CUT for every dimension. The
        size of <Nl> should be equal to the dimension of z.
    Nr : Numpy 1D array
        Lengths of the window at the right of the CUT for every dimension. The
        size of <Nr> should be equal to the dimension of z.
    Gl : Numpy 1D array
        Lengths of the guard interval at the left of the CUT for every dimension.
        The size of <Gl> should be equal to the dimension of z.
    Gr : Numpy 1D array
        Lengths of the guard interval at the right of the CUT for every dimension.
        The size of <Gr> should be equal to the dimension of z.
    kind : String, optional
        CFAR algorithm:
            - "CA" = Cell Averaging;
            - "OS" = Ordered Statistic.
        The default is "OS".
    order : Order, optional
        Order of the OS-CFAR detector. In the detector, the noise variance is 
        computed by taking the int(<order>*<N_cells>)-th greatest value in the 
        test cells around the cell under test, with <N_cells> being the number 
        of test cells. 
        The default is 0.75.
    save : Boolean, optional
        If set to True, the computed CFAR thresholds are saved in a file in order
        to avoid to compute it again if the same parameters are forwarded. 
        The default is True. See "compute_CFAR_threshold" documentation.
    verbose : Boolean, optional
        If set to True, informations about the operations done for the computations
        of the CFAR thresholds are printed in the console. 
        The default is False. See "compute_CFAR_threshold" documentation.

    Returns
    -------
    thresh_map : Numpy array
        Threshold map of the detector, with the same dimensions as <z>. If the 
        values of |<z>|^2 are higher than the threshold map for some indices, 
        targets are detected at these indices.
    binary_map : Numpy array
        Binary map of the detector, with the same dimensions as <z>. The values
        1 in the binary map corresponds to the indices where targets has been
        detected.
    """
    
    thresh_map = zeros(z.shape)
    binary_map = zeros(z.shape)
    for indices,value in ndenumerate(z):
        cells_indices = []
        guard_indices = []
        for dim_idx,(nl,nr,gl,gr) in enumerate(zip(Nl,Nr,Gl,Gr)):
            M = z.shape[dim_idx]
            val_idx = indices[dim_idx]
            
            full_indices_d = arange(val_idx-gl-nl,val_idx+gr+nr+1)
            full_indices_d = full_indices_d[(full_indices_d >= 0) * (full_indices_d < M)]
            cut_indices_d = arange(val_idx-gl,val_idx+gr+1)
            cut_indices_d = cut_indices_d[(cut_indices_d >= 0) * (cut_indices_d < M)]
            
            cut_indices_d -= min(full_indices_d)

            cells_indices.append(full_indices_d.astype(int))        
            guard_indices.append(cut_indices_d.astype(int))
        
        full_cube = abs(z[ix_(*cells_indices)])**2
        cells_cube = delete(full_cube,ravel_multi_index(ix_(*guard_indices),full_cube.shape))
        
        N = cells_cube.size
        CFAR_coef = compute_CFAR_threshold(P_FA,N,kind,order,save,verbose)
        
        if kind == "CA":
            thresh_map[indices] = CFAR_coef * sum(cells_cube)/N
        else:       
            k = int(round(order * N))
            thresh_map[indices] = CFAR_coef * sort(cells_cube)[k-1]
            
        binary_map[indices] = abs(value)**2 >= thresh_map[indices]
        
    return thresh_map,binary_map

def binary_map_processing(z,binary_map):
    """
    Process a binary map <binary_map> by filling each area, except the element
    at the index corresponding to the maximum value of <z> in the area. 
    
    For example, if the couple <z> / <binary_map> in 2D is given by
    
                            <z>      <binary_map>
                        1 2 3 4 5  |  0 1 1 0 0
                        6 7 8 9 1  |  0 1 1 0 0 
                        2 3 4 5 6  |  0 0 0 1 1
                        7 8 9 1 2  |  0 0 1 1 1
                        3 4 5 6 7  |  0 0 1 1 1 
                        
    The output is 
                     
                              0 0 0 0 0
                              0 0 1 0 0
                              0 0 0 0 0
                              0 0 1 0 0
                              0 0 0 0 0

    Parameters
    ----------
    z : Numpy array
        Value map of the same dimension as <binary_map>. 
    binary_map : Numpy array
        Binary map of the same dimension as <z>.

    Returns
    -------
    Numpy array
        Output binary map, of the same dimension as <z> and <binary_map>.
    """
    
    value_map = zeros(z.shape)
    value_map[binary_map.astype(bool)] = abs(z[binary_map.astype(bool)])**2
    
    max_list = []
    while value_map.any():
        max_idx = unravel_index(argmax(value_map),value_map.shape)
        max_list.append(max_idx)
        
        empty_map = zeros(binary_map.shape)
        empty_map[max_idx] = 1
        mask = binary_dilation(empty_map, structure=None, iterations=0, mask=binary_map)
        value_map[mask] = 0
        
    empty_map = zeros(binary_map.shape)
    for idx in max_list:
        empty_map[idx] = 1
    return empty_map

def targets_extraction(z,targets_map,dx,centered):
    """
    Extracts the targets parameters from a targets map <targets_map>, and based
    on the different parameters resolutions.

    Parameters
    ----------
    z : Numpy array
        Input complex data array.
    targets_map : Numpy array
        Binary map of the same dimension as <z>.
    dx : List
        Parameters resolutions. Set to Inf the resolution of irrelevant parameters
        in order to not take them into account.
    centered : List of booleans
        List of booleans of the same dimension as <dx>. For each element set to 
        True, the corresponding parameter range is centered around 0.
    Returns
    -------
    List of list
        Estimated targets parameters. For example, with a SISO channel with two
        targets, the list is
        
                [[alpha_0,fD_0,tau_0], [alpha_1,fD_1,tau_1]]
                
        with <alpha_i>, <fD_i> and <tau_i> are respectively the complex attenuations,
        Doppler frequencies and delays of the two paths.
    """
    targets_params_hat = []
    for indices in argwhere(targets_map):
        idx = ravel_multi_index(indices,z.shape)
        amplitude = z.item(idx)
        parameters = (indices - array(centered)*floor(array(targets_map.shape)/2)) * dx
        target_params = [amplitude] + parameters.tolist()
        targets_params_hat.append(target_params)
    return sorted(targets_params_hat, key=lambda x: abs(x[0]), reverse=True)
        
    

def compare_targets(targets_params,targets_params_hat,dx):
    """
    Compares estimated targets parameters <targets_params> with the true parameters 
    <targets_params_hat> to compute the number of detections and false alarms. 

    Parameters
    ----------
    targets_params : List of lists
        Targets parameters. Each element is a list containing the parameters
        associated to a given target. For example, with a SISO channel with two
        targets, the list is
        
                [[alpha_0,fD_0,tau_0], [alpha_1,fD_1,tau_1]]
                
        with <alpha_i>, <fD_i> and <tau_i> are respectively the complex attenuations,
        Doppler frequencies and delays of the two paths.
        
    targets_params_hat : List of lists
        Estimated targets parameters, with the same structure as <targets_params>.
        Each element is a list containing the parameters associated to a given target.
    dx : List
        Parameters resolutions. Set to Inf the resolution of irrelevant parameters
        in order to not take them into account.

    Returns
    -------
    detections : List
        List containing a 1 for every detected target and a 0 otherwise. The 
        indices correspond to the elements of <targets_params>.
    false_alarms : List
        List containing a 1 for every false alarm, and 0 otherwise. The indices
        correspond to the elements of <targets_params_hat>
    """
    
    detections = zeros((len(targets_params),))
    false_alarms = ones((len(targets_params_hat),))
    
    for i,target_params in enumerate(targets_params):
        for j,target_params_hat in enumerate(targets_params_hat):
            if (abs(array(target_params)-array(target_params_hat))[1:] <= array(dx)/2).all():
                detections[i] = 1
                false_alarms[j] = 0
    
    return detections,false_alarms