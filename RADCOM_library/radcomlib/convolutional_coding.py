"""
"""

__all__ = ["plot_trellis",
           "poly2trellis",
           "conv_encoder",
           "viterbi_decoder",
           "BCJR_decoder",
           "interleaver",
           "deinterleaver",
           "turboencoder",
           "turbodecoder"]


from numpy import zeros, array, reshape, ones, abs, exp, dot, sum, log, sign
from numpy import argsort, arange, real, imag
from numpy import int32, Inf, pi, float64, complex64

from matplotlib.pyplot import subplots

from numba import jit

def plot_trellis(R1,R0,out_R1,out_R0):    
    fig,ax = subplots(1,1)
    
    nb_states,nb_outputs = out_R1.shape
    
    for i in range(len(R1)):

        ax.plot([0,1],[nb_states-1-i,nb_states-1-R1[i]],'b--') 
        ax.text(-0.1*nb_outputs,nb_states-1-i,s="".join(str(e) for e in out_R1[i,:]),
                  fontsize=14,horizontalalignment='center',verticalalignment='center',color='b')
        ax.plot([0,1],[nb_states-1-i,nb_states-1-R0[i]],'r-')
        ax.text(-0.1*nb_outputs*nb_outputs,nb_states-1-i,s="".join(str(e) for e in out_R0[i,:]),
                  fontsize=14,horizontalalignment='center',verticalalignment='center',color='r')
        

        ax.plot(0,i,'k.',markersize=15)
        ax.plot(1,i,'k.',markersize=15)
    ax.axis('off')
    
    return fig,ax

@jit(nopython=True,error_model="numpy")
def number2binary(x0,length):
    binary_array = zeros((length,))
    
    x = x0
    i = 0
    
    while x > 1 and i < length:
        binary_array[i] = x % 2
        x = int(x / 2)
        i = i + 1
    
    if x > 0 and i < length:
        binary_array[i] = 1
    
    return binary_array[::-1]

@jit(nopython=True,error_model="numpy")
def binary2number(x):
    out = 0
    for i in x: 
        out = out*2 + i 
    return out

@jit(nopython=True,error_model="numpy")
def poly2trellis(gn,gd):
    M = max(len(gn),len(gd)) - 1
    nb_states = 2**M
    
    alpha = zeros((M+1,))
    beta = zeros((M+1,))
    
    alpha[:len(gn)] = gn
    beta[:len(gd)] = gd

    R1 = zeros((nb_states,),dtype=int32)
    R0 = zeros((nb_states,),dtype=int32)
    
    out_R1 = zeros((nb_states,2),dtype=int32)
    out_R0 = zeros((nb_states,2),dtype=int32)
    
    out_R1[:,0] = 1
    
    for i in range(nb_states):
        states = zeros((M+1,))
        states[:M] = number2binary(i,M)[::-1]
        
        y_1 = (alpha[0] + states[0]) % 2
        y_0 = states[0]
        
        new_states_1 = (alpha[1:] + beta[1:]*y_1 + states[1:]) % 2
        new_states_0 = (beta[1:]*y_0 + states[1:]) % 2
        
        R1[i] = binary2number(new_states_1[::-1])
        R0[i] = binary2number(new_states_0[::-1])
        
        out_R1[i,1] = int(y_1)
        out_R0[i,1] = int(y_0)
    
    return R1,R0,out_R1,out_R0

@jit(nopython=True,error_model="numpy") # to uncomment for Numba compilation
def conv_encoder(u,R1,R0,out_R1,out_R0,len_b):
    """
    This function encodes the bit stream <u> with a convolutional encoder 
    whose trellis is described by <R1>, <R0>, <out_R1> and <out_R0>, producing 
    a bit stream <c>. The encoding process works on blocks of <len_b> bits, 
    each block being encoded seprately.
    
    Parameters
    ----------
    u : 1D numpy array
        Input sequence.
    R1 : 1D numpy array
        Trellis decomposition - transitions if 1.
    R0 : 1D numpy array
        Trellis decomposition - transitions of 0.
    out_R1 : 2D numpy array
        Trellis decomposition - output bits corresponding to transitions with 1.
    out_R0 : 2D numpy array
        Trellis decomposition - output bits corresponding to transitions with 1.
    len_b : int
        Length of each block. We assume that N_b = len(u)/len_b is an integer!

    Returns
    -------
    u : 1D numpy array
        Systematic part of the coded sequence (i.e. the input bit stream).
    c : 1D numpy array
        Non-systematic part of the coded sequence.
    """
    
    ## Block decomposition for the non-systematic output
    N_b = int(len(u)/len_b)
    
    u_b = reshape(u,(N_b,len_b))
    c_b = zeros(u_b.shape,dtype=int32)
    
    # block convolutional encoder (non-systematic output)
    for i in range(0,N_b): 
        # input of block i
        u_i = u_b[i,:]
        # non systematic output of block i (TO FILL!)
        c_i = c_b[i,:]
        
        #IF SOLUCE
        state = 0
        for j in range(0,len_b):
            if u_i[j] == 1: 
                c_i[j] = out_R1[state,1]
                state = R1[state]
            else:
                c_i[j] = out_R0[state,1]
                state = R0[state]
        #ELSE
        #TO COMPLETE
        #END
                              
    # non-systematic output
    c = reshape(c_b,u.shape)
    
    return u,c          

@jit(nopython=True,error_model="numpy")
def viterbi_decoder(R1,R0,symb_R1,symb_R0,len_b,x_tilde):
    def dist(a,b):
        return abs(a-b)**2
    
    N_b = int(len(x_tilde)/len_b)
    
    x_tilde_b = reshape(x_tilde,(N_b,len_b))
    u_hat_b = zeros(x_tilde_b.shape,dtype=int32)
    
    nb_states = len(R1)

    for i in range(N_b):           
        x_tilde_i  = x_tilde_b[i,:]
        u_hat_i = u_hat_b[i,:]
        
        bits = zeros((nb_states,len_b))
        weights = Inf*ones((nb_states,))
        weights[0] = 0
        
        new_states = zeros((2,nb_states))
        new_weights = zeros((2,nb_states))
        new_bits = zeros((2,nb_states,len_b))  
        
        for j in range(len_b):
            for k in range(nb_states):
                new_states[1,k] = R1[k]
                new_states[0,k] = R0[k]
                new_weights[1,k] = weights[k] + dist(x_tilde_i[j],symb_R1[k])
                new_weights[0,k] = weights[k] + dist(x_tilde_i[j],symb_R0[k])       
                new_bits[1,k,:] = bits[k,:]
                new_bits[0,k,:] = bits[k,:]
                new_bits[1,k,j] = 1
                
            for k in range(nb_states):
                idx_0_filled = False
                for l in range(nb_states):
                    if new_states[0,l] == k:
                        if idx_0_filled:
                            idx_10 = 0
                            idx_11 = l
                        else:
                            idx_00 = 0
                            idx_01 = l 
                            idx_0_filled = True
                            
                    if new_states[1,l] == k:
                        if idx_0_filled:
                            idx_10 = 1
                            idx_11 = l
                        else:
                            idx_00 = 1
                            idx_01 = l 
                            idx_0_filled = True
                
                if new_weights[idx_00,idx_01] <= new_weights[idx_10,idx_11]:
                    weights[k] = new_weights[idx_00,idx_01]
                    bits[k,:] = new_bits[idx_00,idx_01,:]
                else:
                    weights[k] = new_weights[idx_10,idx_11]
                    bits[k,:] = new_bits[idx_10,idx_11,:]

        final_weight = Inf
        for k in range(nb_states):
            if weights[k] < final_weight:
                final_weight = weights[k]
                u_hat_i[:] = bits[k,:]
    
    u_hat = reshape(u_hat_b,(u_hat_b.size,))
    return u_hat 

@jit(nopython=True,error_model="numpy") # to uncomment for Numba compilation
def compute_gamma(R1,R0,symb_R1,symb_R0,x_tilde,var_noise,L,gamma):
    """
    This function fills in the <gamma> array, based on the received noisy symbol 
    sequence <x_tilde>, the trellis parametrized by <R1>, <R0>, <symb_R1> and 
    <symb_R0>, the noise variance <var_noise>, the a priori log-likelihood ratio <L>. 

    Parameters
    ----------
    R1 : 1D numpy array
        Trellis decomposition - transitions if 1.
    R0 : 1D numpy array
        Trellis decomposition - transitions of 0.
    symb_R1 : 1D numpy array
        Trellis decomposition - symbols corresponding to transitions with 1.
    symb_R0 : 1D numpy array
        Trellis decomposition - symbols corresponding to transitions with 0.
    x_tilde : 1D numpy array
        Received noisy symbol sequence.
    var_noise : float
        Noise variance of the channel.
    L : 1D numpy array
        A priori information (log-likelihood ratio).
    gamma : 3D numpy array
        Gamma matrix to fill.
        
    Returns
    -------
    No output, the <gamma> argument is a reference to the array, hence the array
    is directly filled.
    """
    
    # number of states in the trellis
    nb_states = len(R1)
    
    len_x = len(x_tilde)
    for k in range(0,len_x):
        #IF SOLUCE
        for sp in range(nb_states):
            for s in range(nb_states):
                if R1[sp] == s:
                    if exp(L[k]) == Inf:
                        gamma[sp,s,k] = 1/(pi*var_noise)*exp(-1/var_noise*abs(x_tilde[k]-symb_R1[sp])**2) 
                    else:
                        gamma[sp,s,k] = 1/(pi*var_noise)*exp(-1/var_noise*abs(x_tilde[k]-symb_R1[sp])**2)  * exp(L[k])/(1.0 + exp(L[k]))
                elif R0[sp] == s:
                    if exp(-L[k]) == Inf:
                        gamma[sp,s,k] = 1/(pi*var_noise)*exp(-1/var_noise*abs(x_tilde[k]-symb_R0[sp])**2) 
                    else:
                        gamma[sp,s,k] = 1/(pi*var_noise)*exp(-1/var_noise*abs(x_tilde[k]-symb_R0[sp])**2)  * exp(-L[k])/(1.0 + exp(-L[k]))
                else:
                    gamma[sp,s,k] = 0.0
        #ELSE
        #pass
        ## TO COMPLETE
        #END

@jit(nopython=True,error_model="numpy")     
def gamma_clean(gamma,R1,R0,len_x):
    ## Clean for alpha => start to end   

    nb_states = len(R0)
    
    possible_states = zeros((nb_states,))
    next_possible_states = zeros((nb_states,))
    possible_states[0] = 1
    next_possible_states[0] = 1
    
    for i in range(0,len_x):
        for j in range(0,nb_states):
            for k in range(0,nb_states):
                if possible_states[j] and (R1[j] == k or R0[j] == k):
                    next_possible_states[k] = 1 
                else:
                    gamma[j,k,i] = 0.0 # kill impossible state transitions
                    
        if  sum(next_possible_states) == nb_states:
            break
        else: 
            possible_states[:] = next_possible_states   

@jit(nopython=True,error_model="numpy") # to uncomment for Numba compilation
def compute_alpha(R1,R0,len_x,gamma,alpha):
    """
    This function fills in the <alpha> array, based on the trellis parametrized by 
    <R1> and <R0>, and the <gamma> values. Moreover, the initial conditions are 
    given as an input in alpha[:,0].

    Parameters
    ----------
    R1 : 1D numpy array
        Trellis decomposition - transitions if 1.
    R0 : 1D numpy array
        Trellis decomposition - transitions of 0.
    len_x : int
        Number of received symbols.
    gamma : 3D numpy array
        Gamma matrix.
    alpha : 2D numpy array
        Alpha matrix to fill.

    Returns
    -------
    No output, the <alpha> argument is a reference to the array, hence the array
    is directly filled.
    """    
    
    # number of states in the trellis
    nb_states = len(R1)
    
    for k in range(1,len_x+1):
        #IF SOLUCE
        for s in range(0,nb_states):
            alpha[s,k] = dot(gamma[:,s,k-1],alpha[:,k-1])
        
        alpha[:,k] = alpha[:,k]/sum(alpha[:,k])
        #ELSE
        #pass
        ## TO COMPLETE
        #END

@jit(nopython=True,error_model="numpy") # to uncomment for Numba compilation
def compute_beta(R1,R0,len_x,gamma,beta):
    """
    This function fills in the <beta> array, based on the trellis parametrized by 
    <R1> and <R0>, and the <gamma> values. Moreover, the initial conditions are 
    given as an input in beta[:,len_x].

    Parameters
    ----------
    R1 : 1D numpy array
        Trellis decomposition - transitions if 1.
    R0 : 1D numpy array
        Trellis decomposition - transitions of 0.
    len_x : int
        Number of received symbols.
    gamma : 3D numpy array
        Gamma matrix.
    beta : 2D numpy array
        Beta matrix to fill.

    Returns
    -------
    No output, the <beta> argument is a reference to the array, hence the array
    is directly filled.
    """
    
    # number of states in the trellis
    nb_states = len(R1)
    
    for k in range(len_x-1,-1,-1):
        #IF SOLUCE
        for s in range(0,nb_states):
            beta[s,k] = dot(gamma[s,:,k],beta[:,k+1])
        
        beta[:,k] = beta[:,k]/sum(beta[:,k])
        #ELSE
        #pass
        ## TO COMPLETE
        #END

@jit(nopython=True,error_model="numpy") # to uncomment for Numba compilation
def BCJR_decoder(R1,R0,symb_R1,symb_R0,len_b,var_noise,x_tilde,L):
    """
    This function decodes the symbol stream <x_tilde> with has been encoded 
    by a convolutional encoder whose trellis is described by <R1>, <R0>, <out_R1> and 
    <out_R0>. The encoding process has worked on blocks of <len_b> bits, each block 
    being encoded seprately. The decoding process is furthermore based on the noise 
    variance <var_noise> and a bit-per-bit a priori log-likelihood ration <L>.
    
    In the below function, the block separation has already been handled. You
    need to fill in the numpy array <L_c_i> which is the conditionnal log-likelihood
    ratio corresponding to the symbol block <x_tilde_i> and the a-priori log-likelihood
    ratio L_i = L_b[i,:]. You also need to impose the right initial conditions in 
    the arrays <beta> and <alpha>.
    
    Parameters
    ----------
    R1 : 1D numpy array
        Trellis decomposition - transitions if 1.
    R0 : 1D numpy array
        Trellis decomposition - transitions of 0.
    symb_R1 : 1D numpy array
        Trellis decomposition - symbols corresponding to transitions with 1.
    symb_R0 : 1D numpy array
        Trellis decomposition - symbols corresponding to transitions with 0.
    len_b : int
        Length of each block. We assume that N_b = len(u)/len_b is an integer!
    var_noise : float
        Noise variance of the channel.
    x_tilde : 1D numpy array
        Input received noisy symbols.
    L : 1D numpy array
        A priori information (log-likelihood ratio).

    Returns
    -------
    u_hat : 1D numpy array
        Estimated bit sequence.
    L_c : 1D numpy array
        Conditionnal log-likelihood ratio.
    """
   
    # number of states in the trellis
    nb_states = len(R1)
    
    ## Block decomposition
    N_b = int(len(x_tilde)/len_b)
    
    x_tilde_b  = reshape(x_tilde,(N_b,len_b))
    L_b = reshape(L,(N_b,len_b))
    L_c_b = zeros(x_tilde_b.shape)
    
    ## Block BCJR decoder
    for i in range(0,N_b):                 
        x_tilde_i  = x_tilde_b[i,:]
        L_i = L_b[i,:]
        L_c_i = L_c_b[i,:]

        ## alpha/beta/gamma matrices
        gamma = zeros((nb_states,nb_states,len_b))
        alpha = zeros((nb_states,len_b+1))
        beta  = zeros((nb_states,len_b+1))    

        ## Initial / final conditions
        #IF SOLUCE
        alpha[0,0] = 1.0 
        beta[:,-1] = 1.0/nb_states
        #ELSE
        #alpha[:,0] = 0.0 # TO COMPLETE
        #beta[:,-1] = 0.0 # TO COMPLETE
        #END

        # compute gamma
        compute_gamma(R1,R0,symb_R1,symb_R0,x_tilde_i,var_noise,L_i,gamma)
        gamma_clean(gamma,R1,R0,len_b)
        # compute alpha
        compute_alpha(R1,R0,len_b,gamma,alpha)    
        # compute beta
        compute_beta(R1,R0,len_b,gamma,beta)

        #### conditional log-likelihood ratio
        for k in range(1,len_b+1):
            
            #IF SOLUCE
            num = 0.0 # numerator of the conditional log-likelihood function (in the log)
            den = 0.0 # denominator of the conditional log-likelihood function (in the log)
            for j in range(nb_states):
                num = num + beta[R1[j],k]*gamma[j,R1[j],k-1]*alpha[j,k-1]
            for j in range(nb_states):
                den = den + beta[R0[j],k]*gamma[j,R0[j],k-1]*alpha[j,k-1]
            #ELSE
            # TO COMPLETE
            #num = 0.0 # numerator of the conditional log-likelihood function (in the log)
            #den = 0.0 # denominator of the conditional log-likelihood function (in the log)
            #END
            
            L_c_i[k-1] = log(num/den)

    L_c = reshape(L_c_b,x_tilde.shape)
    u_hat = array([int(i) for i in (sign(L_c)+1)/2])
    
    return u_hat,L_c

@jit(nopython=True,error_model="numpy")
def interleaver(x,pattern):
    Nb = int(len(x)/len(pattern))
    x_matrix = reshape(x,(Nb,len(pattern)))
    y_matrix = x_matrix[:,pattern-1]
    y = reshape(y_matrix,(len(x),))
    return y

@jit(nopython=True,error_model="numpy")
def deinterleaver(x,pattern):
    return interleaver(x,argsort(pattern)+1)

@jit(nopython=True,error_model="numpy")
def puncturer(u1,u2):
    len_u1 = len(u1)
    u = zeros(u1.shape,dtype=int32)
    u[arange(0,len_u1,2)] = u1[arange(0,len_u1,2)]
    u[arange(1,len_u1,2)] = u2[arange(1,len_u1,2)]
    return u

@jit(nopython=True,error_model="numpy")
def split_symbols(x_tilde,pattern):
    x_tilde_I = real(x_tilde).astype(float64)
    x_tilde_Q = imag(x_tilde).astype(float64)
    
    x_tilde_I_int = interleaver(x_tilde_I,pattern)
    
    x_tilde_1 = x_tilde_I.astype(complex64)
    x_tilde_2 = x_tilde_I_int.astype(complex64)
    x_tilde_1[arange(0,len(x_tilde),2)] = x_tilde_1[arange(0,len(x_tilde),2)] + 1j * x_tilde_Q[arange(0,len(x_tilde),2)]
    x_tilde_2[arange(1,len(x_tilde),2)] = x_tilde_2[arange(1,len(x_tilde),2)] + 1j * x_tilde_Q[arange(1,len(x_tilde),2)]
    return x_tilde_1,x_tilde_2

@jit(nopython=True,error_model="numpy") # to uncomment for Numba compilation
def turboencoder(R1,R0,out_R1,out_R0,pattern,len_b,u):       
    """
    This function encodes the bit stream <u> with a turboencoder whose constituent
    codes have a treillis described by <R1>, <R0>, <out_R1> and <out_R0>, and whose 
    interleaving pattern is given by <pattern>. The encoding process works on blocks 
    of <len_b> bits, each block being encoded seprately.

    Parameters
    ----------
    R1 : 1D numpy array
        Trellis decomposition - transitions if 1.
    R0 : 1D numpy array
        Trellis decomposition - transitions of 0.
    out_R1 : 2D numpy array
        Trellis decomposition - output bits corresponding to transitions with 1.
    out_R0 : 2D numpy array
        Trellis decomposition - output bits corresponding to transitions with 0.
    pattern : 1D numpy array
        Interleaver pattern.
    len_b : int
        Length of each block.
    u : 1D numpy array
        Input sequence.

    Returns
    -------
    u_enc : 1D numpy array
        Turbocoded bit sequence. There is a systematic part u_s, and a non-systematic one u_c.
    """
    
    #IF SOLUCE
    u_int = interleaver(u,pattern)

    (u_s,u_c1) = conv_encoder(u,R1,R0,out_R1,out_R0,len_b)
    (_,u_c2) = conv_encoder(u_int,R1,R0,out_R1,out_R0,len_b)

    u_c = puncturer(u_c1,u_c2)
    #ELSE
    #u_s = # TO COMPLETE (systematic output)
    #u_c = # TO COMPLETE (non systematic output)
    #END
    
    u_enc = zeros((2*len(u_s),),dtype=int32)
    u_enc[::2] = u_s
    u_enc[1::2] = u_c
    
    return u_enc

@jit(nopython=True,error_model="numpy") # to uncomment for Numba compilation
def turbodecoder(R1,R0,symb_R1,symb_R0,pattern,len_b,var_noise,xi,n_iter,x_tilde):    
    """
    This function decodes the symbol stream <x_tilde> with has been encoded 
    by a turboencoder whose treillis is described by <R1>, <R0>, <out_R1> and 
    <out_R0>, and whose interleaving pattern is <pattern>. The encoding process 
    has worked on blocks of <len_b> bits, each block being encoded seprately. 
    The decoding process is furthermore based on the noise variance <var_noise>. 
    The decoder performs <n_iter> iterations before stopping, but returns the bit 
    sequence and conditionnal log-likelihood ratio of all iterations.

    Parameters
    ----------
    R1 : 1D numpy array
        Trellis decomposition - transitions if 1.
    R0 : 1D numpy array
        Trellis decomposition - transitions if 0.
    symb_R1 : 1D numpy array
        Trellis decomposition - output bits corresponding to transitions with 1.
    symb_R0 : 1D numpy array
        Trellis decomposition - output bits corresponding to transitions with 0.
    pattern : 1D numpy array
        Interleaver pattern.
    len_b : int
        Length of each block. We assume that N_b = len(x)/len_b is an integer!
    var_noise : float
        Noise variance of the channel.
    n_iter : int
        Number of decoding iterations (1 iteration = D1 + D2).
    x_tilde : 1D numpy array
        Input symbols.

    Returns
    -------
    u_hat : 2D numpy array
        Estimated bit sequence for each iteration. u_hat[i,:] is the bit sequence 
        of the i-th iteration.
    L_c_2 : 2D numpy array
        Conditional log-likelihood ratio. L_c_2[i,:] is the log-likelihood sequence 
        of the i-th iteration.  
    """
    
    #IF SOLUCE
    (x_tilde_1,x_tilde_2) = split_symbols(x_tilde,pattern)
    #ELSE
    #END
    
    L_c_2 = zeros((n_iter,len(x_tilde))) #log-likelihood ratio of all bits for each iteration
    u_hat = zeros((n_iter,len(x_tilde))) #decoded bit sequence for each iteration
    
    L2 = zeros(len(x_tilde)) # a priori log likelohood ratio, at the first iteration, =0. Afterwards, output of the second decoder
    for i in range(0,n_iter):
        
        #IF SOLUCE
        (_,L_c_1) = BCJR_decoder(R1,R0,symb_R1,symb_R0,len_b,var_noise,x_tilde_1,L2)
        L1 = L_c_1 - L2 - xi*real(x_tilde_1)
        L1[L1 != L1] = L_c_1[L1 != L1]
        
        L1_int = interleaver(L1,pattern)
        
        (_,L_c_2_int) = BCJR_decoder(R1,R0,symb_R1,symb_R0,len_b,var_noise,x_tilde_2,L1_int)
        L2_int = L_c_2_int - L1_int - xi*real(x_tilde_2)
        L2_int[L2_int != L2_int] = L_c_2_int[L2_int != L2_int]
        
        L2 = deinterleaver(L2_int,pattern)    
        L_c_2[i,:] = deinterleaver(L_c_2_int,pattern)
        #ELSE
        # TO COMPLETE 
        #L_c_2[i,:] =  # TO COMPLETE
        #END
        
        u_hat[i,:] = [int(i) for i in (sign(L_c_2[i,:])+1)/2]
        
    return u_hat,L_c_2