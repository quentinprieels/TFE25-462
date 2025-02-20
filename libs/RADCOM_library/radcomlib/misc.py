"""
"""
from radcomlib.ofdm import SISO_OFDM_DFRC_TX

from radcomlib.fmcw import SISO_PC_FMCW_DFRC_TX

from radcomlib.channel import SISO_channel
from radcomlib.channel import add_signals

from numpy import arange, zeros, floor, real, imag
from scipy.interpolate import interp2d

__all__ = ["signal_reconstruction", 
           "signal_cancellation"]

def signal_reconstruction(symbols, preamble, channel_parameters, M, B, T, T_PRI, waveform="fmcw", L_CP=0, up=True, centered=True, periodic_preamble=True):
    _,Lc = symbols.shape 
    fs_rx = M*Lc/T
    fs_tx = fs_rx

    if waveform == "fmcw":
        tx_sig = SISO_PC_FMCW_DFRC_TX(symbols, preamble, M, B, T, T_PRI, up, centered)
    elif waveform == "ofdm":
        tx_sig = SISO_OFDM_DFRC_TX(symbols, preamble, L_CP, M, periodic_preamble)

    rx_sig = SISO_channel(tx_sig, channel_parameters, 0, fs_tx, fs_rx)
    return rx_sig

def signal_cancellation(sigA, sigB):
    return add_signals(sigA, -sigB)

def detection_map_fusion_2D(p_vec,n_vec,z,weights,dx,centered):
    out = zeros((len(p_vec),len(n_vec)),dtype=z[0].dtype)
    for z_i, weights_i, dx_i, centered_i in zip(z,weights,dx,centered):
        P,N = z_i.shape
        p_axis = (arange(P) - floor(P/2)*centered_i[0])*dx_i[0]
        n_axis = (arange(N) - floor(N/2)*centered_i[1])*dx_i[1]
        fun_z_i_real = interp2d(p_axis,n_axis,real(z_i.T),kind="linear")
        fun_z_i_imag = interp2d(p_axis,n_axis,imag(z_i.T),kind="linear")
        
        out += weights_i * (fun_z_i_real(p_vec,n_vec).T + 1j * fun_z_i_imag(p_vec,n_vec).T)
        
    return out