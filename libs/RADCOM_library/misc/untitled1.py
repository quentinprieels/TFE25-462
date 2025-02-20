# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:39:00 2021

@author: fdesaintmoul
"""

from numpy import zeros,ones,arange,sum
from numpy import sqrt,log
from numpy.random import normal
from matplotlib.pyplot import figure,plot,title,close
from radcomlib.radar_toolbox import CFAR_detector

from scipy.stats import ncx2

close('all')

## CFAR detector parameters
P_FA = 1e-4 # targeted false alarm probability
kind = "OS" # kind of detector, "CA" or "OS"
order = 0.75 # Order of the OS-CFAR 
# CFAR geometry : |------| Nl | Gl | CUT | Gr | Nr |------|, specified for each dimension
Nl = 100 * ones((1,)) # test cells at the left of the CUT
Nr = 100 * ones((1,)) # test cells at the right of the CUT
Gl = 0 * ones((1,)) # guard cells at the left of the CUT
Gr = 0 * ones((1,)) # guard cells at the right of the CUT

N = 1000

n_iter = 1000


SNR_dB_vec = arange(0,21,1)
SNR_vec = 10**(SNR_dB_vec/10)

P_D_MC = zeros((len(SNR_dB_vec),))
P_FA_MC = zeros((len(SNR_dB_vec),))

for SNR_idx,SNR in enumerate(SNR_vec):
    var_noise = 1/SNR
    for i in range(n_iter):
        z = normal(scale=sqrt(var_noise/2),size=(N,)) + 1j*normal(scale=sqrt(var_noise/2),size=(N,)) 
        z[int(len(z)/2)] += 1
        
        thresh_map, binary_map = CFAR_detector(z,P_FA,Nl,Nr,Gl,Gr,kind=kind,
                                               order=order,save=True,force_manual_solver=True,verbose=False)
    
        if binary_map[int(len(z)/2)] == 1:
            P_D_MC[SNR_idx] += 1
        
        P_FA_MC[SNR_idx] += sum(binary_map) - binary_map[int(len(z)/2)]
    
    P_D_MC[SNR_idx] /= n_iter
    P_FA_MC[SNR_idx] /= n_iter * (N - 1)

P_D_th = 1 - ncx2.cdf(-2*log(P_FA),2,2*SNR_vec)

figure()
title("Detection probability")
plot(SNR_dB_vec,P_D_th,'k')
plot(SNR_dB_vec,P_D_MC)

print(P_FA_MC)