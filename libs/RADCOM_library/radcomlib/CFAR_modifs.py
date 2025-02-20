from numpy import zeros, arange, min, abs, sum, delete, ix_,ndenumerate,ravel_multi_index, sort, ones
from numba import jit

from time import time

from radcomlib.radar_toolbox import CFAR_detector

import matplotlib.pyplot as plt

import numpy as np
min = np.min
abs = np.abs
max = np.max
round = np.round
sum = np.sum

@jit(nopython=True,error_model="numpy")
def CFAR_detector_2D(z,Nl,Nr,Gl,Gr,CFAR_coefs_array,kind="CA",order=0.75):    
    thresh_map = zeros(z.shape)
    binary_map = zeros(z.shape)
    
    if kind == "CA":
        CFAR_type = 0
    else: 
        CFAR_type = 1
    
    (Nz,Mz) = z.shape
        
    for n in range(Nz):
        for m in range(Mz):
            idx_full_x = np.arange(n-Nl[0]-Gl[0],n+Nr[0]+Gr[0]+1)
            idx_guard_x = np.arange(n-Gl[0],n+Gr[0]+1)
            idx_full_y = np.arange(m-Nl[1]-Gl[1],m+Nr[1]+Gr[1]+1)
            idx_guard_y = np.arange(m-Gl[1],m+Gr[1]+1)
    
            idx_full_x = idx_full_x[(idx_full_x >= 0)*(idx_full_x < Nz)]    
            idx_guard_x = idx_guard_x[(idx_guard_x >= 0)*(idx_guard_x < Nz)]    
            idx_full_y = idx_full_y[(idx_full_y >= 0)*(idx_full_y < Mz)]    
            idx_guard_y = idx_guard_y[(idx_guard_y >= 0)*(idx_guard_y < Mz)]
            
            N = len(idx_full_x)*len(idx_full_y) - len(idx_guard_x)*len(idx_guard_y)

            cells = np.zeros((N,))
            idx_cnt = 0
            for idx_x in idx_full_x:
                for idx_y in idx_full_y:
                    if not ((idx_x in idx_guard_x) and (idx_y in idx_guard_y)):
                        cells[idx_cnt] = np.abs(z[idx_x,idx_y])**2
                        idx_cnt += 1
            
            CFAR_coef = 1
            for i,N_v in enumerate(CFAR_coefs_array[CFAR_type,0,:]):
                if N_v == N:
                    CFAR_coef = CFAR_coefs_array[CFAR_type,1,i]
                    break
            
            if CFAR_type == 0:
                thresh_map[n,m] = CFAR_coef * np.sum(cells)/N
            else:
                k = np.int(np.round(order * N))
                thresh_map[n,m] = CFAR_coef * np.sort(cells)[k-1]
            
            binary_map[n,m] = np.abs(z[n,m])**2 >= thresh_map[n,m]
    
    return thresh_map,binary_map
            
    
CFAR_coefs_array = np.array([[np.arange(1000).T,np.ones((1000,)).T],[np.arange(1000).T,np.ones((1000,)).T]])
z = np.zeros((30,30),dtype=np.complex128) # np.random.normal(0,0.5,(30,30))
z[15,15] = 1
z[20,20] = 1
z[5,5] = 1
Nl = np.array([3,3])
Nr = np.array([3,3])
Gl = np.array([2,2])
Gr = np.array([2,2])

thresh_map,binary_map =  CFAR_detector_2D(z,Nl,Nr,Gl,Gr,CFAR_coefs_array,kind="CA",order=0.75)

t = time()
thresh_map,binary_map =  CFAR_detector_2D(z,Nl,Nr,Gl,Gr,CFAR_coefs_array,kind="CA",order=0.75)
print("Elapsed time (nb) : ", time()-t," sec")
t = time()
thresh_map_true,binary_map_true = CFAR_detector(z, 1e-4,Nl,Nr,Gl,Gr,kind="CA",order=0.75,save=False,verbose=False)
print("Elapsed time: ",time()-t, " sec")

plt.figure()
plt.subplot(2,1,1)
plt.pcolormesh(thresh_map)
plt.colorbar()
plt.subplot(2,1,2)
plt.pcolormesh(thresh_map_true)
plt.colorbar()