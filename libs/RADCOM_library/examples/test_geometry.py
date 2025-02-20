from radcomlib.geometry import state_update
from radcomlib.geometry import relative_geometric_parameters

import matplotlib.pyplot as plt
import time as time
import numpy as np

plt.close('all')    

positions = [np.array([0.,0.]),np.array([-10.,-5.]),np.array([5.,5.])]
speeds = [np.array([-1.,1.]),np.array([2.,1.]),np.array([0.,-1.])]
accelerations = [np.array([-0.2,-0.1]),np.array([0.1,-0.2]),np.array([-0.2,0.])]
directions = [np.pi/2, 0, 0]

parameters = [positions,speeds,accelerations,directions]

delta_t = 0.05
t_max = 10.

fig,axs = plt.subplots(2,2, figsize=(20,20))

for k in range(int(t_max/delta_t)+1):
    state_update(parameters,delta_t)
    
    axs[0][0].clear()    
    axs[0][0].set_xlim(-30,30)
    axs[0][0].set_ylim(-30,30)
    axs[0][0].set_title("Absolute view")
    
    for pos,speed,direction in zip(positions,speeds,directions):
        axs[0][0].plot(pos[0],pos[1],'.k',markersize=20)
        axs[0][0].quiver(pos[0],pos[1],speed[0],speed[1],angles='xy',scale_units='xy',scale=1/4,color='k')
        dir_vec = np.array([np.cos(direction),np.sin(direction)])
        axs[0][0].quiver(pos[0],pos[1],dir_vec[0],dir_vec[1],angles='xy',scale_units='xy',scale=1/4,width=0.003,color='r')
        
    for i in range(3):
        j = i+1
        reference_parameters = [positions[i],speeds[i],accelerations[i],directions[i]]
        targets_positions = positions[:]
        targets_positions.pop(i)
        targets_speeds = speeds[:]
        targets_speeds.pop(i)
        targets_accelerations = accelerations[:]
        targets_accelerations.pop(i)
        targets_directions = directions[:]
        targets_directions.pop(i)
        targets_parameters = [targets_positions,targets_speeds,targets_accelerations,targets_directions]

        relative_positions,relative_speeds,relative_accelerations,relative_directions = relative_geometric_parameters(reference_parameters,targets_parameters)

        axs[int(j/2)][j%2].clear()    
        axs[int(j/2)][j%2].set_xlim(-30,30)
        axs[int(j/2)][j%2].set_ylim(-30,30)
        axs[int(j/2)][j%2].set_title("Node {}".format(i+1))

        axs[int(j/2)][j%2].plot(0,0,'.b',markersize=20)
        axs[int(j/2)][j%2].quiver(0,0,1,0,angles='xy',scale_units='xy',scale=1/4,width=0.003,color='r')
        
        for pos,speed,direction in zip(relative_positions,relative_speeds,relative_directions):    
            axs[int(j/2)][j%2].plot(pos[0],pos[1],'.k',markersize=20)
            axs[int(j/2)][j%2].quiver(pos[0],pos[1],speed[0],speed[1],angles='xy',scale_units='xy',scale=1/4,color='k')
            dir_vec = np.array([np.cos(direction),np.sin(direction)])
            axs[int(j/2)][j%2].quiver(pos[0],pos[1],dir_vec[0],dir_vec[1],angles='xy',scale_units='xy',scale=1/4,width=0.003,color='r')
        
    fig.suptitle('Time t={:.3f}sec'.format(k*delta_t), fontsize=16)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    time.sleep(delta_t)
    
    
