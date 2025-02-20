"""

"""

__all__ = ["relative_geometric_parameters", 
           "absolute_geometric_parameters", 
           "compute_channel_parameters",
           "compute_geometric_parameters",
           "rotate_directions", 
           "state_update",
           "channel_parameters_update"]

from numpy import array
from numpy import cos, sin, arctan2, sign, exp, abs, pi, sqrt, power
from numpy.linalg import norm

c = 299792458 # m/s

def propagation_model(r,fc,link="radar"):
    alpha = 2.
    beta = 4*pi

    if link == "radar":
        return (c/fc)**2/(4*pi) * beta**(-1)*power(2*r,-alpha)
    else:
        return (c/fc)**2/(4*pi) * beta**(-1)*power(r,-alpha)

def rotation_matrix(theta):
    return array([[cos(theta), -sin(theta)], \
                     [sin(theta),  cos(theta)]])

def relative_geometric_parameters(reference_parameters,targets_parameters):    
    reference_position = reference_parameters[0][0]
    reference_speed = reference_parameters[1][0]
    reference_acceleration = reference_parameters[2][0]
    reference_direction = reference_parameters[3][0]
    
    targets_positions = targets_parameters[0]
    targets_speeds = targets_parameters[1]
    targets_accelerations = targets_parameters[2]
    targets_directions = targets_parameters[3]
    
    relative_positions = []
    relative_speeds = []
    relative_accelerations = []
    relative_directions = []    
    for target_pos,target_speed,target_acc,target_direction in zip(targets_positions,targets_speeds,targets_accelerations,targets_directions):
        relative_positions.append(rotation_matrix(-reference_direction) @ (target_pos - reference_position))
        relative_speeds.append(rotation_matrix(-reference_direction) @ (target_speed - reference_speed))
        relative_accelerations.append(rotation_matrix(-reference_direction) @ (target_acc - reference_acceleration))
        relative_directions.append(target_direction - reference_direction)

    return [relative_positions,relative_speeds,relative_accelerations,relative_directions]

def absolute_geometric_parameters(reference_parameters,relative_parameters):
    reference_position = reference_parameters[0][0]
    reference_speed = reference_parameters[1][0]
    reference_acceleration = reference_parameters[2][0]
    reference_direction = reference_parameters[3][0]
    
    relative_positions = relative_parameters[0]
    relative_speeds = relative_parameters[1]
    relative_accelerations = relative_parameters[2]
    relative_directions = relative_parameters[3]
    
    targets_positions = []
    targets_speeds = []
    targets_accelerations = []
    targets_directions = []    
    for relative_pos,relative_speed,relative_acc,relative_direction in zip(relative_positions,relative_speeds,relative_accelerations,relative_directions):
        targets_positions.append(rotation_matrix(reference_direction) @ relative_pos + reference_position)
        targets_speeds.append(rotation_matrix(reference_direction) @ relative_speed + reference_speed)
        targets_accelerations.append(rotation_matrix(reference_direction) @ relative_acc + reference_acceleration)
        targets_directions.append(relative_direction + reference_direction)
        
    return [targets_positions,targets_speeds,targets_directions]

def compute_channel_parameters(reference_parameters,targets_parameters,fc,c,link="radar"):
    relative_parameters = relative_geometric_parameters(reference_parameters,targets_parameters)
    relative_positions = relative_parameters[0]
    relative_speeds = relative_parameters[1]
    
    targets_delays = []
    targets_doppler = []
    targets_angles = []
    targets_coefs = []
    for relative_pos,relative_speed in zip(relative_positions,relative_speeds):
        if link == "radar":
            targets_delays.append(2*norm(relative_pos)/c)
            targets_doppler.append(-sign(relative_pos[0])*2*relative_speed[0]*fc/c)
            targets_angles.append(arctan2(relative_pos[1],relative_pos[0]))
            targets_coefs.append(-sqrt(propagation_model(norm(relative_pos),fc,"radar"))*exp(-1j*2*pi*fc*2*norm(relative_pos)/c))
        else:
            targets_delays.append(norm(relative_pos)/c)
            targets_doppler.append(-sign(relative_pos[0])*relative_speed[0]*fc/c)
            targets_angles.append(arctan2(relative_pos[1],relative_pos[0]))
            targets_coefs.append(sqrt(propagation_model(norm(relative_pos),fc,"comm"))*exp(-1j*2*pi*fc*norm(relative_pos)/c))

    channel_parameters = []            
    for target_coef,target_doppler,target_delay,target_angle in zip(targets_coefs,targets_doppler,targets_delays,targets_angles):
        channel_parameters.append([target_coef,target_doppler,target_delay,target_angle])

    return channel_parameters

def compute_geometric_parameters(reference_parameters,targets_channel_parameters,fc,c):
    targets_positions = []
    targets_speeds = []
    
    if len(targets_channel_parameters[0]) == 4:
        for target_coef,target_doppler,target_delay,target_angle in targets_channel_parameters:            
            target_relative_position = c*target_delay/2 * array([cos(target_angle),sin(target_angle)])
            target_relative_speed = -target_doppler*c/fc/2 * array([cos(target_angle),sin(target_angle)])
            
            target_absolute_parameters = absolute_geometric_parameters(reference_parameters,[[target_relative_position,target_relative_speed,0,0]])
            targets_positions.append(target_absolute_parameters[0][0])
            targets_speeds.append(target_absolute_parameters[0][1])
    else:
        for target_coef,target_doppler,target_delay in targets_channel_parameters:
            target_relative_position = array([c*target_delay/2,0])
            target_relative_speed = -array([target_doppler*c/fc/2,0])
    
            target_absolute_parameters = absolute_geometric_parameters(reference_parameters,[[target_relative_position],[target_relative_speed],[array([0.,0.])],[0.]])
            targets_positions.append(target_absolute_parameters[0][0])
            targets_speeds.append(target_absolute_parameters[1][0])    
    
    return targets_positions,targets_speeds
        

def rotate_directions(parameters,theta):
    directions = parameters[3]
    directions += theta
    
def state_update(parameters,delta_t):
    positions = parameters[0]
    speeds = parameters[1]
    accelerations = parameters[2]
    
    if isinstance(positions,list):
        for pos,speed,acc in zip(positions,speeds,accelerations):
            pos += speed * delta_t
            speed += acc * delta_t
    else:
        positions += speeds * delta_t
        speeds += accelerations * delta_t 

def channel_parameters_update(channel_parameters,delta_t,fc,delta_fD=-1,link="radar"):
    new_channel_parameters = []
    for i,(coef,doppler,delay,angle) in enumerate(channel_parameters):  
        x_new = abs(c/2*delay - c/2*doppler/fc*delta_t)
        
        if isinstance(delta_fD,list):
            doppler_new = doppler + delta_fD[i]
        else:
            doppler_new = doppler

        angle_new = angle
        
        if link == "radar":
            delay_new = 2*x_new/c
            coef_new = -sqrt(propagation_model(x_new,fc,"radar"))*exp(-1j*2*pi*fc*delay_new)
        else:
            delay_new = x_new/c
            coef_new = sqrt(propagation_model(x_new,fc,"comm"))*exp(-1j*2*pi*fc*delay_new)
        new_channel_parameters.append([coef_new,doppler_new,delay_new,angle_new])
    return new_channel_parameters