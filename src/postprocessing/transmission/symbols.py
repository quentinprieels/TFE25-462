"""
Symbols generation (pilots and data) for the transmission.
"""

### Expose the preamble generation function
__all__ = ["generate_symbols"]


### Imports
import numpy as np


### Utils
def __generate_pilots_idx(N_sc: int, Nt: int, Nf: int, P: int) -> tuple:
    """
    Generate the indices of the pilot subcarriers.
    
    Parameters
    ----------
    N_sc: int
        Number of subcarriers
    Nt: int
        Pilot spacing in the time domain
    Nf: int
        Pilot spacing in the frequency domain
    P: int
        Number of transmitted pulses (number of OFDM symbols
        
    Returns
    -------
    tuple
        Tuple containing the indices of the pilot subcarriers in the time and frequency domain
    """
    # Note: the first and the last subcarrier is always included in the pilots. 
    # Same for the first and the last OFDM symbol.
    # This ensure that we will not perform any extrapolation.
    pilots_idx_f = np.concatenate((np.arange(0, N_sc - 1, Nf), [N_sc - 1]))
    pilots_idx_t = np.concatenate((np.arange(0, P - 1, Nt), [P - 1]))
    return pilots_idx_t, pilots_idx_f


### Fuctions - Symbols generation
def generate_symbols(config: dict) -> tuple:
    """
    Generate the symbols for the transmission.
    
    Parameters
    ----------
    config: dict
        Configuration dictionary. It must contain the following keys:
            - 'P': Number of transmitted pulses (number of OFDM symbols)
            - 'N_sc': Number of subcarriers
            - 'const': Constellation symbols
            - 'Nt': Pilot spacing in the time domain
            - 'Nf': Pilot spacing in the frequency domain
            
    Returns
    -------
    tuple
        Tuple containing the symbols matrix and the pilots matrix
    """
    # Generate the pilots indices
    pilots_idx_t, pilots_idx_f = __generate_pilots_idx(config['N_sc'], config['Nt'], config['Nf'], config['P'])
    
    # Generate the symbols
    # Matrix of size: Number of OFDM symbols x Number of subcarriers (countains both the pilots and data symbols)
    symbols = np.random.choice(config['const'], (config['P'], config['N_sc']))
    
    # Get the corresponding pilots
    # Matrix of size: sizeof(pilots_idx_t) x sizeof(pilots_idx_f)
    pilots_idx_t_mesh, pilots_idx_f_mesh = np.meshgrid(pilots_idx_t, pilots_idx_f)
    pilots = symbols[pilots_idx_t_mesh.T, pilots_idx_f_mesh.T]
    
    return symbols, pilots


def save_symbols(symbols: np.ndarray, pilots: np.ndarray, config: dict):
    """
    Save the symbols and the pilots to a file.
    
    Parameters
    ----------
    symbols: np.ndarray
        Symbols matrix
    pilots: np.ndarray
        Pilots matrix
    config: dict
        Configuration dictionary. It must contain the following keys:
            - 'output_folder': Output folder
            - 'scenario_name': Transmission scenario name
            - 'save': Save the configuration
    """
    if not config["save"]:
        return
    
    np.save(config["output_folder"] + config["scenario_name"] + "_" + "symbols.npy", symbols)
    np.save(config["output_folder"] + config["scenario_name"] + "_" + "pilots.npy", pilots)
    
    

"""
Test and analysis of the symbols generation.
"""
if __name__ == "__main__":
    # Plot the symbols matrix and color the pilots
    from configuration import transmission_config
    import matplotlib.pyplot as plt
    
    def visualise_symbols(symbols, config):
        pilots_idx_t, pilots_idx_f = __generate_pilots_idx(config['N_sc'], config['Nt'], config['Nf'], config['P'])
        
        pilot_mask = np.zeros_like(symbols, dtype=bool)
        pilot_mask[pilots_idx_t[:, None], pilots_idx_f] = True
        
        color_matrix = np.zeros_like(symbols, dtype=bool)
        color_matrix[pilot_mask] = 1

        plt.figure()
        plt.imshow(color_matrix, aspect='equal', cmap='coolwarm', interpolation='none')
        plt.title("Symbols matrix with pilots")
        plt.xlabel("Subcarrier index")
        plt.ylabel("OFDM symbol index")
        plt.colorbar(ticks=[0, 1], label='0: Data, 1: Pilots')
        plt.show()
        
    # Configuration
    config = transmission_config(save=False, Nf=1, Nt=1)  #? Modify Nf and Nt here
    symbols, pilots = generate_symbols(config)
    
    # Visualisation
    visualise_symbols(symbols, config)
    