"""
Preamble generation. The preamble is a sequence of symbols, known at the receiver side,
that is used to synchronize the receiver with the transmitter.
"""

### Expose the preamble generation function
__all__ = ["generate_preamble"]


### Imports
import numpy as np


### Utils
def __generate_zadoff_chu(N, u, q=0):
    """
    Generate a Zadoff-Chu sequence.
    
    Parameters
    ----------
    N: int
        Number of samples
    u: int
        Root index
    q: int
        Cyclic shift
        
    Returns
    -------
    np.ndarray
        Zadoff-Chu sequence
    """
    n = np.arange(N)
    cf = N % 2
    zc_seq = np.exp(-1j * np.pi * u * n * (n + cf + 2 * q) / N)
    return zc_seq


### Fuctions - Preamble generation
def random_preamble(config: dict) -> np.ndarray:
    """
    Generate a random preamble. 
    In this case the preamble is not designed to have good autocorrelation properties.
    
    Parameters
    ----------
    config: dict
        Configuration dictionary. It must contain the following keys:
            - 'N_sc': Number of subcarriers
            - 'const': Constellation symbols
            
    Returns
    -------
    np.ndarray
        Random preamble of size 'N_sc'
    """
    # Preamble generation
    preamble = np.random.choice(config['const'], (int(config['N_sc']),))
    
    return preamble


def zadoff_chu_preamble(config: dict) -> np.ndarray:
    """
    Generate a Zadoff-Chu preamble.

    Parameters
    ----------
    config: dict
        Configuration dictionary. It must contain the following keys:
            - 'N_sc': Number of subcarriers
            - 'u': Root index of the Zadoff-Chu sequence
    """    
    # Zadoff-Chu preamble generation
    preamble = __generate_zadoff_chu(config['N_sc'], config['u'])
    
    return preamble


def lstf_preamble(config: dict) -> np.ndarray:
    """
    Generate a Long Training Symbol Field (LSTF) preamble.
    """
    # TODO: Implement the LSTF preamble generation
    raise NotImplementedError("LSTF preamble generation is not implemented yet.")


### Export the desired function
generate_preamble = zadoff_chu_preamble #? Chose the desired preamble generation method here

def save_preamble(preamble: np.ndarray, config: dict):
    """
    Save the preamble to a file.
    
    Parameters
    ----------
    preamble: np.ndarray
        Preamble
    config: dict
        Configuration dictionary. It must contain the following keys:
            - 'output_folder': Output folder
            - 'scenario_name': Transmission scenario name
    """
    if not config["save"]:
        return
    
    np.save(config['output_folder'] + config['scenario_name'] + "_" + "preamble.npy", preamble)



"""
Test and analysis of the preamble generation.
"""
if __name__ == "__main__":
    # Visualise the correlation of the preamble with itself
    from configuration import transmission_config
    import matplotlib.pyplot as plt
    
    def visualize_correlation(preamble: np.ndarray):
        """
        Visualize the preamble correlation.
        """
        # Compute the correlation
        corr = np.correlate(preamble, preamble, mode='full')
        
        # Plot the correlation
        plt.figure("Preamble correlation")
        plt.subplot(2, 1, 1)
        plt.stem(preamble)
        plt.subplot(2, 1, 2)
        plt.plot(corr)
        plt.show()
    
    config = transmission_config()
    preamble = zadoff_chu_preamble(config)
    visualize_correlation(preamble)
    