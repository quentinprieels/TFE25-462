"""
Configuration module for the transmission scenario.
"""

### Expose the configuration function
__all__ = ["transmission_config", "save_config"]


### Imports
from radcomlib.comm_toolbox import BPSK_const, QPSK_const, QAM16_const
import numpy as np
from math import gcd


### Functions
def transmission_config(
    # Transmission
    P: int = 256, B: int = 40e6, N_sc: int = 1024, L_CP: int = 256, L: int = 5,
    
    # Synchronisation
    u: int = 25,
    
    # Modulation
    const_type: str = "QPSK", Nt: int = 1, Nf: int = 1,
    
    # Outputs
    output_folder: str = "../data/", scenario_name: str = "Setup_40MHz", save: bool = True
) -> dict:
    """
    Configure the transmission scenario.
    
    Parameters
    ----------
    P: int
        Number of transmitted pulses (number of OFDM symbols)
    B: int
        Bandwidth [Hz]: The maximum available bandwith is 100 MHz
    N_sc: int
        Number of subcarriers
    L_CP: int
        Cyclic Prefix (CP) length
    L: int
        Tx oversampling factor
        
    const_type: str
        Constellation ; either "BPSK", "QPSK", "16QAM"
    Nt: int
        Pilot spacing in the time domain (defines how often the OFDM symbol will contain pilots)
    Nf: int
        Pilot spacing in the frequency domain (defines the spacing in number of subcarriers)
        
    output_folder: str
        Output folder
    scenario_name: str
        Transmission scenario name
    
    Returns
    -------
    dict
        Configuration dictionary
        
    Raises
    ------
    ValueError
        If the configuration contains potential errors
    """
        
    # Check for potential errors (logical errors)
    if L * B > 200e6:
        raise ValueError("The product of the bandwith and the oversampling factor must be smaller or equal to 200 MHz")
    if L_CP >= N_sc:
        raise ValueError("The CP length must be smaller than the number of subcarriers")
    if u < 0 or u >= N_sc:
        raise ValueError("The root index of the Zadoff-Chu sequence must be in the range [0, N_sc-1]")
    if gcd(N_sc, u) != 1:
        raise ValueError("The root index of the Zadoff-Chu sequence must be coprime with the number of subcarriers")
    
    # Define values
    Df = B / N_sc           # Subcarrier spacing [Hz]
    T = 1 / Df              # OFDM symbol duration [s]
    Tc = 1 / B              # Time between two samples in time domain if no oversampling factor [s]
    fs_tx = L / Tc          # TX sampling frequency [Hz]
    T_CP = L_CP * Tc        # Duration of the CP [s]
    T_PRI = T + T_CP        # Duration of the full OFDM symbol (with the CP included) [s]
    pulse_width = N_sc * L  # Number of samples in the useful part of the OFDM symbols
    Ng = L_CP * L           # Number of samples in the CP part of the OFDM symbols
    
    # Define the constellation
    if const_type == "BPSK":
        const = BPSK_const###
    elif const_type == "QPSK":
        const = QPSK_const
    elif const_type == "16QAM":
        const = QAM16_const
    else:
        raise ValueError("Unknown constellation type")
    
    return {
        "P": P, "B": B, "N_sc": N_sc, "L_CP": L_CP, "L": L,
        "u": u,
        "const_type": const_type, "const": const, "Nt": Nt, "Nf": Nf,
        "output_folder": output_folder, "scenario_name": scenario_name, "save": save,
        "Df": Df, "T": T, "Tc": Tc, "fs_tx": fs_tx, "T_CP": T_CP, "T_PRI": T_PRI,
        "pulse_width": pulse_width, "Ng": Ng
    }


def save_config(config: dict):
    """
    Save the configuration to a file.
    
    Parameters
    ----------
    config: dict
        Configuration dictionary. It must contains the following keys:
            - 'output_folder': Output folder
            - 'scenario_name': Transmission scenario name
            - 'P': Number of transmitted pulses (number of OFDM symbols)
            - 'B': Bandwidth [Hz]
            - 'N_sc': Number of subcarriers
            - 'L_CP': Cyclic Prefix (CP) length
            - 'L': Tx oversampling factor
            - 'const_type': Constellation type
            - 'Nt': Pilot spacing in the time domain
            - 'Nf': Pilot spacing in the frequency domain
            - 'u': Root index of the Zadoff-Chu sequence
            - 'save': Save the configuration
    """
    if not config["save"]:
        return
    
    saved_config = {
        "P": config["P"], 
        "B": config["B"], 
        "N_sc": config["N_sc"], 
        "L_CP": config["L_CP"], 
        "L": config["L"],
        "const_type": config["const_type"], 
        "Nt": config["Nt"], 
        "Nf": config["Nf"],
        "u": config["u"]
    }
    
    np.save(config["output_folder"] + config["scenario_name"] + "_" + "config.npy", saved_config)
