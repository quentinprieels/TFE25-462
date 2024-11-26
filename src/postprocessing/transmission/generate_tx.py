"""
Signal processing functions for transmission data.
"""

### Expose the transmission function
__all__ = ["generate_tx"]


### Imports
import numpy as np


### Functions
def generate_tx(tx_sig, config) -> int:
    """
    The function creates the <.txt> file for the transmission with the
    experimental setup. The file must be moved into RamDisk to be transmited as
    defined in the jupyter notebook "section 3: Transmission d'un signal"

    Parameters
    ----------
    tx_sig : numpy complex vector
        vector with the I/Q samples to transmits

    config : dict
        Configuration dictionary. It must contain the following keys:
            - 'output_folder': Output folder for the transmission data
            - 'output_file': Output file for the transmission data

    Returns
    -------
    sig_len : int
        Number of complex samples to transmit. The value must be encoded in the
        execution command for the transmission (see section 3:
        Transmission d'un signal of the jupyter notebook)

    """
    # The number of complex symbols
    sig_len = len(tx_sig)

    # Interleaving between the real and the imaginary part
    split_sig = np.zeros((sig_len * 2))
    split_sig[::2] = np.real(tx_sig)
    split_sig[1::2] = np.imag(tx_sig)

    # Normalisation for the tranmission of the signal. We ensure here that the
    # norm of the samples to transmit is smaller than 1
    split_sig = split_sig / np.max(split_sig) * 0.7

    # Save the result
    if config["save"]:
        np.savetxt(config["output_folder"] + config["scenario_name"] + "_tx_sig.txt", split_sig)

    return sig_len
