"""
Transmission preprocessing steps.
This module outputs the necessary data in the defined output folder (see configuration.py file).
"""

from configuration import transmission_config, save_config
from preamble import generate_preamble, save_preamble
from symbols import generate_symbols, save_symbols
from generate_tx import generate_tx

from radcomlib.ofdm import SISO_OFDM_DFRC_TX

import numpy as np
import matplotlib.pyplot as plt

def main():
    ####################################
    # Configuration
    ####################################
    config = transmission_config(save=False)
    save_config(config)
    
    
    ####################################
    # Preamble generation
    ####################################
    preamble = generate_preamble(config)
    save_preamble(preamble, config)
    
    
    ####################################
    # Symbol generation (data + pilots)
    ####################################
    symbols, pilots = generate_symbols(config)
    save_symbols(symbols, pilots, config)
    
    ####################################
    # OFDM modulation
    ####################################
    tx_sig = SISO_OFDM_DFRC_TX(symbols, preamble, config['L_CP'], L=config['L'], periodic_preamble=False)
    
    # Save the transmission samples
    sig_len = generate_tx(tx_sig, config)
    print(f"   SIG_LEN: {sig_len}")
    print(f"2x SIG_LEN: {2 * sig_len}")

    # Plot the transmitted signal
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.real(tx_sig))
    plt.subplot(2, 1, 2)
    plt.plot(np.imag(tx_sig))
    plt.title("TX signal")
    plt.show()

    
if __name__ == "__main__":
    main()