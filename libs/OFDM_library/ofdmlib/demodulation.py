import numpy as np
import matplotlib.pyplot as plt

# Radcomlib imports
from radcomlib.comm_toolbox import inverse_mapping

# Local imports
from .frame import Frame
from .modulation import Modulation
from .channel import Channel
from .schmidlAndCox import SchmidlAndCoxAvg


class Demodulation:
    """
    This class represents the demodulation block of an OFDM system.
    """
    def __init__(self, frame: Frame, sync_point: int, channel: Channel, remove_first: bool = False, verbose: bool = False) -> None:
        """
        """
        self.frame = frame
        self.sync_point = sync_point
        self.channel = channel
        self.remove_first = remove_first
        self.verbose = verbose
        
        self.H = None
        self.fpreamble, self.fpayload = None, None
    
    
    def demodulate_symbols(self, tsymbols: np.ndarray, N: int, CP: int, M: int) -> np.ndarray:
        """
        Demodulate the given time domain symbols to the frequency domain.
        """
        n_sym = N * (CP + self.frame.K) * M
        
        # Remove excess samples
        if len(tsymbols) > n_sym:
            if self.verbose: print(f"CAUTION: Excess samples for {n_sym} symbols, got {len(tsymbols)}. Removing excess samples.")
            tsymbols = tsymbols[:n_sym]
        
        # Check if there are enough samples
        if len(tsymbols) < n_sym:
            if self.verbose: print(f"CAUTION: Not enough samples for {n_sym} symbols, got {len(tsymbols)}. Adding zero padding.")
            zeropad = np.zeros((n_sym - len(tsymbols),))
            tsymbols = np.concatenate([tsymbols, zeropad])
            
        # Reshape the time domain symbols to a matrix and remove the cyclic prefix
        if CP == 0:
            tsymbols = np.reshape(tsymbols, (N, (self.frame.K) * M))
        elif self.remove_first:
            tsymbols = np.reshape(tsymbols, (N, (CP + self.K) * M))[:, CP * M:]  # remove the FIRST M*CP samples
        else:
            tsymbols = np.reshape(tsymbols, (N, (CP + self.frame.K) * M))[:, :-CP * M]  # remove the LAST M*CP samples
        
        # Perform the FFT
        fsymbols = 1/np.sqrt(self.frame.K * M) * np.fft.fft(tsymbols, axis=1)        
        return fsymbols[:, :self.frame.K] # Shape: (N, K)
    
    
    def demodulate_frame(self, tsymbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Demodulate the frame.
        """
        # Premable demodulation
        n_preamble_tsymbol = 2
        
        tpreamble = tsymbols[:n_preamble_tsymbol * (self.frame.CP_preamble + self.frame.K) * self.frame.M]
        fpreamble = self.demodulate_symbols(tpreamble, n_preamble_tsymbol, self.frame.CP_preamble, self.frame.M)  
        
        tpayload = tsymbols[n_preamble_tsymbol * (self.frame.CP_preamble + self.frame.K) * self.frame.M:]  
        fpayload = self.demodulate_symbols(tpayload, self.frame.N, self.frame.CP, self.frame.M)
        
        return fpreamble, fpayload
    
    
    def demodulate(self) -> None:
        """
        Demodulate the frame.
        """
        # Demodulate each symbol
        tsymbols = self.frame.get_frame()[self.sync_point:]
        self.fpreamble, self.fpayload = self.demodulate_frame(tsymbols)
        
    def equalize(self) -> np.ndarray:
        """
        Equalize the received symbols.
        """        
        # Estimate the channel
        self.H = self.frame.fpreamble[1] / (self.fpreamble[1] + 1e-10)
        
        # Equalize the received symbols
        self.fpayload = self.fpayload * self.H
        
    
    def get_ber(self) -> float:
        """
        Compute the bit error rate.
        """        
        ber = 0
        for i in range(self.frame.N):
            ber += np.mean(self.frame.bits[i] != inverse_mapping(self.fpayload[i], self.frame.payload_mod))
        return ber / self.frame.N
        
    
    def plot_constellation(self, title: str = None) -> None:
        """
        Plot the constellation diagram.
        """
        if title is None:
            title = "Constellation Diagram"
            
        subtitle_params_values = {
            "Payload modulation": self.frame.payload_mod,
            "SNR": f"{self.channel.SNR:}",
        }
        subtitle = f"Parameters: {' - '.join(sorted([f'{k}: {v}' for k, v in subtitle_params_values.items()]))}"
        
        plt.figure(title, figsize=(6, 6))
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.title(subtitle, fontsize=10, fontstyle="italic")
        plt.scatter(np.real(self.fpayload), np.imag(self.fpayload), label="Recieved", c="tab:blue")
        plt.scatter(np.real(self.frame.fpayload), np.imag(self.frame.fpayload), label="Sent", c="tab:orange", marker="x")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(np.arange(-2, 2.1, 0.5))
        plt.yticks(np.arange(-2, 2.1, 0.5))
        plt.grid(True, which='both', linestyle='--')
        plt.legend()
        plt.tight_layout()
