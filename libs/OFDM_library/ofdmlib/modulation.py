import numpy as np
import matplotlib.pyplot as plt

# Radcomlib imports

# Local imports
from .frame import Frame


class Modulation():
    """
    This class represents the modulation block of an OFDM system.
    """
    
    def __init__(self, frame: Frame, verbose: bool = False) -> None:
        """
        """        
        self.frame = frame
        self.verbose = verbose


    def modulate_symbols(self, fsymbols: np.ndarray, CP: int, M: int) -> np.ndarray:
        """
        Modulate the given frequency domain symbols to the time domain.
        
        Parameters:
        - fsymbols: The frequency domain symbol matrix
        - CP: Cyclic prefix length
        - M: Oversampling factor
        
        Returns:
        - out_blk: The time domain 1D array containing the modulated symbol
        """
        assert len(fsymbols.shape) == 2, "The frequency domain symbols must be a 2D matrix"
        
        ifft_I = np.sqrt(self.frame.K * M) * np.fft.ifft(fsymbols, self.frame.K * M)
        if CP == 0:
            return np.reshape(ifft_I, (ifft_I.size,)) # Shape: N * K * M
        else:
            out_blk = np.concatenate([ifft_I[:, -CP * M:], ifft_I], axis=1)
            return np.reshape(out_blk, (out_blk.size,)) # Shape: N * ((CP + K) * M)    


    def modulate_frame(self, fpreamble: np.ndarray, fsymbols: np.ndarray) -> np.ndarray:
        """
        Modulate the frame.
        
        Returns:
        - frame: The modulated frame (1D array preambles + payload)
        - bits: The bits used to generate the frame (2D array payload data)
        """
        tpreamble = self.modulate_symbols(fpreamble, self.frame.CP_preamble, self.frame.M)
        tpayload = self.modulate_symbols(fsymbols, self.frame.CP, self.frame.M)
        return np.concatenate([tpreamble, tpayload]) # Shape: [2 * (CP_preamble + K) * M] + [N * ((CP + K) * M)]


    def modulate(self) -> np.ndarray:
        """
        Modulate the frame.
        """
        tsymbols = self.modulate_frame(self.frame.fpreamble, self.frame.fpayload)
        self.frame.set_frame(tsymbols)
        return tsymbols


    def plot(self):
        """
        Plot the modulated frame.
        """
        plt.figure("OFDM Frame", figsize=(15, 5))
        
        # Title and subtitle
        title = "OFDM Frame"
        subtitle_params_values = {
            "K": f"{self.frame.K:}",
            "CP": f"{self.frame.CP:}",
            "CP_preamble": f"{self.frame.CP_preamble:}",
            "M": f"{self.frame.M:}",
            "Preamble modulation": self.frame.preamble_mod,
            "Payload modulation": self.frame.payload_mod
        }
        subtitle = f"Parameters: {' - '.join(sorted([f'{k}: {v}' for k, v in subtitle_params_values.items()]))}"
        
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.title(subtitle, fontsize=10, fontstyle="italic")
        
        line_annotation_height = 1.1
        line2_annotation_height = 1.08
        text_annotation_height = 1.11
            
        # Signal plot
        normalized_frame = np.abs(self.frame.get_frame()) / np.max(np.abs(self.frame.get_frame()))
        plt.plot(normalized_frame, label="OFDM frame", color='tab:blue', alpha=0.4)       
        
        # Signal information (sto, cp, ...)
        t_preamble_start = 0
        e_preamble_start = (self.frame.CP_preamble + self.frame.K) * self.frame.M
        payload_start = 2 * (self.frame.CP_preamble + self.frame.K) * self.frame.M
        payload_end = payload_start + self.frame.N * (self.frame.CP + self.frame.K) * self.frame.M
        
        assert payload_end == len(normalized_frame), "The frame length is not correct"
        
        # Global frame information
        plt.text(t_preamble_start + (e_preamble_start - t_preamble_start) // 2, text_annotation_height, "Sync. preamble", horizontalalignment='center', clip_on=True)
        plt.annotate("", xy=(e_preamble_start, line_annotation_height), xytext=(t_preamble_start, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
        
        plt.text(e_preamble_start + (payload_start - e_preamble_start) // 2, text_annotation_height, "Eq. preamble", horizontalalignment='center', clip_on=True)
        plt.annotate("", xy=(payload_start, line_annotation_height), xytext=(e_preamble_start, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
        
        plt.text(payload_start + (payload_end - payload_start) // 2, text_annotation_height, "Payload", horizontalalignment='center', clip_on=True)
        plt.annotate("", xy=(payload_end, line_annotation_height), xytext=(payload_start, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
        
        plt.axvline(x=0, linestyle='--', color='black')
        plt.axvline(x=e_preamble_start, linestyle='--', color='black')
        plt.axvline(x=payload_start, linestyle='--', color='black')
        plt.axvline(x=payload_end, linestyle='--', color='black')
        
        # Annotation for the cyclic prefix of the preamble
        if self.frame.CP_preamble > 0:
            cp_preamble_start = 0
            cp_preamble_end = self.frame.CP_preamble * self.frame.M
            plt.text(cp_preamble_start + (cp_preamble_end - cp_preamble_start) // 2, text_annotation_height, "CP", horizontalalignment='center', clip_on=True, color='tab:gray')
            plt.annotate("", xy=(cp_preamble_end, line2_annotation_height), xytext=(cp_preamble_start, line2_annotation_height), arrowprops=dict(arrowstyle="<->", color="tab:gray"), clip_on=True)
            plt.vlines(x=cp_preamble_end, ymax=line_annotation_height, linestyles='--', color='tab:gray', ymin=0)
            
            cp2_preamble_start = e_preamble_start
            cp2_preamble_end = e_preamble_start + self.frame.CP_preamble * self.frame.M
            plt.text(cp2_preamble_start + (cp2_preamble_end - cp2_preamble_start) // 2, text_annotation_height, "CP", horizontalalignment='center', clip_on=True, color='tab:gray')
            plt.annotate("", xy=(cp2_preamble_end, line2_annotation_height), xytext=(cp2_preamble_start, line2_annotation_height), arrowprops=dict(arrowstyle="<->", color="tab:gray"), clip_on=True)
            plt.vlines(x=cp2_preamble_end, ymax=line_annotation_height, linestyles='--', color='tab:gray', ymin=0)
            
        for i in range(self.frame.N):
            symbol_start = payload_start + i * (self.frame.CP + self.frame.K) * self.frame.M
            symbol_end = payload_start + (i + 1) * (self.frame.CP + self.frame.K) * self.frame.M
            plt.text(symbol_start + (symbol_end - symbol_start) // 2, 0.94 * text_annotation_height, f"Symbol {i}", horizontalalignment='center', clip_on=True, color='tab:blue')
            plt.vlines(x=symbol_end, ymax=line_annotation_height, linestyle='--', color='tab:blue', ymin=0)
            
        if self.frame.CP > 0:
            pass
            
        plt.xlabel("Samples [n]")
        plt.ylabel("Amplitude |x[n]| (normalized)")
        plt.ylim(0, 1.2)
        plt.grid(linestyle='--')  
        plt.tight_layout()
