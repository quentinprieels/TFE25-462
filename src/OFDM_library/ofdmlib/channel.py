import numpy as np
import matplotlib.pyplot as plt

# Radcomlib imports

# Local imports
from .frame import Frame
from .modulation import Modulation


class Channel():
    """
    This class represents the channel block of an OFDM system.
    """

    def __init__(self, frame: Frame, random_seed: int = None, verbose: bool = False) -> None:
        if random_seed is None:
            random_seed = np.random.default_rng().integers(0, 2**32)
            if verbose:
                print(f"Channel random seed: {random_seed}")
        self.generator = np.random.default_rng(random_seed)
        
        self.frame = frame
        self.SNR = None
        self.STO = None
        self.paths = None

    
    def add_noise(self, SNR: float) -> None:
        """
        Add AWG noise to the given frame.
        """
        assert SNR >= 0, "The SNR must be a positive float"
        
        self.SNR = SNR
        if SNR == np.inf:
            return

        snr_lin = 10 ** (SNR / 10)
        signal_power = np.mean(np.abs(self.frame.get_frame()) ** 2)
        noise_power = signal_power / snr_lin
        noise_frame = np.sqrt(noise_power / 2) * (self.generator.normal(size=len(self.frame.get_frame())) + 1j * self.generator.normal(size=len(self.frame.get_frame())))
        noisy_frame = self.frame.get_frame() + noise_frame
        self.frame.set_frame(noisy_frame)


    def add_sto(self, STO: int) -> None:
        """
        Add a sample timing offset to the given frame.
        """
        assert STO >= 0, "The STO must be a positive integer"
        
        self.STO = STO
        if STO == 0:
            return

        sto_frame = np.hstack([np.zeros(STO), self.frame.get_frame()])
        self.frame.set_frame(sto_frame)
    
        
    def add_multipath(self, paths: list[tuple[int, float]]) -> None:
        """
        Add multipath to the given frame. 
        Each path is defined by a tuple (delay, attenuation).
        """
        assert len(paths) > 0, "The paths list must not be empty"
        
        self.paths = paths
        
        multipath_frame = self.frame.get_frame()
        for path in paths:
            delay, attenuation = path
            new_frame = np.zeros(len(self.frame.get_frame()), dtype=complex)
            if delay > 0:
                new_frame[delay:] = attenuation * self.frame.get_frame()[:-delay]
            elif delay < 0:
                new_frame[:delay] = attenuation * self.frame.get_frame()[-delay:]
            else:
                new_frame = attenuation * self.frame.get_frame()
            multipath_frame += new_frame
        self.frame.set_frame(multipath_frame)
        
        
    def plot(self) -> None:
        """
        Plot the channel.
        """
        plt.figure("OFDM Frame with channel effects", figsize=(15, 5))
        
        # Title and subtitle
        title = "OFDM Frame with channel effects"
        subtitle_infos = {
            "K": f"{self.frame.K:}",
            "CP": f"{self.frame.CP:}",
            "CP_preamble": f"{self.frame.CP_preamble:}",
            "M": f"{self.frame.M:}",
            "Preamble modulation": self.frame.preamble_mod,
            "Payload modulation": self.frame.payload_mod
        }
        subtitle = f"{' - '.join(sorted([f'{k}: {v}' for k, v in subtitle_infos.items()]))}"
        subtitle_infos = {
            "STO": f"{self.STO:}",
            "SNR": f"{self.SNR:}",
            "Paths": f"{self.paths:}",
        }
        subtitle += f"\n{' - '.join(sorted([f'{k}: {v}' for k, v in subtitle_infos.items()]))}"
        
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.title(subtitle, fontsize=10, fontstyle="italic")
        
        line_annotation_height = 1.1
        line2_annotation_height = 1.08
        text_annotation_height = 1.11
            
        # Signal plot
        normalized_frame = np.abs(self.frame.get_frame()) / np.max(np.abs(self.frame.get_frame()))
        plt.plot(normalized_frame, label="OFDM frame", color='tab:blue', alpha=0.4)       
        
        # Signal information (sto, cp, ...)
        t_preamble_start = self.STO
        e_preamble_start = t_preamble_start + (self.frame.CP_preamble + self.frame.K) * self.frame.M
        payload_start = t_preamble_start + 2 * (self.frame.CP_preamble + self.frame.K) * self.frame.M
        payload_end = payload_start + self.frame.N * (self.frame.CP + self.frame.K) * self.frame.M
        
        assert payload_end == len(normalized_frame), "The frame length is not correct"
        
        # Global frame information
        plt.text((t_preamble_start) // 2, text_annotation_height, "STO", horizontalalignment='center', clip_on=True)
        plt.annotate("", xy=(t_preamble_start, line_annotation_height), xytext=(0, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
        
        plt.text(t_preamble_start + (e_preamble_start - t_preamble_start) // 2, text_annotation_height, "Sync. preamble", horizontalalignment='center', clip_on=True)
        plt.annotate("", xy=(e_preamble_start, line_annotation_height), xytext=(t_preamble_start, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
        
        plt.text(e_preamble_start + (payload_start - e_preamble_start) // 2, text_annotation_height, "Eq. preamble", horizontalalignment='center', clip_on=True)
        plt.annotate("", xy=(payload_start, line_annotation_height), xytext=(e_preamble_start, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
        
        plt.text(payload_start + (payload_end - payload_start) // 2, text_annotation_height, "Payload", horizontalalignment='center', clip_on=True)
        plt.annotate("", xy=(payload_end, line_annotation_height), xytext=(payload_start, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
        
        plt.axvline(x=0, linestyle='--', color='black')
        plt.axvline(x=t_preamble_start, linestyle='--', color='black')
        plt.axvline(x=e_preamble_start, linestyle='--', color='black')
        plt.axvline(x=payload_start, linestyle='--', color='black')
        plt.axvline(x=payload_end, linestyle='--', color='black')
        
        # Annotation for the cyclic prefix of the preamble
        if self.frame.CP_preamble > 0:
            cp_preamble_start = t_preamble_start
            cp_preamble_end = t_preamble_start + self.frame.CP_preamble * self.frame.M
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
