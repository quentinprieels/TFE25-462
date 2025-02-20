from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Radcomlib imports

# Local imports
from .frame import Frame
from .modulation import Modulation
from .channel import Channel


class SchmidlAndCox(ABC):
    """
    This abstract class represents the Schmidl and Cox synchronization algorithm.
    It defines the basic structure of the algorithm and common/shared methods with
    a basic behavior that subclasses can override.
    
    All subclasses must implement a 'run' method that runs the algorithm and a 'plot'
    method that plots the results.
    """
    
    def __init__(self, frame: Frame, channel: Channel, verbose: bool = False) -> None:
        self.frame = frame
        self.channel = channel
        self.verbose = verbose
        
        # Those variables must be set by the subclasses
        self.P = None  # Metric P(d) of S&C
        self.R = None  # Metric R(d) of S&C
        self.M = None  # Metric M(d) of S&C
        self.N = None
        self.sync = None  # The synchronization point in the frame
        
    
    def find(self, metric: np.ndarray, threshold: int, min: int) -> int:
        """
        The synchronisation point is the maximum of the 'metric' when 
        the it is above the 'threshold' during at least 'min' samples.
        """
        state = "SEARCH" # SEARCH or DETECTING or FOUND
        nbr_samples_above_threshold = 0
        detected_point_val = -1
        detected_point_idx = -1
        
        # 2. Detection loop
        for i in range(0, len(metric)):
            # SEARCH -> SEARCH
            if state == "SEARCH" and metric[i] < threshold:
                continue
            
            # SEARCH -> DETECTING
            elif state == "SEARCH" and metric[i] >= threshold:
                state = "DETECTING"
                nbr_samples_above_threshold += 1
                
            # DETECTING -> DETECTING
            elif state == "DETECTING" and metric[i] >= threshold:
                nbr_samples_above_threshold += 1
                
                # Update the detected point
                if abs(metric[i] - detected_point_val) < 1e-5: # Last maximum value
                    detected_point_val = metric[i]
                    detected_point_idx = i
                
                # Above the threshold during at least 'min' samples
                if nbr_samples_above_threshold >= min:
                    state = "FOUND"
            
            # DETECTING -> SEARCH
            elif state == "DETECTING" and metric[i] < threshold:
                state = "SEARCH"
                nbr_samples_above_threshold = 0
                detected_point_val = -1
                detected_point_idx = -1
                
            # FOUND -> FOUND
            elif state == "FOUND" and metric[i] >= threshold:
                nbr_samples_above_threshold += 1
                
                if metric[i] > detected_point_val:
                    detected_point_val = metric[i]
                    detected_point_idx = i
                    
            # FOUND -> SEARCH
            elif state == "FOUND" and metric[i] < threshold:
                break
        
        return detected_point_idx if state == "FOUND" else -1


    @abstractmethod
    def run(self, threshold: int, min: int, width: int) -> None:
        raise NotImplementedError("The 'run' method must be implemented in the subclass")
    
    
    def plot(self, metrics: list[tuple[np.ndarray, str]], subtitle_dict: dict, limitate: bool = False, title: str = None) -> None:
        """
        Plot the Schmidl and Cox synchronization algorithm.
        The 'metrics' parameter is a list of tuples containing the metric and its name.
        The first metric in this list is the one considered to be compared with the threshold.
        """
        assert len(metrics) <= 2, "The number of metrics must be less or equal to 2"
        
        # Title and subtitle
        if title is None:
            title = "Schmidl and Cox Synchronization Algorithm"
        plt.figure(title, figsize=(15, 5))
        
        subtitle_info = {
            "K": f"{self.frame.K:}",
            "CP": f"{self.frame.CP:}",
            "CP_preamble": f"{self.frame.CP_preamble:}",
            "M": f"{self.frame.M:}",
            "Preamble modulation": self.frame.preamble_mod,
            "Payload modulation": self.frame.payload_mod,
        }
        subtitle = f"{' - '.join(sorted([f'{k}: {v}' for k, v in subtitle_info.items()]))}"
        subtitle_info = {
            "STO": f"{self.channel.STO:}",
            "SNR": f"{self.channel.SNR:}",
            "Paths": f"{self.channel.paths:}",
        }
        subtitle += f"\n{' - '.join([f'{k}: {v}' for k, v in subtitle_info.items()])}"
        subtitle += f"\n{' - '.join([f'{k}: {v}' for k, v in subtitle_dict.items()])}"
        
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.title(subtitle, fontsize=10, fontstyle="italic")
        
        # Signal plot
        plt.plot(np.abs(self.frame.get_frame()), label="OFDM Frame", color='tab:blue', alpha=0.5)
        plt.ylabel("Amplitude |x[n]|")
        plt.xlabel("Samples [n]")
        plt.twinx()
        
        # Detection zone
        above_threshold = metrics[0][0] > self.threshold
        crossing_points = np.where(np.diff(above_threshold.astype(int)))[0][:2]
        
        # X-axis limits
        left_xlim = self.get_sync_point() - (self.frame.CP_preamble * self.frame.M)
        if len(crossing_points) == 0:
            right_xlim = self.channel.STO + 1.25 * (self.frame.CP_preamble + self.frame.K) * self.frame.M
        else:
            right_xlim = max(crossing_points[-1] + (self.frame.K * self.frame.M) * 0.5, self.channel.STO + 1.5 * (self.frame.CP_preamble + self.frame.K) * self.frame.M)
        
        # Annotations limits
        line_annotation_height = 1.1
        line2_annotation_height = 1.08
        text_annotation_height = 1.11
        
        # Signal information (sto, cp, ...)
        frame_plot_start = 0 if limitate == False else left_xlim
        t_preamble_start = self.channel.STO
        e_preamble_start = t_preamble_start + (self.frame.CP_preamble + self.frame.K) * self.frame.M
        payload_start = t_preamble_start + 2 * (self.frame.CP_preamble + self.frame.K) * self.frame.M
        payload_end = payload_start + self.frame.N * (self.frame.CP + self.frame.K) * self.frame.M
        
        assert payload_end == len(self.frame.get_frame()), "The frame length is not correct"
        
        # Global frame information
        plt.text(frame_plot_start + (t_preamble_start - frame_plot_start) // 2, text_annotation_height, "STO", horizontalalignment='center', clip_on=True)
        plt.annotate("", xy=(t_preamble_start, line_annotation_height), xytext=(frame_plot_start, line_annotation_height), arrowprops=dict(arrowstyle="->"), clip_on=True)
        
        plt.text(t_preamble_start + (e_preamble_start - t_preamble_start) // 2, text_annotation_height, "tsync. preamble", horizontalalignment='center', clip_on=True)
        plt.annotate("", xy=(e_preamble_start, line_annotation_height), xytext=(t_preamble_start, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
        
        if not limitate:
            plt.text(e_preamble_start + (payload_start - e_preamble_start) // 2, text_annotation_height, "Eq. preamble", horizontalalignment='center', clip_on=True)
            plt.annotate("", xy=(payload_start, line_annotation_height), xytext=(e_preamble_start, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
            
            plt.text(payload_start + (payload_end - payload_start) // 2, text_annotation_height, "Payload", horizontalalignment='center', clip_on=True)
            plt.annotate("", xy=(payload_end, line_annotation_height), xytext=(payload_start, line_annotation_height), arrowprops=dict(arrowstyle="<->"), clip_on=True)
        
            plt.axvline(x=0, linestyle='--', color='black', clip_on=True)
            plt.axvline(x=payload_end, linestyle='--', color='black', clip_on=True)
            plt.axvline(x=payload_start, linestyle='--', color='black', clip_on=True)
        
        else:
            plt.text(e_preamble_start + (right_xlim - e_preamble_start) // 2, text_annotation_height, "Eq. preamble", horizontalalignment='center', clip_on=True)
            plt.annotate("", xy=(right_xlim, line_annotation_height), xytext=(e_preamble_start, line_annotation_height), arrowprops=dict(arrowstyle="<-"), clip_on=True)
                    
        plt.axvline(x=t_preamble_start, linestyle='--', color='black', clip_on=True)
        plt.axvline(x=e_preamble_start, linestyle='--', color='black', clip_on=True)
        
        if self.frame.CP_preamble > 0:
            cp_preamble_start = t_preamble_start
            cp_preamble_end = t_preamble_start + self.frame.CP_preamble * self.frame.M
            cp_preamble_copy_start = e_preamble_start - self.frame.CP_preamble * self.frame.M
            plt.text(cp_preamble_start + (cp_preamble_end - cp_preamble_start) // 2, text_annotation_height, "CP", horizontalalignment='center', clip_on=True, color='tab:gray')
            plt.annotate("", xy=(cp_preamble_end, line2_annotation_height), xytext=(cp_preamble_start, line2_annotation_height), arrowprops=dict(arrowstyle="<->", color="tab:gray"), clip_on=True)
            plt.vlines(x=cp_preamble_end, ymax=line_annotation_height, linestyles='--', color='tab:gray', ymin=0)
            
            plt.text(cp_preamble_copy_start + (e_preamble_start - cp_preamble_copy_start) // 2, text_annotation_height, "Copied", horizontalalignment='center', clip_on=True, color='tab:gray')
            plt.annotate("", xy=(e_preamble_start, line2_annotation_height), xytext=(cp_preamble_copy_start, line2_annotation_height), arrowprops=dict(arrowstyle="<->", color="tab:gray"), clip_on=True)
            plt.vlines(x=cp_preamble_copy_start, ymax=line_annotation_height, linestyles='--', color='tab:gray', ymin=0)
        
        # Metrics plot
        plt.plot(metrics[0][0], label=metrics[0][1], color='tab:red')
        if len(metrics) == 2:
            plt.plot(metrics[1][0], label=metrics[1][1], color='tab:red', linestyle='--')
         
        # Threshold and sync point   
        plt.axhline(y=self.threshold, color='tab:brown', linestyle=':', label="Threshold")
        plt.axvline(x=self.sync, color='tab:green', linestyle='-.', label="Sync point")
        plt.axvline(x=self.get_sync_point(), color='tab:green', linestyle='-.')
        
        # Detection zone
        if len(crossing_points) == 2: 
            plt.axvspan(crossing_points[0], crossing_points[1], color='gray', alpha=0.2, hatch='//')
        plt.ylabel("Metrics M[n]")
        
        plt.legend(loc="lower right")
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.25, 0.1))
        if limitate: plt.xlim(left_xlim, right_xlim)
        plt.grid(linestyle='--')  
        plt.tight_layout()
        

class SchmidlAndCoxBasic(SchmidlAndCox):
    """
    This class represents the basic Schmidl and Cox synchronization algorithm
    as described in the original paper "Robust Frequency and Timing Synchronization
    for OFDM" by T. Schmidl and D. Cox.
    """
    def sc_metrics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Schmidl and Cox synchronization metrics as described in the 
        orignal paper.
        """
        y = self.frame.frame
        L = int((self.frame.K / 2) * self.frame.M)
        
        # Initialize the metrics
        frame_length = len(y)
        P = np.zeros(frame_length, dtype=np.complex128)
        R = np.zeros(frame_length, dtype=np.float128)
        M = np.zeros(frame_length, dtype=np.float128)
        
        # Calculate the metrics
        for i in range(frame_length - 1):
            y_d   = y[i]
            y_dL  = y[i - L]     if i - L >= 0 else (0j)    # Ensure valid index
            y_d2L = y[i - 2*L]   if i - 2*L >= 0 else (0j)  # Ensure valid index
            
            # 2.2 Compute the values of P(d), R(d) and M(d)
            P[i + 1] = P[i] + np.conj(y_dL) * y_d - np.conj(y_d2L) * y_dL
            R[i + 1] = R[i] + np.abs(y_d) ** 2 - np.abs(y_dL) ** 2
            M[i + 1] = np.abs(P[i + 1]) ** 2 / (R[i + 1]) ** 2 if R[i + 1] != 0 else 0
        
        return P, R, M
     

    def run(self, threshold: int, min: int, width: int = None) -> None:
        """
        Run the Schmidl and Cox synchronization algorithm.
        """
        self.threshold = threshold
        self.min = min
        
        # 1. Calculate the metrics
        self.P, self.R, self.M = self.sc_metrics()
        
        # 2. Detect the synchronization point in the frame
        self.sync = self.find(self.M, threshold=threshold, min=min)
        
        # 3. Print a warning if the synchronization point is not found
        if self.sync == -1 and self.verbose:
            print("CAUTION: Synchronization point not found.")
    

    def get_sync_point(self) -> int:
        """
        Get the synchronization point in the frame.
        """
        M_delay = (self.frame.K * self.frame.M)  # Delay due to causality of the M metric
        true_sync = self.sync - M_delay
        return true_sync
    

    def get_sync_error(self):
        true_e_preamble_start = self.channel.STO + (self.frame.CP_preamble + self.frame.K) * self.frame.M
        return self.sync - true_e_preamble_start


    def plot(self, limitate: bool = False) -> None:
        """
        Plot the Schmidl and Cox synchronization algorithm.
        """
        title = "Basic Schmidl and Cox Synchronization Algorithm"
        subtitle = {
            "Threshold": f"{self.threshold:}",
            "Plateau size": f"{self.min:}",
            "Sync error": f"{self.get_sync_error():}"
        }
        super().plot([(self.M, "M")], subtitle, limitate=limitate, title=title)


class SchmidlAndCoxAvg(SchmidlAndCox):
    """
    This class represents the Schmidl and Cox synchronization algorithm
    with an added sliding window to average the metric.
    """
    def sc_metrics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Schmidl and Cox synchronization metrics as described in the 
        orignal paper.
        """
        y = self.frame.frame
        L = int((self.frame.K / 2) * self.frame.M)
        
        # Initialize the metrics
        frame_length = len(y)
        P = np.zeros(frame_length, dtype=np.complex128)
        R = np.zeros(frame_length, dtype=np.float128)
        M = np.zeros(frame_length, dtype=np.float128)
        
        # Calculate the metrics
        for i in range(frame_length - 1):
            y_d   = y[i]
            y_dL  = y[i - L]     if i - L >= 0 else (0j)    # Ensure valid index
            y_d2L = y[i - 2*L]   if i - 2*L >= 0 else (0j)  # Ensure valid index
            
            # 2.2 Compute the values of P(d), R(d) and M(d)
            P[i + 1] = P[i] + np.conj(y_dL) * y_d - np.conj(y_d2L) * y_dL
            R[i + 1] = R[i] + np.abs(y_d) ** 2 - np.abs(y_dL) ** 2
            M[i + 1] = np.abs(P[i + 1]) ** 2 / (R[i + 1]) ** 2 if R[i + 1] != 0 else 0
        
        return P, R, M
    
    
    def sliding_window(self, signal: np.ndarray, width: int) -> np.ndarray:
        """
        Compute the average of the 'signal' over a sliding window of size 'width'.
        """
        avg = np.zeros(len(signal), dtype=np.float128)
        
        reg = np.zeros(width, dtype=np.float128)
        running_sum = 0.0
        ptr = 0
        
        # Average the metric
        width = width * self.frame.M
        avg = np.zeros(len(signal), dtype=np.float128)
        
        reg = np.zeros(width, dtype=np.float128)
        running_sum = 0.0
        ptr = 0
        
        # Average the metric
        for i in range(len(signal)):
            running_sum -= reg[ptr]         # Remove the oldest value
            reg[ptr] = signal[i]            # Add the new value
            running_sum += signal[i]
            ptr = (ptr + 1) % width         # Circular buffer
            avg[i] = running_sum / width    # Average
        return avg


    def run(self, threshold: int, min: int, width: int) -> None:
        """
        Run the Schmidl and Cox synchronization algorithm with an added sliding window.
        """
        self.threshold = threshold
        self.min = min
        self.width = width
        
        # 1. Calculate the metrics
        self.P, self.R, self.M = self.sc_metrics()
        
        # 2. Compute the averaged metric
        self.N = self.sliding_window(self.M, width)
        
        # 3. Detect the synchronization point in the frame
        self.sync = self.find(self.N, threshold=threshold, min=min)
        
        # 4. Print a warning if the synchronization point is not found
        if self.sync == -1 and self.verbose:
            print("CAUTION: Synchronization point not found.")
        

    def get_sync_point(self) -> int:
        """
        Get the synchronization point in the frame.
        """
        M_delay = (self.frame.K * self.frame.M)                 # Delay due to causality of the M metric
        AVG_delay = self.width // 2 * self.frame.M              # Delay due to the causality of the average window
        # MID_delay = self.frame.CP_preamble // 2 * self.frame.M  Delay due to the fact that the max of the metric is in the middle of the CP (of length CP_preamble)
        true_sync = self.sync - M_delay - AVG_delay
        return true_sync
    
    
    def get_sync_error(self):
        true_e_preamble_start = self.channel.STO + (self.frame.CP_preamble + self.frame.K) * self.frame.M
        return self.sync - true_e_preamble_start
    
    
    def plot(self, limitate: bool = False, title: str = "Schmidl and Cox Synchronization Algorithm with Averaged Metric") -> None:
        """
        Plot the Schmidl and Cox synchronization algorithm with an added sliding window.
        """
        subtitle = {
            "Threshold": f"{self.threshold:}",
            "Plateau size": f"{self.min:}",
            "Window width": f"{self.width:}",
            "Sync error": f"{self.get_sync_error():}"
        }
        super().plot([(self.M, "N"), (self.N, "M")], subtitle, limitate=limitate, title=title)
        

class SchmidlAndCoxAvgR1(SchmidlAndCoxAvg):
    """
    This class represents the Schmidl and Cox synchronization algorithm
    with an added sliding window as discribed in the paper "On Timing 
    Offset Estimation for OFDM Systems" by H. Minn, M. Zeng, 
    and V. K. Bhargava
    """
    def sc_metrics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Schmidl and Cox synchronization metrics with a modified
        R[d] metric that use the whole K symbol sample instead of the last
        L symbol sample.
        """
        y = self.frame.frame
        L = int((self.frame.K / 2) * self.frame.M)
        
        # Initialize the metrics
        frame_length = len(y)
        P = np.zeros(frame_length, dtype=np.complex128)
        R = np.zeros(frame_length, dtype=np.float128)
        M = np.zeros(frame_length, dtype=np.float128)
        
        # Calculate the metrics
        for i in range(frame_length - 1):
            y_d   = y[i]
            y_dL  = y[i - L]     if i - L >= 0 else (0j)    # Ensure valid index
            y_d2L = y[i - 2*L]   if i - 2*L >= 0 else (0j)  # Ensure valid index
            
         # 2.2 Compute the values of P(d), R(d) and M(d)
            P[i + 1] = P[i] + np.conj(y_dL) * y_d - np.conj(y_d2L) * y_dL
            R[i + 1] = R[i] + np.abs(y_d) ** 2 - np.abs(y_d2L) ** 2
            M[i + 1] = np.abs(P[i + 1]) ** 2 / (1/2 * R[i + 1]) ** 2 if R[i + 1] != 0 else 0
        
        return P, R, M


    def plot(self, limitate: bool = False) -> None:
        """
        Plot the Schmidl and Cox synchronization algorithm with an added sliding window.
        """
        title = "Schmidl and Cox Synchronization Algorithm with Averaged Metric R1"
        super().plot(limitate=limitate, title=title)
    

class SchmidlAndCoxAvgR2(SchmidlAndCoxAvg):
    """
    This class represents the Schmidl and Cox synchronization algorithm
    with an added sliding window with a modified R[d] metric that use the
    the whole K symbol sample instead of the last L symbol sample. 
    This method is described in the paper "A Modified Schmidl-Cox OFDM 
    Timing Detector" by S. Wilson and R. Shang
    """ 
    def sc_metrics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Schmidl and Cox synchronization metrics with a modified
        R[d] metric that use the whole K symbol sample instead of the last
        L symbol sample.
        """
        y = self.frame.frame
        L = int((self.frame.K / 2) * self.frame.M)
        
        # Initialize the metrics
        frame_length = len(y)
        P = np.zeros(frame_length, dtype=np.complex128)
        R = np.zeros(frame_length, dtype=np.float128)
        Rl = np.zeros(frame_length, dtype=np.float128)
        M = np.zeros(frame_length, dtype=np.float128)
        
        # Calculate the metrics
        for i in range(frame_length - 1):
            y_d   = y[i]
            y_dL  = y[i - L]     if i - L >= 0 else (0j)    # Ensure valid index
            y_d2L = y[i - 2*L]   if i - 2*L >= 0 else (0j)  # Ensure valid index
            
            # 2.2 Compute the values of P(d), R(d) and M(d)
            P[i + 1] = P[i] + np.conj(y_dL) * y_d - np.conj(y_d2L) * y_dL
            R[i + 1] = R[i] + np.abs(y_d) ** 2 - np.abs(y_dL) ** 2
            Rl[i + 1] = Rl[i] + np.abs(y_dL) ** 2 - np.abs(y_d2L) ** 2
            M[i + 1] = np.abs(P[i + 1]) ** 2 / (R[i + 1] * Rl[i + 1]) if R[i + 1] != 0 and Rl[i + 1] != 0 else 0
        
        return P, R, M
        

    def plot(self, limitate: bool = False) -> None:
        """
        Plot the Schmidl and Cox synchronization algorithm with an added sliding window.
        """
        title = "Schmidl and Cox Synchronization Algorithm with Averaged Metric R2"
        super().plot(limitate=limitate, title=title)
