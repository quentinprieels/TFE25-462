import numpy as np
import pandas as pd
import tqdm
from vcdvcd import VCDVCD
import matplotlib.pyplot as plt
from tabulate import tabulate

from ofdmlib.frame import Frame
from ofdmlib.modulation import Modulation
from ofdmlib.channel import Channel

# Constants for SC16 and SC32 formats
SC16_MIN = -2**15
SC16_MAX = 2**15 - 1
SC32_MIN = -2**31
SC32_MAX = 2**31 - 1
ROUND = lambda x: np.round(x)


###########
# Results #
###########
signals = {}
signals_scaled = {}
signals_normalized = {}


###########
# Helpers #
###########
def get_signal_from_vcd(vcdfile: str, 
                        signal_tdata_name: str, signal_tvalid_name: str, signal_tready_name: str,
                        signal_clk_name: str
    ) -> np.ndarray:
    """
    Extracts the signal from the VCD file and returns it as a numpy array.
    This function uses all the signal's bits and convert them into an integer.
    """
    vcd = VCDVCD(vcdfile)
    signal_tdata = vcd[signal_tdata_name]
    signal_tvalid = vcd[signal_tvalid_name]
    signal_tready = vcd[signal_tready_name]
    signal_clk = vcd[signal_clk_name]
    signal_values = []
    
    # Get the clock edges (rising edges)
    clk_rising_edges = [t for t, v in signal_clk.tv if int(v) == 1]
    
    # Iterate over the signal data
    for timestamp in clk_rising_edges:
        # Check if the signal is valid and ready
        if signal_tvalid[timestamp] == 'x'or int(signal_tvalid[timestamp]) == 0 or int(signal_tready[timestamp] == 0):
            continue
        
        # Sample the signal at the clock edge
        tdata_value = signal_tdata[timestamp]
        
        # Get the tdata value
        tdata_value = signal_tdata[timestamp]
        
        # Append the value to the list
        signal_values.append(int(tdata_value, 2))  # Ensure it's 32 bits long
        
    return np.array(signal_values)
    

########
# RMSE #
########
def rmse(signal: np.ndarray, reference: np.ndarray, is_complex: bool) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between the input signal and the reference signal.
    """
    if is_complex:
        error = np.abs(signal - reference)
    else:
        error = signal - reference
    rmse_value = np.sqrt(np.mean(error**2))
    return rmse_value


##############
# Convertion #
##############
def check_format(signal: np.ndarray, is_complex: bool, is_int: bool, bus_size_by_components: int) -> None:
    """
    Checks the format of the input signal and prints the results.
    """
    if is_complex:
        real_part = signal.real
        imag_part = signal.imag
        
        # Check that there are no decimal values
        if is_int:
            # Check that there are no decimal values
            assert np.all(np.mod(real_part, 1) == 0), "Real part has decimal values"
            assert np.all(np.mod(imag_part, 1) == 0), "Imaginary part has decimal values"

        # Check that the real part is within the range
        min_val = -2**(bus_size_by_components - 1)
        max_val = 2**(bus_size_by_components - 1) - 1
        assert np.all(real_part >= min_val) and np.all(real_part <= max_val), "Real part is out of range"
        assert np.all(imag_part >= min_val) and np.all(imag_part <= max_val), "Imaginary part is out of range"
    
    else:
        # Check that there are no decimal values
        if is_int:
            assert np.all(np.mod(signal, 1) == 0), "Signal has decimal values" 
        
        # Check that the signal is within the range
        min_val = -2**(bus_size_by_components - 1)
        max_val = 2**(bus_size_by_components - 1) - 1
        assert np.all(signal >= min_val) and np.all(signal <= max_val), "Signal is out of range"
        

def convert_to_sc16(complex_signal: np.ndarray) -> np.ndarray:
    """
    Converts the input signal to SC16 format.
    """
    scale_factor = SC16_MAX
    real_part = np.clip(ROUND(complex_signal.real * scale_factor), SC16_MIN, SC16_MAX)
    imag_part = np.clip(ROUND(complex_signal.imag * scale_factor), SC16_MIN, SC16_MAX)
    quatized = (real_part + 1j * imag_part)
    return quatized


def clip(signal: np.ndarray, min_val: float, max_val: float, is_complex: bool) -> np.ndarray:
    """
    Clips the input signal to the specified range.
    """
    if is_complex:
        real_part = np.clip(signal.real, min_val, max_val)
        imag_part = np.clip(signal.imag, min_val, max_val)
        clipped_signal = real_part + 1j * imag_part
    else:
        clipped_signal = np.clip(signal, min_val, max_val)
    return clipped_signal

def trunc(signal: np.ndarray, from_nbits: int, to_nbits: int, is_complex: bool) -> np.ndarray:
    """
    Truncates the input signal by keeping only the to_nbits most significant bits
    from a signal represented on from_nbits bits.
    """
    # # Calculate how many bits to truncate
    bits_to_truncate = (from_nbits) - (to_nbits)
    if bits_to_truncate <= 0:
        return signal  # No truncation needed
        
    # Scale factor to keep only MSB
    scale_factor = 2**(bits_to_truncate)

    if is_complex:               
        real_part = np.floor(signal.real / (scale_factor))
        imag_part = np.floor(signal.imag / (scale_factor))
        truncated_signal = real_part + 1j * imag_part
    else:
        truncated_signal = np.floor(signal / scale_factor)
    return truncated_signal

def scale(signal: np.ndarray, scale_factor: float, is_complex: bool) -> np.ndarray:
    """
    Scales the input signal by the specified scale factor.
    """
    if scale_factor == 0:
        raise ValueError("Scale factor cannot be zero.")
    if scale_factor is None:
        scale_factor = np.max(np.abs(signal))
        
    if is_complex:
        real_part = signal.real / scale_factor
        imag_part = signal.imag / scale_factor
        scaled_signal = real_part + 1j * imag_part
    else:
        scaled_signal = signal / scale_factor
    return scaled_signal


###############
# Computation #
###############
def processing(input_signal: np.ndarray, fft_size: int, M: int) -> None:
    """
    Computes the P(d) metrics for the input signal, as in hardware.
    """
    
    def moving_sum(signal: np.ndarray, L: int) -> np.ndarray:
        """
        Computes the moving sum of the input signal with a window size of L.
        """
        result = np.zeros_like(signal)
        if np.iscomplexobj(signal):
            # Handle complex signals by operating on real and imaginary parts separately
            real_result = np.zeros(len(signal))
            imag_result = np.zeros(len(signal))
            
            # Process real part
            signal_real = signal.real
            for i in range(len(signal)):
                if i < L:
                    real_result[i] = np.sum(signal_real[:i])
                else:
                    real_result[i] = np.sum(signal_real[i-L:i])
            
            # Process imaginary part
            signal_imag = signal.imag
            for i in range(len(signal)):
                if i < L:
                    imag_result[i] = np.sum(signal_imag[:i])
                else:
                    imag_result[i] = np.sum(signal_imag[i-L:i])
            
            result = real_result + 1j * imag_result
        else:
            # For real signals, use the original algorithm
            for i in range(len(signal)):
                if i < L:
                    result[i] = np.sum(signal[:i])
                else:
                    result[i] = np.sum(signal[i-L:i])
        
        return result
    
    # Input signal
    L = int((fft_size / 2) * M)
    r_d = input_signal
    
    ## SIGNAL DELAY PATH
    # Delay the r signal by L
    r_dL = np.roll(r_d, L); r_dL[:L] = 0  # Delay by L samples
    
    # Conjugate the delayed signal
    r_dL = np.conj(r_dL)

    # Multiply the signals
    r_mul = r_d * r_dL    
    signals["r_mul_32_fp"] = r_mul
    signals["r_mul_16_clip"] = clip(r_mul, SC16_MIN, SC16_MAX, is_complex=True)
    signals["r_mul_16_trunc"] = trunc(r_mul, 31, 15, is_complex=True)
    
    # Pass the full precision signal through the moving sum
    r_sum = moving_sum(signals["r_mul_32_fp"], L)
    signals["r_sum_41_fp"] = r_sum
    signals["r_sum_16_clip"] = clip(r_sum, SC16_MIN, SC16_MAX, is_complex=True)
    signals["r_sum_16_trunc"] = trunc(r_sum, 41, 15, is_complex=True)
    r_sum_clip = moving_sum(signals["r_mul_16_clip"], L)
    signals["r_sum_clip_25_fp"] = r_sum_clip
    signals["r_sum_clip_16_clip"] = clip(r_sum_clip, SC16_MIN, SC16_MAX, is_complex=True)
    signals["r_sum_clip_16_trunc"] = trunc(r_sum_clip, 25, 15, is_complex=True)
    r_sum_trunc = moving_sum(signals["r_mul_16_trunc"], L)
    signals["r_sum_trunc_25_fp"] = r_sum_trunc
    signals["r_sum_trunc_16_clip"] = clip(r_sum_trunc, SC16_MIN, SC16_MAX, is_complex=True)
    signals["r_sum_trunc_16_trunc"] = trunc(r_sum_trunc, 25, 15, is_complex=True)
    
    ## SIGNAL POWER PATH
    # Compute the power of the signal
    r_abs = np.abs(r_d)**2
    r_sum_abs = moving_sum(r_abs, L)    
    signals["r_sum_abs_41_fp"] = r_sum_abs
    signals["r_sum_abs_16_clip"] = clip(r_sum_abs, SC16_MIN, SC16_MAX, is_complex=False)
    signals["r_sum_abs_16_trunc"] = trunc(r_sum_abs, 41, 15, is_complex=False)
    
    ## FINAL METRIC - full precision
    #! We select only the r_sum_41_fp and the r_sum_abs_41_fp signals for the final metric analysis
    p2 = np.round(np.abs(signals["r_sum_41_fp"])**2)
    signals["p2_fp"] = p2
    r2 = (signals["r_sum_abs_41_fp"]**2)
    signals["r2_fp"] = r2
    
    # Compute the final metric
    safe_denom = np.where(r2 == 0, 1, r2)  # Avoid division by zero
    m = p2 / safe_denom
    signals["m_fp"] = m
    
    ## FINAL METRIC - truncate path
    #! We select only the r_sum_16_trunc and the r_sum_abs_16_trunc signals for the final metric analysis
    p2_d = np.round(np.abs(signals["r_sum_16_trunc"])**2)
    signals["p2_best"] = p2_d
    r2_d = (signals["r_sum_abs_16_trunc"]**2)
    signals["r2_best"] = r2_d
    
    # Compute the final metric
    safe_denom = np.where(r2_d == 0, 1, r2_d)  # Avoid division by zero
    m_d = p2_d / safe_denom
    signals["m_best"] = m_d
    
      

########
# Main #
########
if __name__ == "__main__":

    print('-----------------Signal creation-----------------')
    frame = Frame(K=1024, CP=128, CP_preamble=128, M=1, N=1, preamble_mod="BPSK", payload_mod="QPSK", verbose=True, random_seed=3572128029)
    mod = Modulation(frame)
    mod.modulate()
    channel = Channel(frame, verbose=True, random_seed=199317234)
    channel.add_sto(0)
    channel.add_noise(np.inf)
    frame.load_txt("signal_K1024_CP128_CPp128_M1_N1_preambleBPSK_payloadQPSK_usrp_recv.txt", ignore_zero=False, crop=False)
    print("--------------------------------------------------\n\n")
    
    
    print("------------------Signal analysis-----------------")
    # Transform the signal to SC16 format and check loss
    r = frame.get_frame()
    r_sc16 = convert_to_sc16(r)
    check_format(r_sc16, is_complex=True, is_int=True, bus_size_by_components=16)
    
    # Process the signal
    processing(r_sc16, frame.K, frame.M)
    
    # Get VCD signals
    vcd_file = "../../../rfnoc_block_schmidl_cox_tb.vcd"
    signal_clk_name = "rfnoc_block_schmidl_cox_tb.dut.axis_data_clk"
    
    # P
    signal_tdata_name = "rfnoc_block_schmidl_cox_tb.dut.mc0.r10_tdata[31:0]"
    signal_tvalid_name = "rfnoc_block_schmidl_cox_tb.dut.mc0.r10_tvalid"
    signal_tready_name = "rfnoc_block_schmidl_cox_tb.dut.mc0.r10_tready"
    signal_tdata = get_signal_from_vcd(vcd_file, signal_tdata_name, signal_tvalid_name, signal_tready_name, signal_clk_name)
    signal_tdata = np.pad(signal_tdata, (0, len(signals["p2_fp"]) - len(signal_tdata)), mode='constant', constant_values=0)
    signals["p2_vcd"] = signal_tdata
    
    # R
    signal_tdata_name = "rfnoc_block_schmidl_cox_tb.dut.mc0.r11_tdata[31:0]"
    signal_tvalid_name = "rfnoc_block_schmidl_cox_tb.dut.mc0.r11_tvalid"
    signal_tready_name = "rfnoc_block_schmidl_cox_tb.dut.mc0.r11_tready"
    signal_tdata = get_signal_from_vcd(vcd_file, signal_tdata_name, signal_tvalid_name, signal_tready_name, signal_clk_name)
    signal_tdata = np.pad(signal_tdata, (0, len(signals["r2_fp"]) - len(signal_tdata)), mode='constant', constant_values=0)
    signals["r2_vcd"] = signal_tdata
    
    # M
    signal_tdata_name = "rfnoc_block_schmidl_cox_tb.dut.mc0.r12_tdata[31:0]"
    signal_tvalid_name = "rfnoc_block_schmidl_cox_tb.dut.mc0.r12_tvalid"
    signal_tready_name = "rfnoc_block_schmidl_cox_tb.dut.mc0.r12_tready"
    signal_tdata = get_signal_from_vcd(vcd_file, signal_tdata_name, signal_tvalid_name, signal_tready_name, signal_clk_name)
    signal_tdata = np.pad(signal_tdata, (0, len(signals["m_fp"]) - len(signal_tdata)), mode='constant', constant_values=0)
    signals["m_vcd"] = signal_tdata
    
    print("--------------------------------------------------\n\n")
    
    
    print("-----------After multiplication analysis----------")
    # Check the format of the processed signal
    check_format(signals["r_mul_32_fp"], is_complex=True, is_int=False, bus_size_by_components=32)
    check_format(signals["r_mul_16_clip"], is_complex=True, is_int=True, bus_size_by_components=16)
    check_format(signals["r_mul_16_trunc"], is_complex=True, is_int=True, bus_size_by_components=16)
    
    # Scale the signals
    signals_scaled["r_mul_32_fp"] = scale(signals["r_mul_32_fp"], 2**31, is_complex=True)
    signals_scaled["r_mul_16_clip"] = scale(signals["r_mul_16_clip"], 2**15, is_complex=True)
    signals_scaled["r_mul_16_trunc"] = scale(signals["r_mul_16_trunc"], 2**15, is_complex=True)
    
    # Normalize the signals
    signals_normalized["r_mul_32_fp"] = scale(signals["r_mul_32_fp"], None, is_complex=True)
    signals_normalized["r_mul_16_clip"] = scale(signals["r_mul_16_clip"], None, is_complex=True)
    signals_normalized["r_mul_16_trunc"] = scale(signals["r_mul_16_trunc"], None, is_complex=True)
    
    # Compute the RMSE between the signals
    rmse_original = [
        rmse(signals["r_mul_32_fp"], signals["r_mul_32_fp"], is_complex=True),
        rmse(signals["r_mul_32_fp"], signals["r_mul_16_clip"], is_complex=True),
        rmse(signals["r_mul_32_fp"], signals["r_mul_16_trunc"], is_complex=True)
    ]
    rmse_scaled = [
        rmse(signals_scaled["r_mul_32_fp"], signals_scaled["r_mul_32_fp"], is_complex=True),
        rmse(signals_scaled["r_mul_32_fp"], signals_scaled["r_mul_16_clip"], is_complex=True),
        rmse(signals_scaled["r_mul_32_fp"], signals_scaled["r_mul_16_trunc"], is_complex=True)
    ]
    rmse_normalized = [
        rmse(signals_normalized["r_mul_32_fp"], signals_normalized["r_mul_32_fp"], is_complex=True),
        rmse(signals_normalized["r_mul_32_fp"], signals_normalized["r_mul_16_clip"], is_complex=True),
        rmse(signals_normalized["r_mul_32_fp"], signals_normalized["r_mul_16_trunc"], is_complex=True)
    ]
    new_data = {
        "Signal": ["r_mul_32_fp", "r_mul_16_clip", "r_mul_16_trunc"],
        "RMSE (original)": rmse_original,
        "RMSE (Scaled)": rmse_scaled,
        "RMSE (Normalized)": rmse_normalized
    }
    rmses = pd.DataFrame(new_data)
    rmses.to_csv("results/signals_after_multiplication_rmses.csv", index=False)
    print(tabulate(rmses.map(lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x), headers='keys', tablefmt='pretty', showindex=False))
    
    # Plot the signals
    plots_config = [("scaled", signals_scaled), ("normalized", signals_normalized)]
    for name, signal in plots_config:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), num=f"Signals after multiplication ({name})")
        fig.suptitle(f"Signals after multiplication ({name})", fontsize=14, fontweight='bold')
        axs[0].plot(np.real(signal['r_mul_16_clip']), label="clipped (16 bits)")
        axs[0].plot(np.real(signal['r_mul_16_trunc']), label="truncated (16 bits)")
        axs[0].plot(np.real(signal['r_mul_32_fp']), label="full precision (32 bits)")
        axs[0].legend(loc="upper right")
        axs[0].grid()
        axs[0].set_ylabel("Real part")
        axs[1].plot(np.imag(signal['r_mul_16_clip']), label="clipped (16 bits)")
        axs[1].plot(np.imag(signal['r_mul_16_trunc']), label="truncated (16 bits)")
        axs[1].plot(np.imag(signal['r_mul_32_fp']), label="full precision (32 bits)")
        axs[1].legend(loc="upper right")
        axs[1].grid()
        axs[1].set_xlabel("Sample index [n]")
        axs[1].set_ylabel("Imaginary part")
        plt.tight_layout()
        plt.savefig(f"results/signals_after_multiplication_plot_{name}.pdf")
        plt.close()
    print("--------------------------------------------------\n\n")
    
    
    print("------------After moving sum analysis-------------")
    # Check the format of the processed signal
    check_format(signals["r_sum_41_fp"], is_complex=True, is_int=False, bus_size_by_components=41)
    check_format(signals["r_sum_16_clip"], is_complex=True, is_int=True, bus_size_by_components=16)
    check_format(signals["r_sum_16_trunc"], is_complex=True, is_int=True, bus_size_by_components=16)
    check_format(signals["r_sum_clip_25_fp"], is_complex=True, is_int=False, bus_size_by_components=25)
    check_format(signals["r_sum_clip_16_clip"], is_complex=True, is_int=True, bus_size_by_components=16)
    check_format(signals["r_sum_clip_16_trunc"], is_complex=True, is_int=True, bus_size_by_components=16)
    check_format(signals["r_sum_trunc_25_fp"], is_complex=True, is_int=False, bus_size_by_components=25)
    check_format(signals["r_sum_trunc_16_clip"], is_complex=True, is_int=True, bus_size_by_components=16)
    check_format(signals["r_sum_trunc_16_trunc"], is_complex=True, is_int=True, bus_size_by_components=16)
    
    # Scale the signals
    signals_scaled["r_sum_41_fp"] = scale(signals["r_sum_41_fp"], 2**40, is_complex=True)
    signals_scaled["r_sum_16_clip"] = scale(signals["r_sum_16_clip"], 2**15, is_complex=True)
    signals_scaled["r_sum_16_trunc"] = scale(signals["r_sum_16_trunc"], 2**15, is_complex=True)
    signals_scaled["r_sum_clip_25_fp"] = scale(signals["r_sum_clip_25_fp"], 2**24, is_complex=True)
    signals_scaled["r_sum_clip_16_clip"] = scale(signals["r_sum_clip_16_clip"], 2**15, is_complex=True)
    signals_scaled["r_sum_clip_16_trunc"] = scale(signals["r_sum_clip_16_trunc"], 2**15, is_complex=True)
    signals_scaled["r_sum_trunc_25_fp"] = scale(signals["r_sum_trunc_25_fp"], 2**24, is_complex=True)
    signals_scaled["r_sum_trunc_16_clip"] = scale(signals["r_sum_trunc_16_clip"], 2**15, is_complex=True)
    signals_scaled["r_sum_trunc_16_trunc"] = scale(signals["r_sum_trunc_16_trunc"], 2**15, is_complex=True)
    
    # Normalize the signals
    signals_normalized["r_sum_41_fp"] = scale(signals["r_sum_41_fp"], None, is_complex=True)
    signals_normalized["r_sum_16_clip"] = scale(signals["r_sum_16_clip"], None, is_complex=True)
    signals_normalized["r_sum_16_trunc"] = scale(signals["r_sum_16_trunc"], None, is_complex=True)
    signals_normalized["r_sum_clip_25_fp"] = scale(signals["r_sum_clip_25_fp"], None, is_complex=True)
    signals_normalized["r_sum_clip_16_clip"] = scale(signals["r_sum_clip_16_clip"], None, is_complex=True)
    signals_normalized["r_sum_clip_16_trunc"] = scale(signals["r_sum_clip_16_trunc"], None, is_complex=True)
    signals_normalized["r_sum_trunc_25_fp"] = scale(signals["r_sum_trunc_25_fp"], None, is_complex=True)
    signals_normalized["r_sum_trunc_16_clip"] = scale(signals["r_sum_trunc_16_clip"], None, is_complex=True)
    signals_normalized["r_sum_trunc_16_trunc"] = scale(signals["r_sum_trunc_16_trunc"], None, is_complex=True)
    
    # Compute the RMSE between the signals
    rmse_original = [
        rmse(signals["r_sum_41_fp"], signals["r_sum_41_fp"], is_complex=True),
        rmse(signals["r_sum_41_fp"], signals["r_sum_16_clip"], is_complex=True),
        rmse(signals["r_sum_41_fp"], signals["r_sum_16_trunc"], is_complex=True),
        rmse(signals["r_sum_41_fp"], signals["r_sum_clip_25_fp"], is_complex=True),
        rmse(signals["r_sum_41_fp"], signals["r_sum_clip_16_clip"], is_complex=True),
        rmse(signals["r_sum_41_fp"], signals["r_sum_clip_16_trunc"], is_complex=True),
        rmse(signals["r_sum_41_fp"], signals["r_sum_trunc_25_fp"], is_complex=True),
        rmse(signals["r_sum_41_fp"], signals["r_sum_trunc_16_clip"], is_complex=True),
        rmse(signals["r_sum_41_fp"], signals["r_sum_trunc_16_trunc"], is_complex=True)
    ]
    rmse_scaled = [
        rmse(signals_scaled["r_sum_41_fp"], signals_scaled["r_sum_41_fp"], is_complex=True),
        rmse(signals_scaled["r_sum_41_fp"], signals_scaled["r_sum_16_clip"], is_complex=True),
        rmse(signals_scaled["r_sum_41_fp"], signals_scaled["r_sum_16_trunc"], is_complex=True),
        rmse(signals_scaled["r_sum_41_fp"], signals_scaled["r_sum_clip_25_fp"], is_complex=True),
        rmse(signals_scaled["r_sum_41_fp"], signals_scaled["r_sum_clip_16_clip"], is_complex=True),
        rmse(signals_scaled["r_sum_41_fp"], signals_scaled["r_sum_clip_16_trunc"], is_complex=True),
        rmse(signals_scaled["r_sum_41_fp"], signals_scaled["r_sum_trunc_25_fp"], is_complex=True),
        rmse(signals_scaled["r_sum_41_fp"], signals_scaled["r_sum_trunc_16_clip"], is_complex=True),
        rmse(signals_scaled["r_sum_41_fp"], signals_scaled["r_sum_trunc_16_trunc"], is_complex=True)
    ]
    rmse_normalized = [
        rmse(signals_normalized["r_sum_41_fp"], signals_normalized["r_sum_41_fp"], is_complex=True),
        rmse(signals_normalized["r_sum_41_fp"], signals_normalized["r_sum_16_clip"], is_complex=True),
        rmse(signals_normalized["r_sum_41_fp"], signals_normalized["r_sum_16_trunc"], is_complex=True),
        rmse(signals_normalized["r_sum_41_fp"], signals_normalized["r_sum_clip_25_fp"], is_complex=True),
        rmse(signals_normalized["r_sum_41_fp"], signals_normalized["r_sum_clip_16_clip"], is_complex=True),
        rmse(signals_normalized["r_sum_41_fp"], signals_normalized["r_sum_clip_16_trunc"], is_complex=True),
        rmse(signals_normalized["r_sum_41_fp"], signals_normalized["r_sum_trunc_25_fp"], is_complex=True),
        rmse(signals_normalized["r_sum_41_fp"], signals_normalized["r_sum_trunc_16_clip"], is_complex=True),
        rmse(signals_normalized["r_sum_41_fp"], signals_normalized["r_sum_trunc_16_trunc"], is_complex=True)
    ]
    new_data = {
        "Signal": ["r_sum_41_fp", "r_sum_16_clip", "r_sum_16_trunc", "r_sum_clip_25_fp", "r_sum_clip_16_clip", "r_sum_clip_16_trunc", "r_sum_trunc_25_fp", "r_sum_trunc_16_clip", "r_sum_trunc_16_trunc"],
        "RMSE (original)": rmse_original,
        "RMSE (Scaled)": rmse_scaled,
        "RMSE (Normalized)": rmse_normalized
    }
    rmses = pd.DataFrame(new_data)
    rmses.to_csv("results/signals_after_moving_sum_rmses.csv", index=False)
    print(tabulate(rmses.map(lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x), headers='keys', tablefmt='pretty', showindex=False))
    
    # Plot the signals
    plots_config = [("scaled", signals_scaled), ("normalized", signals_normalized)]
    for name, signal in plots_config:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), num=f"Signals after moving sum - fp in ({name})")
        fig.suptitle(f"Signals after moving sum - fp in ({name})", fontsize=14, fontweight='bold')
        axs[0].plot(np.real(signal['r_sum_16_clip']), label="clipped (16 bits)")
        axs[0].plot(np.real(signal['r_sum_16_trunc']), label="truncated (16 bits)")
        axs[0].plot(np.real(signal['r_sum_41_fp']), label="full precision (41 bits)")
        axs[0].legend(loc="upper right")
        axs[0].grid()
        axs[0].set_ylabel("Real part")
        axs[1].plot(np.imag(signal['r_sum_16_clip']), label="clipped (16 bits)")
        axs[1].plot(np.imag(signal['r_sum_16_trunc']), label="truncated (16 bits)")
        axs[1].plot(np.imag(signal['r_sum_41_fp']), label="full precision (41 bits)")
        axs[1].legend(loc="upper right")
        axs[1].grid()
        axs[1].set_xlabel("Sample index [n]")
        axs[1].set_ylabel("Imaginary part")
        plt.tight_layout()
        plt.savefig(f"results/signals_after_moving_sum_fp_plot_{name}.pdf")
        plt.close()
        
        # Plot the signals
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), num=f"Signals after moving sum - clip in ({name})")
        fig.suptitle(f"Signals after moving sum - clip in ({name})", fontsize=14, fontweight='bold')
        axs[0].plot(np.real(signal['r_sum_clip_16_clip']), label="clipped (16 bits)")
        axs[0].plot(np.real(signal['r_sum_clip_16_trunc']), label="truncated (16 bits)")
        axs[0].plot(np.real(signal['r_sum_clip_25_fp']), label="full precision (25 bits)")
        axs[0].plot(np.real(signal['r_sum_41_fp']), label="full precision (41 bits)")
        axs[0].legend(loc="upper right")
        axs[0].grid()
        axs[0].set_ylabel("Real part")
        axs[1].plot(np.imag(signal['r_sum_clip_16_clip']), label="clipped (16 bits)")
        axs[1].plot(np.imag(signal['r_sum_clip_16_trunc']), label="truncated (16 bits)")
        axs[1].plot(np.imag(signal['r_sum_clip_25_fp']), label="full precision (25 bits)")
        axs[1].plot(np.imag(signal['r_sum_41_fp']), label="full precision (41 bits)")
        axs[1].legend(loc="upper right")
        axs[1].grid()
        axs[1].set_xlabel("Sample index [n]")
        axs[1].set_ylabel("Imaginary part")
        plt.tight_layout()
        plt.savefig(f"results/signals_after_moving_sum_clip_plot_{name}.pdf")
        plt.close()
        
        # Plot the signals
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), num=f"Signals after moving sum - trunc in ({name})")
        fig.suptitle(f"Signals after moving sum - trunc in ({name})", fontsize=14, fontweight='bold')
        axs[0].plot(np.real(signal['r_sum_trunc_16_clip']), label="clipped (16 bits)")
        axs[0].plot(np.real(signal['r_sum_trunc_16_trunc']), label="truncated (16 bits)")
        axs[0].plot(np.real(signal['r_sum_trunc_25_fp']), label="full precision (25 bits)")
        axs[0].plot(np.real(signal['r_sum_41_fp']), label="full precision (41 bits)")
        axs[0].legend(loc="upper right")
        axs[0].grid()
        axs[0].set_ylabel("Real part")
        axs[1].plot(np.imag(signal['r_sum_trunc_16_clip']), label="clipped (16 bits)")
        axs[1].plot(np.imag(signal['r_sum_trunc_16_trunc']), label="truncated (16 bits)")
        axs[1].plot(np.imag(signal['r_sum_trunc_25_fp']), label="full precision (25 bits)")
        axs[1].plot(np.imag(signal['r_sum_41_fp']), label="full precision (41 bits)")
        axs[1].legend(loc="upper right")
        axs[1].grid()
        axs[1].set_xlabel("Sample index [n]")
        axs[1].set_ylabel("Imaginary part")
        plt.tight_layout()
        plt.savefig(f"results/signals_after_moving_sum_trunc_plot_{name}.pdf")
        plt.close() 
    print("--------------------------------------------------\n\n")
    
    
    print("---------------Energy analysis-----------------")
    # Check the format of the processed signal
    check_format(signals["r_sum_abs_41_fp"], is_complex=False, is_int=False, bus_size_by_components=41)
    check_format(signals["r_sum_abs_16_clip"], is_complex=False, is_int=True, bus_size_by_components=16)
    check_format(signals["r_sum_abs_16_trunc"], is_complex=False, is_int=True, bus_size_by_components=16)
    
    # Scale the signals
    signals_scaled["r_sum_abs_41_fp"] = scale(signals["r_sum_abs_41_fp"], 2**41, is_complex=False)
    signals_scaled["r_sum_abs_16_clip"] = scale(signals["r_sum_abs_16_clip"], 2**15, is_complex=False)
    signals_scaled["r_sum_abs_16_trunc"] = scale(signals["r_sum_abs_16_trunc"], 2**15, is_complex=False)
    
    # Normalize the signals
    signals_normalized["r_sum_abs_41_fp"] = scale(signals["r_sum_abs_41_fp"], None, is_complex=False)
    signals_normalized["r_sum_abs_16_clip"] = scale(signals["r_sum_abs_16_clip"], None, is_complex=False)
    signals_normalized["r_sum_abs_16_trunc"] = scale(signals["r_sum_abs_16_trunc"], None, is_complex=False)
    
    # Compute the RMSE between the signals
    rmse_original = [
        rmse(signals["r_sum_abs_41_fp"], signals["r_sum_abs_41_fp"], is_complex=False),
        rmse(signals["r_sum_abs_41_fp"], signals["r_sum_abs_16_clip"], is_complex=False),
        rmse(signals["r_sum_abs_41_fp"], signals["r_sum_abs_16_trunc"], is_complex=False)
    ]
    rmse_scaled = [
        rmse(signals_scaled["r_sum_abs_41_fp"], signals_scaled["r_sum_abs_41_fp"], is_complex=False),
        rmse(signals_scaled["r_sum_abs_41_fp"], signals_scaled["r_sum_abs_16_clip"], is_complex=False),
        rmse(signals_scaled["r_sum_abs_41_fp"], signals_scaled["r_sum_abs_16_trunc"], is_complex=False)
    ]
    rmse_normalized = [
        rmse(signals_normalized["r_sum_abs_41_fp"], signals_normalized["r_sum_abs_41_fp"], is_complex=False),
        rmse(signals_normalized["r_sum_abs_41_fp"], signals_normalized["r_sum_abs_16_clip"], is_complex=False),
        rmse(signals_normalized["r_sum_abs_41_fp"], signals_normalized["r_sum_abs_16_trunc"], is_complex=False)
    ]
    new_data = {
        "Signal": ["r_sum_abs_41_fp", "r_sum_abs_16_clip", "r_sum_abs_16_trunc"],
        "RMSE (original)": rmse_original,
        "RMSE (Scaled)": rmse_scaled,
        "RMSE (Normalized)": rmse_normalized
    }
    rmses = pd.DataFrame(new_data)
    rmses.to_csv("results/signals_after_energy_rmses.csv", index=False)
    print(tabulate(rmses.map(lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x), headers='keys', tablefmt='pretty', showindex=False))
    
    # Plot the signals
    plots_config = [("scaled", signals_scaled), ("normalized", signals_normalized)]
    for name, signal in plots_config:
        fig, axs = plt.subplots(1, 1, figsize=(12, 8), num=f"Signals after energy - fp in ({name})")
        fig.suptitle(f"Signals after energy - fp in ({name})", fontsize=14, fontweight='bold')
        axs.plot(signal['r_sum_abs_16_clip'], label="clipped (16 bits)")
        axs.plot(signal['r_sum_abs_16_trunc'], label="truncated (16 bits)")
        axs.plot(signal['r_sum_abs_41_fp'], label="full precision (41 bits)")
        axs.legend(loc="upper right")
        axs.grid()
        axs.set_xlabel("Sample index [n]")
        axs.set_ylabel("Signal")
        plt.tight_layout()
        plt.savefig(f"results/signals_after_energy_fp_plot_{name}.pdf")
        plt.close()
    print("--------------------------------------------------\n\n")


    print("---------------|P(d)|^2 metric analysis-----------------")
    # Check the format of the processed signal
    # ?This should be done
    
    # Scale the signals
    signals_scaled["p2_fp"] = scale(signals["p2_fp"], 2**(41 * 2), is_complex=False)
    signals_scaled["p2_best"] = scale(signals["p2_best"], 2**31, is_complex=False)
    signals_scaled["p2_vcd"] = scale(signals["p2_vcd"], 2**31, is_complex=False)
    
    # Normalize the signals
    signals_normalized["p2_fp"] = scale(signals["p2_fp"], None, is_complex=False)
    signals_normalized["p2_best"] = scale(signals["p2_best"], None, is_complex=False)
    signals_normalized["p2_vcd"] = scale(signals["p2_vcd"], None, is_complex=False)
    
    # Compute the RMSE between the signals
    rmse_original = [
        rmse(signals["p2_fp"], signals["p2_fp"], is_complex=False),
        rmse(signals["p2_fp"], signals["p2_best"], is_complex=False),
        rmse(signals["p2_fp"], signals["p2_vcd"], is_complex=False),
    ]
    rmse_scaled = [
        rmse(signals_scaled["p2_fp"], signals_scaled["p2_fp"], is_complex=False),
        rmse(signals_scaled["p2_fp"], signals_scaled["p2_best"], is_complex=False),
        rmse(signals_scaled["p2_fp"], signals_scaled["p2_vcd"], is_complex=False)
    ]
    rmse_normalized = [
        rmse(signals_normalized["p2_fp"], signals_normalized["p2_fp"], is_complex=False),
        rmse(signals_normalized["p2_fp"], signals_normalized["p2_best"], is_complex=False),
        rmse(signals_normalized["p2_fp"], signals_normalized["p2_vcd"], is_complex=False)
    ]
    
    new_data = {
        "Signal": ["p2_fp", "p2_best", "p2_vcd"],
        "RMSE (original)": rmse_original,
        "RMSE (Scaled)": rmse_scaled,
        "RMSE (Normalized)": rmse_normalized
    }
    rmses = pd.DataFrame(new_data)
    rmses.to_csv("results/signals_p2_metric_rmses.csv", index=False)
    print(tabulate(rmses.map(lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x), headers='keys', tablefmt='pretty', showindex=False))
    
    # Plot the signals
    plots_config = [("scaled", signals_scaled), ("normalized", signals_normalized)]
    for name, signal in plots_config:
        fig, axs = plt.subplots(1, 1, figsize=(12, 8), num=f"Signals after final metric ({name})")
        fig.suptitle(f"Signals after final metric ({name})", fontsize=14, fontweight='bold')
        axs.plot(signal['p2_best'], label="$|P(d)|^2_{best}$")
        axs.plot(signal['p2_vcd'], label="$|P(d)|^2_{vcd}$")
        axs.plot(signal['p2_fp'], label="$|P(d)|^2_{fp}$")
        axs.legend(loc="upper right")
        axs.grid()
        axs.set_xlabel("Sample index [n]")
        axs.set_ylabel("Signal")
        plt.tight_layout()
        plt.savefig(f"results/signals_p2_plot_{name}.pdf")
        plt.close()
    print("--------------------------------------------------\n\n")
    
    
    print("---------------(R(d))^2 metric analysis-----------------")
    # Check the format of the processed signal
    # ?This should be done
    
    # Scale the signals
    signals_scaled["r2_fp"] = scale(signals["r2_fp"], 2**(41 * 2), is_complex=False)
    signals_scaled["r2_best"] = scale(signals["r2_best"], 2**31, is_complex=False)
    signals_scaled["r2_vcd"] = scale(signals["r2_vcd"], 2**31, is_complex=False)
    
    # Normalize the signals
    signals_normalized["r2_fp"] = scale(signals["r2_fp"], None, is_complex=False)
    signals_normalized["r2_best"] = scale(signals["r2_best"], None, is_complex=False)
    signals_normalized["r2_vcd"] = scale(signals["r2_vcd"], None, is_complex=False)
    
    # Compute the RMSE between the signals
    rmse_original = [
        rmse(signals["r2_fp"], signals["r2_fp"], is_complex=False),
        rmse(signals["r2_fp"], signals["r2_best"], is_complex=False),
        rmse(signals["r2_fp"], signals["r2_vcd"], is_complex=False),
    ]
    rmse_scaled = [
        rmse(signals_scaled["r2_fp"], signals_scaled["r2_fp"], is_complex=False),
        rmse(signals_scaled["r2_fp"], signals_scaled["r2_best"], is_complex=False),
        rmse(signals_scaled["r2_fp"], signals_scaled["r2_vcd"], is_complex=False)
    ]
    rmse_normalized = [
        rmse(signals_normalized["r2_fp"], signals_normalized["r2_fp"], is_complex=False),
        rmse(signals_normalized["r2_fp"], signals_normalized["r2_best"], is_complex=False),
        rmse(signals_normalized["r2_fp"], signals_normalized["r2_vcd"], is_complex=False)
    ]
    
    new_data = {
        "Signal": ["r2_fp", "r2_best", "r2_vcd"],
        "RMSE (original)": rmse_original,
        "RMSE (Scaled)": rmse_scaled,
        "RMSE (Normalized)": rmse_normalized
    }
    rmses = pd.DataFrame(new_data)
    rmses.to_csv("results/signals_after_r2_rmses.csv", index=False)
    print(tabulate(rmses.map(lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x), headers='keys', tablefmt='pretty', showindex=False))
    
    # Plot the signals
    plots_config = [("scaled", signals_scaled), ("normalized", signals_normalized)]
    for name, signal in plots_config:
        fig, axs = plt.subplots(1, 1, figsize=(12, 8), num=f"Signals after final metric ({name})")
        fig.suptitle(f"Signals after final metric ({name})", fontsize=14, fontweight='bold')
        axs.plot(signal['r2_best'], label="$(R(d))^2_{best}$")
        axs.plot(signal['r2_vcd'], label="$(R(d))^2_{vcd}$")
        axs.plot(signal['r2_fp'], label="$(R(d))^2_{fp}$")
        axs.legend(loc="upper right")
        axs.grid()
        axs.set_xlabel("Sample index [n]")
        axs.set_ylabel("Signal")
        plt.tight_layout()
        plt.savefig(f"results/signals_r2_plot_{name}.pdf")
        plt.close()
    print("--------------------------------------------------\n\n")
    
    
    print("---------------Final metric analysis-----------------")
    # Check the format of the processed signal
    # ?This should be done
    
    # Scale the signals
    signals_scaled["m_fp"] = scale(signals["m_fp"], 1, is_complex=False)
    signals_scaled["m_best"] = scale(signals["m_best"], 1, is_complex=False)
    signals_scaled["m_vcd"] = scale(signals["m_vcd"], 2**31, is_complex=False)
    
    # Normalize the signals
    signals_normalized["m_fp"] = scale(signals["m_fp"], None, is_complex=False)
    signals_normalized["m_best"] = scale(signals["m_best"], None, is_complex=False)
    signals_normalized["m_vcd"] = scale(signals["m_vcd"], None, is_complex=False)
    
    # Compute the RMSE between the signals
    rmse_original = [
        rmse(signals["m_fp"], signals["m_fp"], is_complex=False),
        rmse(signals["m_fp"], signals["m_best"], is_complex=False),
        rmse(signals["m_fp"], signals["m_vcd"], is_complex=False),
    ]
    rmse_scaled = [
        rmse(signals_scaled["m_fp"], signals_scaled["m_fp"], is_complex=False),
        rmse(signals_scaled["m_fp"], signals_scaled["m_best"], is_complex=False),
        rmse(signals_scaled["m_fp"], signals_scaled["m_vcd"], is_complex=False)
    ]
    rmse_normalized = [
        rmse(signals_normalized["m_fp"], signals_normalized["m_fp"], is_complex=False),
        rmse(signals_normalized["m_fp"], signals_normalized["m_best"], is_complex=False),
        rmse(signals_normalized["m_fp"], signals_normalized["m_vcd"], is_complex=False)
    ]
    
    new_data = {
        "Signal": ["m_fp", "m_best", "m_vcd"],
        "RMSE (original)": rmse_original,
        "RMSE (Scaled)": rmse_scaled,
        "RMSE (Normalized)": rmse_normalized
    }
    rmses = pd.DataFrame(new_data)
    rmses.to_csv("results/signals_after_final_metric_rmses.csv", index=False)
    print(tabulate(rmses.map(lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x), headers='keys', tablefmt='pretty', showindex=False))
    
    # Plot the signals
    plots_config = [("scaled", signals_scaled), ("normalized", signals_normalized)]
    for name, signal in plots_config:
        fig, axs = plt.subplots(1, 1, figsize=(12, 8), num=f"Signals after final metric ({name})")
        fig.suptitle(f"Signals after final metric ({name})", fontsize=14, fontweight='bold')
        axs.plot(signal['m_best'], label="$M(d)_{best}$")
        axs.plot(signal['m_vcd'], label="$M(d)_{vcd}$")
        axs.plot(signal['m_fp'], label="$M(d)_{fp}$")
        axs.legend(loc="upper right")
        axs.grid()
        axs.set_xlabel("Sample index [n]")
        axs.set_ylabel("Signal")
        plt.tight_layout()
        plt.savefig(f"results/signals_final_metric_plot_{name}.pdf")
        plt.close()
    print("--------------------------------------------------\n\n")
        
    # plt.show()