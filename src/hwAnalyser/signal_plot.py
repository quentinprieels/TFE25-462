import numpy as np
import matplotlib.pyplot as plt
from vcdvcd import VCDVCD
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Analyze signal statistics from a VCD file.")
parser.add_argument("--vcd_file", type=str, default="rfnoc_block_schmidl_cox_tb.vcd", help="Path to the VCD file.")
parser.add_argument("--signal_tdata_name", type=str, default="rfnoc_block_schmidl_cox_tb.dut.mc0.c5_tdata[31:0]", help="Name of the tdata signal.")
parser.add_argument("--signal_tvalid_name", type=str, default="rfnoc_block_schmidl_cox_tb.dut.mc0.c5_tvalid", help="Name of the tvalid signal.")
parser.add_argument("--signal_clk_name", type=str, default="rfnoc_block_schmidl_cox_tb.dut.axis_data_clk", help="Name of the clock signal.")
parser.add_argument("--signal_type", type=str, default="complex", help="Type of the signal to be plotted.(complex or real)")

# Parse arguments
args = parser.parse_args()

# Assign arguments to variables
vcd_file = args.vcd_file
signal_tdata_name = args.signal_tdata_name
signal_tvalid_name = args.signal_tvalid_name
signal_clk_name = args.signal_clk_name
signal_type = args.signal_type

# Get the right signals
vcd = VCDVCD(vcd_file)
signal_data = vcd[signal_tdata_name]
signal_tvalid = vcd[signal_tvalid_name]
signal_clk = vcd[signal_clk_name]

# Bus size
signal_tdata_bus_size = int(signal_data.size)
print(f"Signal tdata bus size: {signal_tdata_bus_size}")

# Initialize signal arrays
signal_times = []
signal_values = []

# Get the clock edges (rising edges)
clk_rising_edges = [t for t, v in signal_clk.tv if int(v) == 1]

# Iterate over the signal data
for timestamp in clk_rising_edges:
    
    # Ensure tvalid is high at current timestamp
    tvalid_value = int(signal_tvalid[timestamp])
    if tvalid_value != 1:
        continue
    
    # Get the tdata value
    tdata_value = signal_data[timestamp]
    
    # Store the signal value and timestamp
    signal_times.append(timestamp)
    signal_values.append(tdata_value.zfill(signal_tdata_bus_size)[-signal_tdata_bus_size:])
    
# Convert to numpy arrays
if signal_type == "complex":
    if signal_tdata_bus_size == 32:
        # Real part in the first 16 bits, imaginary part in the last 16 bits (signed)
        signal_values = np.array([
            complex(
                int(value[:16], 2),  # Convert real part to signed 16-bit integer
                int(value[16:], 2)  # Convert imaginary part to signed 16-bit integer
            ) for value in signal_values
        ])
    elif signal_tdata_bus_size == 64:
        # Real part in the first 32 bits, imaginary part in the last 32 bits (signed)
        signal_values = np.array([
            complex(
                int(value[:32], 2),  # Convert real part to signed 32-bit integer
                int(value[32:], 2)  # Convert imaginary part to signed 32-bit integer
            ) for value in signal_values
        ])
    else:
        raise ValueError(f"Unsupported bus size: {signal_tdata_bus_size}")
else:
    # Convert to numpy array (signed integers)
    signal_values = np.array([np.int32(int(value, 2)) for value in signal_values])
    

# Plot the results
plt.figure(figsize=(12, 6))

if signal_type == "complex":
    # Subplot for the real part
    plt.subplot(2, 1, 1)
    plt.plot(signal_times, np.real(signal_values), label="Real Part", color="tab:blue")
    plt.title(f"Real Part of Signal {signal_tdata_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (ns)")
    plt.ylabel("Real Value")
    plt.grid()
    plt.legend()

    # Subplot for the imaginary part
    plt.subplot(2, 1, 2)
    plt.plot(signal_times, np.imag(signal_values), label="Imaginary Part", color="tab:orange")
    plt.title(f"Imaginary Part of Signal {signal_tdata_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (ns)")
    plt.ylabel("Imaginary Value")
    plt.grid()
    plt.legend()
else:
    # Single plot for real signal
    plt.plot(signal_times, signal_values, label="Signal Value", color="tab:green")
    plt.title(f"Signal {signal_tdata_name} over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Time (ns)")
    plt.ylabel("Signal Value")
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.show()
