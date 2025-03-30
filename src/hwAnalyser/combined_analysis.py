import numpy as np
import matplotlib.pyplot as plt
from vcdvcd import VCDVCD
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Analyze signal statistics from a VCD file.")
parser.add_argument("--vcd_file", type=str, default="rfnoc_block_schmidl_cox_tb_trunc.vcd", help="Path to the VCD file.")
parser.add_argument("--signal_tdata_name", type=str, default="rfnoc_block_schmidl_cox_tb.dut.mc0.c5_tdata[31:0]", help="Name of the tdata signal.")
parser.add_argument("--signal_tvalid_name", type=str, default="rfnoc_block_schmidl_cox_tb.dut.mc0.c5_tvalid", help="Name of the tvalid signal.")
parser.add_argument("--signal_clk_name", type=str, default="rfnoc_block_schmidl_cox_tb.dut.axis_data_clk", help="Name of the clock signal.")
parser.add_argument("--separator_index", type=int, default=-1, help="Index separating I and Q components. -1 for automatic (bus size value / 2).")
parser.add_argument("--signal_type", type=str, default="complex", help="Type of the signal to be plotted (complex or real).", choices=["complex", "real"])

# Parse arguments
args = parser.parse_args()

# Assign arguments to variables
vcd_file = args.vcd_file
signal_tdata_name = args.signal_tdata_name
signal_tvalid_name = args.signal_tvalid_name
signal_clk_name = args.signal_clk_name
separator_index = args.separator_index
signal_type = args.signal_type

# Get the right signals
vcd = VCDVCD(vcd_file)
signal_data = vcd[signal_tdata_name]
signal_tvalid = vcd[signal_tvalid_name]
signal_clk = vcd[signal_clk_name]

# Bus size
signal_tdata_bus_size = int(signal_data.size)
if separator_index == -1:
    separator_index = signal_tdata_bus_size // 2

# Initialize bit counters 
bit_count_1 = np.zeros(signal_tdata_bus_size, dtype=int)
bit_count_0 = np.zeros(signal_tdata_bus_size, dtype=int)

# Get the clock edges (rising edges)
clk_rising_edges = [t for t, v in signal_clk.tv if int(v) == 1]

# Iterate over the signal data
signal_times = []
signal_values = []
for timestamp in clk_rising_edges:
    # Ensure tvalid is high at current timestamp
    tvalid_value = int(signal_tvalid[timestamp])
    if tvalid_value != 1:
        continue
    
    # Count the number of 1s and 0s in the tdata value
    #! CAUTION: Bit array is reversed in VCD => [0] is MSB
    tdata_value = signal_data[timestamp]
    for i, bit in enumerate(tdata_value.zfill(signal_tdata_bus_size)[-signal_tdata_bus_size:]):
        if bit == "1":
            bit_count_1[i] += 1
        elif bit == "0":
            bit_count_0[i] += 1
            
    # Store the tdata bit values
    signal_times.append(timestamp)
    signal_values.append(tdata_value.zfill(signal_tdata_bus_size)[-signal_tdata_bus_size:])

# Assert bit count correctness
total_counts = bit_count_1 + bit_count_0
bit_count_0_percentage = (bit_count_0 / total_counts) * 100
bit_count_1_percentage = (bit_count_1 / total_counts) * 100

# Convert signal values
if signal_type == "complex":
    signal_values = np.array([
        complex(int(value[:separator_index], 2), int(value[separator_index:], 2))
        for value in signal_values
    ])
else:
    signal_values = np.array([int(value, 2) for value in signal_values])

# Plot setup
if signal_type == "complex":
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
else:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle(f"Signal Analysis for {signal_tdata_name}", fontsize=16, fontweight='bold')

# Bit Statistics Plot (50%)
y_ticks = np.arange(0, 120, 10)
x_ticks = np.arange(0, signal_tdata_bus_size, 1)

axes[0].bar(range(signal_tdata_bus_size), bit_count_1_percentage, width=0.5, label="1s", color='tab:blue', bottom=bit_count_0_percentage)
axes[0].bar(range(signal_tdata_bus_size), bit_count_0_percentage, width=0.5, label="0s", color='tab:orange')
axes[0].set_title("Bit Distribution on the bus", fontsize=12)
axes[0].set_xlabel("Bit Position")
axes[0].set_ylabel("Percentage (%)")
axes[0].set_xticks(x_ticks, [f"{signal_tdata_bus_size - i - 1}" for i in x_ticks])
axes[0].set_yticks(y_ticks, [f"{i}" for i in y_ticks])
axes[0].legend()
axes[0].grid(axis='y', linestyle='--')
if separator_index != 0 and separator_index < signal_tdata_bus_size and signal_type == "complex":
    axes[0].axvline(x=separator_index - 0.5, color='black', linestyle='--', linewidth=1.5)
    axes[0].text(separator_index / 2, 105, "I component", ha='center', va='center', fontsize=10)
    axes[0].text((separator_index + signal_tdata_bus_size) / 2, 105, "Q component", ha='center', va='center', fontsize=10)

# Real and Imaginary Components (25% each)
signal_times = np.array(signal_times) * 1e-6  # Convert to microseconds (ps to µs)
if signal_type == "complex":
    axes[1].plot(signal_times, np.real(signal_values), label="Real Part", color="tab:green")
    axes[1].set_title("Real Part of Signal (I)", fontsize=12)
    axes[1].set_xlabel("Time (µs)")
    axes[1].set_ylabel("Real Value")
    axes[1].grid()

    axes[2].plot(signal_times, np.imag(signal_values), label="Imaginary Part", color="tab:green")
    axes[2].set_title("Imaginary Part of Signal (Q)", fontsize=12)
    axes[2].set_xlabel("Time (µs)")
    axes[2].set_ylabel("Imaginary Value")
    axes[2].grid()
else:
    axes[1].plot(signal_times, signal_values, label="Signal Value", color="tab:green")
    axes[1].set_title("Signal Over Time", fontsize=12)
    axes[1].set_xlabel("Time (µs)")
    axes[1].set_ylabel("Signal Value")
    axes[1].grid()

plt.tight_layout()
plt.savefig(f"{signal_tdata_name}_analysis.pdf", bbox_inches='tight')
# plt.show()
