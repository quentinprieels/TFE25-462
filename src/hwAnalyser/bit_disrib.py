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
parser.add_argument("--separator_index", type=int, default=-1, help="Index separating I and Q components. -1 for automatic (bus size value / 2).")

# Parse arguments
args = parser.parse_args()

# Assign arguments to variables
vcd_file = args.vcd_file
signal_tdata_name = args.signal_tdata_name
signal_tvalid_name = args.signal_tvalid_name
signal_clk_name = args.signal_clk_name
separator_index = args.separator_index

# Get the right signals
vcd = VCDVCD(vcd_file)
signal_data = vcd[signal_tdata_name]
signal_tvalid = vcd[signal_tvalid_name]
signal_clk = vcd[signal_clk_name]

# Bus size
signal_tdata_bus_size = int(signal_data.size)
print(f"Signal tdata bus size: {signal_tdata_bus_size}")
if separator_index == -1: 
    separator_index = signal_tdata_bus_size // 2
    print(f"Separator index set to: {separator_index}")
   
# Initialize bit counters 
bit_count_1 = np.zeros(signal_tdata_bus_size, dtype=int)
bit_count_0 = np.zeros(signal_tdata_bus_size, dtype=int)

# Get the clock edges (rising edges)
clk_rising_edges = [t for t, v in signal_clk.tv if int(v) == 1]

# Iterate over the signal data
for timestamp in clk_rising_edges:
    
    # Ensure tvalid is high at current timestamp
    tvalid_value = int(signal_tvalid[timestamp])
    if tvalid_value != 1:
        continue
    
    # Count the number of 1s and 0s in the tdata value
    tdata_value = signal_data[timestamp]
    
    # Store the tdata bit values
    for i, bit in enumerate(tdata_value.zfill(signal_tdata_bus_size)[-signal_tdata_bus_size:]):
        if bit == "1":
            bit_count_1[i] += 1
        elif bit == "0":
            bit_count_0[i] += 1
            
# Assert the bit counts
assert np.unique(bit_count_1 + bit_count_0).size == 1, "Bit counts are not equal"

# Calculate the percentage of 1s and 0s
total_counts = bit_count_1 + bit_count_0
bit_count_0_percentage = (bit_count_0 / total_counts) * 100
bit_count_1_percentage = (bit_count_1 / total_counts) * 100


# Plot the results
x = np.arange(signal_tdata_bus_size)
y_ticks = np.arange(0, 120, 10)

plt.figure(figsize=(15, 6))
plt.title(f"Bit statistics for {signal_tdata_name}", fontsize=14, fontweight='bold')
plt.bar(x, bit_count_1_percentage, width=0.5, label="1s", color='tab:blue', bottom=bit_count_0_percentage)
plt.bar(x, bit_count_0_percentage, width=0.5, label="0s", color='tab:orange')

# Add a vertical line at the separator index
if separator_index != 0 and separator_index < signal_tdata_bus_size:
    plt.text(separator_index / 2, 105, "I component", ha='center', va='center', fontsize=11)
    plt.text((separator_index + signal_tdata_bus_size) / 2, 105, "Q component", ha='center', va='center', fontsize=11)
    plt.axvline(x=separator_index - 0.5, color='black', linestyle='--', linewidth=1.5)

plt.xlabel("Bit Position")
plt.ylabel("Percentage (%)")
plt.xticks(x, [str(signal_tdata_bus_size - i - 1) for i in x])
plt.yticks(y_ticks)
plt.legend()
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
