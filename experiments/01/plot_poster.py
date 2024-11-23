import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

# Assuming `results_poster.csv` is properly formatted and present
results = pd.read_csv('results_poster.csv')

# Remove the iteration column
results = results.drop(columns='iteration')

# Group by function and take the mean of the time
results = results.groupby(['function']).mean().reset_index()

# Compute the total common time (synchronisation + demodulation + channel estimation)
common_time = results.loc[
    results['function'].isin(['OFDM_synchronisation', 'OFDM_demodulation', 'OFDM_channel_estimate']), 'time'
].sum()
results = results._append({'function': 'Total_common', 'time': common_time}, ignore_index=True)

# Colors for each block
block_colors = {
    "OFDM_synchronisation": "#00204e",
    "OFDM_demodulation": "#f0c987",
    "OFDM_channel_estimate": "#5db4e6",
    "OFDM_channel_equalisation": "#8b687f",
    "inverse_mapping": "#5db4e6", # Not shown in the plot
    "new_SISO_OFDM_DFRC_RADAR_RX": "#b5dbab",
    "Total_common": "white",
}

x = ["Communication", "Common", "RADAR" ]

# Bar 1 is "OFDM_synchronisation", "Total_common", "Total_common"
bar1 = [
    common_time,
    results.loc[results['function'] == 'OFDM_synchronisation', 'time'].values[0],
    common_time
]
bar1_colors = [
    block_colors["Total_common"],
    block_colors["OFDM_synchronisation"],
    block_colors["Total_common"],
]
bar1_hatch = [
    "/ /",
    "",
    "/ /",
]
# Bar 2 is "OFDM_demodulation", "OFDM_channel_equalisation", "new_SISO_OFDM_DFRC_RADAR_RX"
bar2 = [
    results.loc[results['function'] == 'OFDM_channel_equalisation', 'time'].values[0],
    results.loc[results['function'] == 'OFDM_demodulation', 'time'].values[0],
    results.loc[results['function'] == 'new_SISO_OFDM_DFRC_RADAR_RX', 'time'].values[0]
    
]
bar2_colors = [
    block_colors["OFDM_channel_equalisation"],
    block_colors["OFDM_demodulation"], 
    block_colors["new_SISO_OFDM_DFRC_RADAR_RX"],
]

# Bar 3 is "OFDM_channel_estimate", "0", "0"
bar3 = [
    0,
    results.loc[results['function'] == 'OFDM_channel_estimate', 'time'].values[0],
    0
]
bar3_colors = [
    "none",  # No color since the value is 0
    block_colors["OFDM_channel_estimate"],
    "none",  # No color since the value is 0
]

# use the Montserrat-Medium font for the plot (.ttf file in the same directory)
font = FontProperties(fname='Montserrat-Medium.ttf')
font.set_size(14)
mpl.rcParams['hatch.linewidth'] = 0.8  # Adjust as needed


# Figure parameters
bin_height = 0.8

# reduce the margins

plt.figure(figsize=(12, 2.5))  # Reduced height
plt.barh(x, bar1, color=bar1_colors, hatch=bar1_hatch, height=bin_height)
plt.barh(x, bar2, left=bar1, color=bar2_colors, height=bin_height)
plt.barh(x, bar3, left=np.add(bar1, bar2), color=bar3_colors, height=bin_height)

plt.xlabel('Time [ms]', fontproperties=font)
#plt.ylabel('Chain', fontproperties=font)
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin
plt.subplots_adjust(left=0.15)  # Adjust left margin

plt.savefig('complexity.pdf', bbox_inches='tight', pad_inches=0)
# plt.legend()
plt.show()