import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

title = "Time taken by each block of the OFDM receiver for different number of transmitted pulses"
results = pd.read_csv('results.csv')

palette = sns.color_palette("Paired", as_cmap=True)

plt.figure("Experiment 01", figsize=(15, 8))
plt.title(title)
sns.set_theme(style='whitegrid')

# Add markers to distinguish the different sizes in grayscale
sns.barplot(x='function', y='time', hue='size', data=results, palette=palette, zorder=3)

plt.xlabel('Function')
plt.ylabel('Time (s)')
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# plt.show()
plt.savefig('results.pdf')