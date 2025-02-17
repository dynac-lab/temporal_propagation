import matplotlib.pyplot as plt
import numpy as np
import config as cfg

# Set directories
dir_save = cfg.dir_fig + 'Paper/'
load_dir = cfg.dir_analysis_results + "4_alignment_ROIs/"

# Gather data
delays = cfg.ROI_delays
results = np.load(load_dir + 'wilcoxon_results.npy', allow_pickle=True).item() # Info on p value
matches_relative = np.load(load_dir + 'matches_relative_over_ROIdelays.npy', allow_pickle=True).item() # Lines to plot
optimal_delays = np.load(load_dir + 'optimal_ROIdelay.npy', allow_pickle=True).item() # Optimal delay based on raw Gaussian match
measure_per_subject = np.load(load_dir + 'measure_per_subject.npy', allow_pickle=True).item() # Max relative gaussian

# Create plot
f,ax = plt.subplots(2,1, figsize=(cfg.fig_width['one_column'] * 1.5, 3.5 * 1.5), gridspec_kw={'height_ratios': [4,1]}, sharex=True)
grayscale = plt.cm.gray(np.linspace(0.5,0.95, len(matches_relative.keys())))
plt.sca(ax[0])
ax[0].set_axisbelow(True)
plt.grid(axis='x', color=cfg.grid_color)
# Plot individual relative matches over delays, and optimal delay
for n_idx, n in enumerate(matches_relative.keys()):
    # Gaussian match
    plt.plot(delays,matches_relative[n], color=grayscale[n_idx], linewidth=2)

    # Optimal delay
    plt.scatter(optimal_delays[n], matches_relative[n][delays == optimal_delays[n]][0], facecolor='green', edgecolor='k', zorder = 5)

# Add average line
avg_relative_match = np.mean(np.asarray(list(matches_relative.values())), axis=0)
plt.plot(delays, avg_relative_match, 'blue', linewidth=2, zorder=3)

# Extra elements and layout
plt.axhline(0, color='k', zorder=1)
plt.axvline(0, color='k', linestyle='--', zorder=1)
plt.ylabel("Relative Gaussian match")


# Scatter boxplot for max relative match
plt.sca(ax[1])
ax[1].set_axisbelow(True)
plt.grid(axis='x', color=cfg.grid_color)
y = np.asarray(list(optimal_delays.values()))
plt.boxplot(y, positions=[0], zorder=2, patch_artist=True, boxprops={'facecolor': 'white'},
            medianprops={'color': 'green', 'linewidth': 3}, vert=False, widths=0.5)
plt.scatter(y, np.zeros(len(y)), edgecolors='k',zorder=3, c='green')
plt.xlabel("Delay between ROIs (s)")
plt.yticks([],[])



# Save
plt.tight_layout()
plt.savefig(dir_save + 'fig4_match_over_delay_ROIs.png')
plt.savefig(dir_save + 'fig4_match_over_delay_ROIs.pdf')
plt.savefig(dir_save + 'fig4_match_over_delay_ROIs.svg')