import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from funcs import p_to_str


dir_save = cfg.dir_fig + 'Paper/'
dir_results = cfg.dir_analysis_results + "2_alignment_speech/"
x_movement = 0.1
box_width = 0.3

## Load results
matches_relative = np.load(dir_results + 'matches_relative_over_delays.npy', allow_pickle=True).item()
p_values = np.load(dir_results + 'wilcoxon_pvalues.npy', allow_pickle=True).item()
optimal_delays = np.load(dir_results + 'optimal_delays.npy', allow_pickle=True).item()
delays = cfg.speech_delays

print(p_values)

## Create figure
f, ax = plt.subplots(1,2, figsize=(cfg.fig_width['one_column'] * 1.5, 3.5 * 1.5), gridspec_kw={'width_ratios': [4,1]}, sharey=True)

## Plot A: max amplitude of raw Gaussian match, match with speech
plt.sca(ax[0])
plt.gca().set_axisbelow(True)
plt.grid(axis='y', color=cfg.grid_color)

# Gather info: max height per subject per ROI per annotation catgory
data = {}
for annotation_category in matches_relative.keys():
    data[annotation_category] = {}
    for ROI in matches_relative[annotation_category].keys():
        data[annotation_category][ROI] = []
        for n in matches_relative[annotation_category][ROI].keys():
            if n == 12 and ROI == 'low':
                continue
            data[annotation_category][ROI].append(np.max(matches_relative[annotation_category][ROI][n]))

# Plot data
for ROI_idx, ROI in enumerate(matches_relative[annotation_category].keys()):
    if ROI_idx == 0:
        x = np.arange(2) - x_movement
    else:
        x = np.arange(2) + x_movement
    y = [data['words'][ROI], data['clauses'][ROI]]

    # Boxplot
    plt.boxplot(y, positions=x, zorder=1, patch_artist=True, boxprops={'facecolor':'white'}, medianprops={'color': cfg.colors_ROI[ROI], 'linewidth': 3})

    # Scatter plot
    plt.scatter([i for y_idx, i in enumerate(x) for j in y[y_idx]], np.asarray(y).flatten(), edgecolors='k', label='ROI ' + ROI, zorder=2, c=cfg.colors_ROI[ROI])

# Plot p values
for annotation_idx, annotation_category in enumerate(p_values.keys()):
    for ROI_idx, ROI in enumerate(p_values[annotation_category].keys()):
        p = p_values[annotation_category][ROI]
        if p < 0.05:
            stars = p_to_str(p)
            if ROI_idx == 0:
                x = annotation_idx - x_movement
            else:
                x = annotation_idx + x_movement
            y = np.max(data[annotation_category][ROI]) + 0.02
            plt.text(x=x,y=y,s=stars, ha='center')


# Additional layout
plt.axhline(0, color='k', linestyle='--', zorder=0)
plt.xticks([0,1], matches_relative.keys())
plt.ylabel('Maximum relative Gaussian match')
plt.legend(loc='upper left')
plt.title("A)", loc='left')

## Plot B: boxplot of ROI-ROI match
plt.sca(ax[1])
plt.gca().set_axisbelow(True)
plt.grid(axis='y', color=cfg.grid_color)
load_dir = cfg.dir_analysis_results + "4_alignment_ROIs/"
measure_per_subject = np.load(load_dir + 'measure_per_subject.npy', allow_pickle=True).item() # Max relative gaussian
y = np.asarray(list(measure_per_subject.values()))
x = np.zeros(len(y))
plt.boxplot(y, positions=[0], zorder=1, patch_artist=True, boxprops={'facecolor': 'white'},
            medianprops={'color': 'green', 'linewidth': 3}, widths=[box_width])
plt.scatter(x, y, edgecolors='k',zorder=2, c='green')
plt.xticks([0],['ROI-ROI'])


# p-value
results = np.load(load_dir + 'wilcoxon_results.npy', allow_pickle=True).item() # Info on p value
p_value = results['p']
if p_value < 0.05:
    stars = p_to_str(p_value)
    x = 0
    y = np.max(list(measure_per_subject.values())) + 0.02
    plt.text(x=x, y=y, s=stars, ha='center')

# Adjust xlim to have p-stars within plot
ylimits = ax[1].get_ylim()
plt.ylim([ylimits[0], ylimits[1]+0.02])

# Extra layout
plt.axhline(0, color='k', linestyle='--', zorder=0)
plt.title("B)", loc='left')

plt.tight_layout()
plt.savefig(dir_save + 'fig1_alignment.png')
plt.savefig(dir_save + 'fig1_alignment.pdf')
plt.savefig(dir_save + 'fig1_alignment.svg')