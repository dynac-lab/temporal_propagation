import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from funcs import load_adjusted_strengths, get_durations_from_timelines


dir_save = cfg.dir_fig + 'Paper/'

subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar
blocks = cfg.speech_blocks
ROIs = ['low', 'high']



# Prepare figure
plt.figure(figsize=(cfg.fig_width['one_column'] * 1.2,cfg.fig_width['one_column'] * 1.5)) # I am using factor 1.5 everywhere for some reason
plt.gca().set_axisbelow(True)
plt.grid(axis='y', color=cfg.grid_color)

# Loop over subjects to gather into
for n in subjects:
    medians = {}
    for ROI in ROIs:
        ROI_name = 'ROI-' + ROI
        subject = str(n).zfill(2)

        if n == 12 and ROI == 'low':
            medians[ROI] = np.nan
            continue  # sub-12 does not have low ROI

        # Get median duration of this ROI
        strengths_timelines = load_adjusted_strengths(subject,ROI, blocks)
        for block in blocks:
            durations = get_durations_from_timelines(strengths_timelines)
            medians[ROI] = np.median(durations/cfg.GSBS_fs) # in seconds
    plt.plot(range(2), [medians[ROIs[0]], medians[ROIs[1]]], color='k', linewidth=1, zorder=1)
    plt.scatter(range(2), [medians[ROIs[0]], medians[ROIs[1]]], facecolors=[cfg.colors_ROI[ROIs[0]], cfg.colors_ROI[ROIs[1]]], edgecolors='k', zorder=2)

ylims = plt.gca().get_ylim()
y_ticks = np.round(np.arange(np.floor(plt.gca().get_ylim()[0] * 10)/10, np.ceil(plt.gca().get_ylim()[1] * 10)/10, 0.1),1)
plt.yticks(y_ticks, y_ticks)

plt.xlim([-0.25,1.25])
plt.xticks(range(2), ["ROI " + ROI for ROI in ROIs])
plt.ylabel("Median duration (s)")

plt.tight_layout()
plt.savefig(dir_save + 'fig2_durations.png')
plt.savefig(dir_save + 'fig2_durations.svg')
plt.savefig(dir_save + 'fig2_durations.pdf')

