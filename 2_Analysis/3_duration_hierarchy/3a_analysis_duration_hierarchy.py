from funcs import load_adjusted_strengths, get_durations_from_timelines
import config as cfg
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar
blocks = cfg.speech_blocks
ROIs = ['low', 'high']



# Prepare figure
plt.figure(figsize=(5,10))

# Loop over subjects to gather into
for n in subjects:
    if n == 12: # sub-12 does not have low ROI
        continue
    medians = {}
    for ROI in ROIs:
        ROI_name = 'ROI-' + ROI
        subject = str(n).zfill(2)

        # Get median duration of this ROI
        strengths_timelines = load_adjusted_strengths(subject,ROI, blocks)
        for block in blocks:
            durations = get_durations_from_timelines(strengths_timelines)
            medians[ROI] = np.median(durations/cfg.GSBS_fs) # in seconds

    # Plot
    if n in cfg.subjects_analysis_novel:
        c = 'red'
    else:
        c = 'blue'
    plt.plot(range(2), [medians[ROIs[0]], medians[ROIs[1]]], color=c, marker='o')
plt.xticks(range(2), ROIs)
plt.xlabel("ROI")
plt.ylabel("Median duration (s)")
plt.tight_layout()
plt.savefig(cfg.dir_analysis_results + '3_duration_hierarchy/' + 'median_durations')
# TODO: get legend for novel vs familiar


# No statistical analysis because this will not lead to anything