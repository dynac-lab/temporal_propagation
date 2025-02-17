import config as cfg
import numpy as np
from funcs import load_adjusted_strengths, get_durations_from_timelines
import os
import matplotlib.pyplot as plt

subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar
band = cfg.band
subROIs = ['TP', 'Broca', 'AG']
GSBS_dir = cfg.dir_fig + 'subROIs/GSBS/'
blocks = cfg.speech_blocks
savedir = cfg.dir_fig + 'Paper/supplementary/'

ROI_colors = {
    'TP': 'blue',
    'AG': 'green',
    'Broca': 'red'
}

nr_of_states_per_subject_per_block = {}
for subROI in subROIs:
    subROI_name = subROI
    if subROI == 'AG':
        subROI_name = 'Wernicke'
    nr_of_states_per_subject_per_block[subROI] = np.load(GSBS_dir + 'states_per_subject_ROI-' + subROI_name + '.npy', allow_pickle=True).item()


f, ax = plt.subplots(1, 3, figsize=(cfg.fig_width['two_column'] * 1.5, cfg.fig_width['two_column'] * 0.75), sharey=True)
for n_idx, n in enumerate(subjects):
    subject = str(n).zfill(2)
    medians = {}
    for subROI in subROIs + ['low', 'high']:
        # Check if this subject has this subROI
        if subROI != 'low' and subROI != 'high':
            subROI_name = subROI
            if subROI == 'AG':
                subROI_name = 'Wernicke'
            if not os.path.exists(GSBS_dir + "GSBS_" + subject + "_ROI-" + subROI_name + "_block1.npy"):
                continue
        else:
            if n == 12:
                continue

        # Get adjusted strengths
        if subROI == 'low' or subROI == 'high':
            strengths_timelines = load_adjusted_strengths(subject, subROI, blocks)
        else:
            strengths_timelines = {}
            subROI_name = subROI
            if subROI == 'AG':
                subROI_name = 'Wernicke'
            for block in blocks:
                filename = GSBS_dir + "GSBS_" + subject + "_ROI-" + subROI_name + "_block" + str(block) + ".npy"
                GSBS_obj = np.load(filename, allow_pickle=True).item()
                strengths_timelines[block] = GSBS_obj.get_strengths(nr_of_states_per_subject_per_block[subROI][subject][block])


        durations = get_durations_from_timelines(strengths_timelines)
        medians[subROI] = np.median(durations / cfg.GSBS_fs)  # in seconds


    # Plot durations
    for ROI in medians.keys():
        if ROI == 'low' or ROI == 'high':
            continue
        if 'low' in medians.keys():
            plt.sca(ax[np.where(np.asarray(list(ROI_colors.keys())) == ROI)[0][0]])
            plt.plot(range(2), [medians['low'], medians['high']], color='lightgray', marker='', linewidth=5, zorder=1)
            plt.plot(range(2), [medians['low'], medians[ROI]], color=ROI_colors[ROI], marker='o', zorder=2)

plt.sca(ax[0])
plt.ylabel("Median duration (s)")
for i in range(3):
    plt.sca(ax[i])
    subROI_name = list(ROI_colors.keys())[i]
    if subROI_name == 'Broca' or subROI_name == 'AG':
        subROI_name += "+"
    plt.xticks(range(2), ['ROI low', subROI_name])

plt.tight_layout()
plt.savefig(savedir + 'S_subROIs_durations.png')
plt.savefig(savedir + 'S_subROIs_durations.pdf')
plt.savefig(savedir + 'S_subROIs_durations.svg')
