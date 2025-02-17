import config as cfg
import matplotlib.pyplot as plt
import numpy as np
from funcs import timepoints_to_timeline, load_adjusted_strengths
import pandas as pd

save_dir = cfg.dir_fig + 'Paper/supplementary/'
subject = '05'
blocks = cfg.speech_blocks
order = ['words', 'low', 'high', 'clauses']

# Get timings per annotation category per run
annotations = dict((label, dict((block, []) for block in blocks)) for label in ['clauses', 'words'])
for label in ['clauses', 'words']:
    filename = cfg.dir_annotations + "sound/sound_annotation_" + label + ".tsv"
    annotations_file = pd.read_csv(filename, sep='\t')
    timepoints_onset = annotations_file.onset.values
    timepoints_offset = annotations_file.offset.values
    timepoints = np.unique(list(timepoints_onset) + list(timepoints_offset))

    for block in blocks:
        idx_bigenough = (timepoints > block * 30)
        idx_smallenough = (timepoints < (block + 1) * 30)
        annotations[label][block] = timepoints_to_timeline(timepoints[np.logical_and(idx_bigenough, idx_smallenough)] - 30 * block, fs = cfg.GSBS_fs, duration = 30)

boundaries_per_ROI = {}
boundaries_per_ROI['low'] = load_adjusted_strengths(subject, 'low', blocks)
boundaries_per_ROI['high'] = load_adjusted_strengths(subject, 'high', blocks)
for block in blocks:
    ylabels = []
    plt.figure(figsize=(cfg.fig_width['two_column'] * 1.5, cfg.fig_width['two_column']/2))
    for l_idx, label in enumerate(order):
        if label == 'high' or label == 'low': # plot neural state boundaries
            timepoints = np.where(boundaries_per_ROI[label][block])[0]/cfg.GSBS_fs
            plt.eventplot(positions=timepoints, lineoffsets=[l_idx], colors=cfg.colors_ROI[label])
            ylabels.append('ROI ' + label)
        else: # Plot annotations
            timepoints = np.where(annotations[label][block])[0]/cfg.GSBS_fs
            plt.eventplot(positions=timepoints, lineoffsets=[l_idx], colors='k')
            ylabels.append(label)
    plt.yticks(range(4), ylabels)
    plt.xlabel('Time (s)')
    plt.ylim([-0.5, 3.5])
    plt.tight_layout()
    plt.savefig(save_dir + 'S_timelines_sub' + subject + '_block' + str(block) + '.png')
    plt.savefig(save_dir + 'S_timelines_sub' + subject + '_block' + str(block) + '.svg')
    plt.savefig(save_dir + 'S_timelines_sub' + subject + '_block' + str(block) + '.pdf')