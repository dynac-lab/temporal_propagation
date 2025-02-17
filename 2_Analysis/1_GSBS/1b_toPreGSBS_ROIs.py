import config as cfg
from funcs import plot_time_correlation
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar


band = cfg.band
blocks = cfg.speech_blocks

# Using the preGSBS data of the whole brain, select the electrodes per ROI.
for n in tqdm(subjects):
    plt.close('all')
    subject = str(n).zfill(2)
    # Get folder
    dir_sub = cfg.dir_preGSBS + 'plots_' + 'sub-' + subject + '/'
    if not os.path.isdir(dir_sub):
        os.mkdir(dir_sub)

    # Load data and info
    data_blocks = np.load(cfg.dir_preGSBS + "preGSBS_wholebrain_" + subject + "_data_" + band + '.npy', allow_pickle=True).item()
    chan_labels = np.load(cfg.dir_preGSBS + "preGSBS_wholebrain_" + subject + "_channels.npy")
    BA_per_channel = np.load(cfg.dir_electrode_labels + "sub-" + subject + "/labels_BA_sub" + subject + '.npy', allow_pickle=True).item()

    # Gather channel name and indices per ROI
    channels_idx_low = []
    channels_idx_high = []
    channels_per_ROI = dict((ROI, []) for ROI in ['low', 'high', 'none']) # Oh whoops also could have gotten them from chan_labels[ch_idx] ofcourse but oh well
    for ch_idx, ch in enumerate(chan_labels):
        BA = int(BA_per_channel[ch].split('.')[-1])
        if BA in cfg.BA_low:
            channels_per_ROI['low'].append(ch)
            channels_idx_low.append(ch_idx)
        elif BA in cfg.BA_high:
            channels_per_ROI['high'].append(ch)
            channels_idx_high.append(ch_idx)
        else:
            channels_per_ROI['none'].append(ch)

    # Select ROI data
    data_low = {}
    data_high = {}
    for block in range(13):
        data_low[block] = data_blocks[block][channels_idx_low,:]
        data_high[block] = data_blocks[block][channels_idx_high, :]

    # Plot ROI selection
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection='3d')
    coords_per_channel = np.load(cfg.dir_electrode_labels + "sub-" + subject + "/coords_per_channel_" + subject + '.npy', allow_pickle=True).item()
    for ROI in channels_per_ROI.keys():
        coords = np.asarray([coords_per_channel[ch_name] for ch_name in channels_per_ROI[ROI]])
        if len(coords) != 0:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], marker='o', edgecolors='black', s=40,
                   label=ROI)
    # Finalize and save
    ax.view_init(0, -180)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.dir_fig + 'ROIs/' + 'sub-' + subject)

    # Save data
    np.save(cfg.dir_preGSBS + "preGSBS_ROI-low_" + subject + "_data_" + band, data_low)
    np.save(cfg.dir_preGSBS + "preGSBS_ROI-high_" + subject + "_data_" + band, data_high)

    # Save channel names
    np.save(cfg.dir_preGSBS + "preGSBS_ROI-low_" + subject + "_channels", chan_labels[channels_idx_low])
    np.save(cfg.dir_preGSBS + "preGSBS_ROI-high_" + subject + "_channels", chan_labels[channels_idx_high])

    # Plot data
    for block in range(13):
        fig, ax = plt.subplots(2, 3, figsize=(30, 20))
        for row, ROI in enumerate(['high', 'low']):
            if ROI == 'low':
                this_data = data_low[block]
                chan_labels_ROI = chan_labels[channels_idx_low]
            if ROI == 'high':
                this_data = data_high[block]
                chan_labels_ROI = chan_labels[channels_idx_high]
            if len(chan_labels_ROI) == 0:
                continue

            # Time correlation
            plot_time_correlation(ax[row,0], this_data.T, GSBS=None)
            ax[row,0].set_title("ROI " + ROI)

            # Data over time
            plt.sca(ax[row,1])
            plt.imshow(this_data, interpolation='none', aspect='auto')
            plt.yticks(np.arange(this_data.shape[0]), chan_labels_ROI)

            # Electrode correlation
            plt.sca(ax[row,2])
            plot_time_correlation(ax[row,2], this_data, GSBS=None)
            plt.yticks(np.arange(this_data.shape[0]), chan_labels_ROI)
            plt.xticks(np.arange(this_data.shape[0]), chan_labels_ROI, rotation='vertical')
            plt.xlabel("")
            plt.ylabel("")

            title = "Sub-" + subject + "block-" + str(block) + " "
            if block in cfg.speech_blocks:
                title += "(speech)"
            else:
                title += "(music)"
            fig.suptitle(title)

        plt.tight_layout()
        plt.savefig(dir_sub + "plot_data_" + subject + "_ROIs_block" + str(block) + "_" + band)