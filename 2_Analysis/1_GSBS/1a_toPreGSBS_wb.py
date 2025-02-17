import config as cfg
from funcs import plot_time_correlation, resample, extract_blocks, zscore, clip
import numpy as np
import matplotlib.pyplot as plt
from mne_bids import BIDSPath
import mne
import os

subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar

band = cfg.band
blocks = cfg.speech_blocks



# Prepares GSBS for all run blocks (in dictionary format), and plots time matrices to be checked
# Bad electrodes can be added to cfg.bad_channels[subject], after which this script should run again
for n in subjects:
    plt.close('all')


    # Load stimulus data, Hasson preproc
    subject = str(n).zfill(2)
    path = BIDSPath(subject=subject,
                    task='film',
                    root=f'{cfg.dir_data}/preprocessed',
                    check=False,
                    datatype=cfg.preproc_per_subject[subject],
                    extension='.fif')
    path.update(suffix='desc-' + band + '_ieeg')
    raw = mne.io.read_raw(path.fpath, verbose=False)

    # Drop extra bad channels (based on visual inspection of 'Preprocessing Results')
    raw.drop_channels(cfg.bad_channels[subject])

    # Get folder
    dir_sub = cfg.dir_preGSBS + 'plots_' + 'sub-' + subject + '/'
    if not os.path.isdir(dir_sub):
        os.mkdir(dir_sub)


    # Divide data into blocks
    data_blocks = extract_blocks(raw)

    # Print number of electrodes
    print("Number of channels " + subject + ": " + str(data_blocks[0].shape[0]))

    for block in data_blocks.keys():
        # Downsample
        data_blocks[block] = resample(data_blocks[block].T, cfg.GSBS_fs, int(raw.info['sfreq'])).T

        # Clip
        data_blocks[block] = clip(data_blocks[block])

        # Z-score
        data_blocks[block] = zscore(data_blocks[block])

    # Save data
    np.save(cfg.dir_preGSBS + "preGSBS_wholebrain_" + subject + "_data_" + band, data_blocks)

    # Save channel names
    chan_labels = raw.info['ch_names']
    np.save(cfg.dir_preGSBS + "preGSBS_wholebrain_" + subject + "_channels", chan_labels)

    for block in range(13):
        fig, ax = plt.subplots(1,3, figsize=(30,10))
        this_data = data_blocks[block]

        # Time correlation
        plot_time_correlation(ax[0], this_data.T, GSBS=None)

        # Data over time
        plt.sca(ax[1])
        plt.imshow(this_data, interpolation='none', aspect='auto')
        plt.yticks(np.arange(this_data.shape[0]), chan_labels)

        # Electrode correlation
        plt.sca(ax[2])
        plot_time_correlation(ax[2], this_data, GSBS=None)
        plt.yticks(np.arange(this_data.shape[0]), chan_labels)
        plt.xticks(np.arange(this_data.shape[0]), chan_labels, rotation='vertical')
        plt.xlabel("")
        plt.ylabel("")



        title = "block " + str(block) + " "
        if block in cfg.speech_blocks:
            title += "(speech)"
        else:
            title += "(music)"
        fig.suptitle(title)

        plt.tight_layout()
        plt.savefig(dir_sub + "plot_data_" + subject  + "_block" + str(block) + "_" + band)