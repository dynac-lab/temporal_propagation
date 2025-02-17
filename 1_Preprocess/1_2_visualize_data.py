import config as cfg
from funcs import plot_time_correlation, resample, extract_blocks, zscore, clip
import numpy as np
import matplotlib.pyplot as plt
from mne_bids import BIDSPath
import mne
import os

subjects = cfg.good_subjects_ECoG + cfg.good_subjects_fMRI
band = cfg.band
blocks = cfg.speech_blocks

# Prepares GSBS for all run blocks (in dictionary format), and plots time matrices to be checked
# After manually inspecting the created plots, decisions can be made on whether to use infomax or car re-referecencing
for n in subjects:
    plt.close('all')


    # Get folder
    subject = str(n).zfill(2)
    dir_sub = cfg.dir_fig + 'Preprocessing results/sub-' + subject + '/'
    if not os.path.isdir(dir_sub):
        os.mkdir(dir_sub)

    for row_idx, method in enumerate(['car', 'infomax']):
        # Load stimulus data, Hasson preproc

        path = BIDSPath(subject=subject,
                        task='film',
                        root=f'{cfg.dir_data}/preprocessed',
                        check=False,
                        datatype=method,
                        extension='.fif')
        path.update(suffix='desc-' + band + '_ieeg')
        raw = mne.io.read_raw(path.fpath, verbose=False)

        # Divide data into blocks
        data_blocks = extract_blocks(raw)

        for block in data_blocks.keys():
            # Downsample
            data_blocks[block] = resample(data_blocks[block].T, cfg.GSBS_fs, int(raw.info['sfreq'])).T

            # Clip
            data_blocks[block] = clip(data_blocks[block])

            # Z-score
            data_blocks[block] = zscore(data_blocks[block])


        # Get channel names
        chan_labels = raw.info['ch_names']

        for block in blocks:
            fig, ax = plt.subplots(2, 2, figsize=(30, 20))


            this_data = data_blocks[block]

            # Time correlation
            plot_time_correlation(ax[0,0], this_data.T, GSBS=None)
            ax[0,0].set_title("Pre-GSBS block" + str(block) + ", " + method)

            # Data over time
            plt.sca(ax[0,1])
            plt.imshow(this_data, interpolation='none', aspect='auto')
            plt.yticks(np.arange(this_data.shape[0]), chan_labels)
            plt.title("Pre-GSBS block" + str(block)+ ", " + method)



            # Data over time z-scored
            z_scored = zscore(this_data.T).T
            plt.sca(ax[1, 1])
            plt.imshow(z_scored, interpolation='none', aspect='auto')
            plt.yticks(np.arange(z_scored.shape[0]), chan_labels)
            plt.title("GSBS-z-scored")


            # Electrode correlation
            plt.sca(ax[1,0])
            plot_time_correlation(ax[1,0], this_data, GSBS=None)
            plt.yticks(np.arange(this_data.shape[0]), chan_labels)
            plt.xticks(np.arange(this_data.shape[0]), chan_labels, rotation='vertical')
            plt.xlabel("")
            plt.ylabel("")

            fig.tight_layout()
            fig.savefig(dir_sub + "plot_data_" + subject  + "_block" + str(block) + "_" + method)
            plt.close('all')