import matplotlib.pyplot as plt
import config as cfg
from funcs import plot_time_correlation, extract_blocks
from funcs import clip, zscore
from mne_bids import BIDSPath
import mne
from utils import resample
import numpy as np

from matplotlib import  rcParams
rcParams['path.simplify'] = True

# Plots figures for each subject that was excluded from the analysis based on visual inspection of the data, as well as one example subject
save_dir = cfg.dir_fig + 'Paper/supplementary/'

def get_data(subject):
    # Load stimulus data, Hasson preproc
    path = BIDSPath(subject=subject,
                    task='film',
                    root=f'{cfg.dir_data}/preprocessed',
                    check=False,
                    datatype=cfg.preproc_per_subject[subject],
                    extension='.fif')
    path.update(suffix='desc-' + 'preproc' + '_ieeg')
    raw = mne.io.read_raw(path.fpath, verbose=False)

    # Drop extra bad channels (based on visual inspection of 'Preprocessing Results')
    # NOTE: actually for these particular subjects the list of bad channels is always empty.
    raw.drop_channels(cfg.bad_channels[subject])

    # Divide data into blocks
    data_blocks = extract_blocks(raw)

    for block in data_blocks.keys():
        # Downsample to consistent fs across subjects
        data_blocks[block] = resample(data_blocks[block].T, cfg.GSBS_fs, int(raw.info['sfreq'])).T

        # Clip
        data_blocks[block] = clip(data_blocks[block])

        # Z-score
        data_blocks[block] = zscore(data_blocks[block])

    # select channels
    for block in cfg.speech_blocks:
        data_blocks[block] = data_blocks[block]
    return data_blocks

example_subject = []
excluded_subjects = [3,7,48,57,60]

chosen_runs = {
    5: 1,
    12: 1,
    3: 3,
    7: 9,
    48: 5,
    57: 3,
    60:1
}

f, ax = plt.subplots(len(excluded_subjects),2, figsize=(cfg.fig_width['two_column']*1.5,cfg.fig_width['two_column']*2.085), gridspec_kw={'width_ratios': [6,3]})
for n_idx, n in enumerate(excluded_subjects):
    # Get data
    subject = str(n).zfill(2)
    data = get_data(subject)

    # Only select one run
    data = data[chosen_runs[n]]

    # Plot timeseries
    plt.sca(ax[n_idx,0])
    plt.imshow(data, interpolation='none', aspect='auto')
    title_plot = "Subject " + subject
    if '12' in subject or '05' in subject:
        title_plot += " (included example subject)"
    else:
        title_plot += " (excluded subject)"
    plt.title(title_plot, loc='left')
    plt.ylabel("electrode")
    plt.yticks([],[])
    plt.xlabel("Time (s)")

    # One tick every 5 seconds
    plt.xticks(np.arange(0,31,5) * cfg.GSBS_fs,np.arange(0,31,5))

    # Plot time correlation
    plt.sca(ax[n_idx,1])
    plot_time_correlation(ax[n_idx,1], data[:, :10 * cfg.GSBS_fs].T)
    plt.xlabel("Time (s)")
    plt.ylabel("Time (s)")
    plt.xticks(np.arange(0, 11, 2) * cfg.GSBS_fs, np.arange(0, 11, 2))
    plt.yticks(np.arange(0, 11, 2) * cfg.GSBS_fs, np.arange(0, 11, 2))



plt.tight_layout()
print("Saving png...")
plt.savefig(save_dir + "S_visual_artifacts.png")

# PDF took a humungous amount of time to save
# print("Saving pdf...")
# plt.savefig(save_dir + "S_visual_artifacts.pdf")
# print("Saving svg...")
# plt.savefig(save_dir + "S_visual_artifacts.svg")
print("Done!")

