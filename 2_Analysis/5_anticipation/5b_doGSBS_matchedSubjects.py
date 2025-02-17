import config as cfg
from funcs import run_GSBS
import numpy as np
import os
import multiprocessing

max_nr_of_processes = 13
band = cfg.band
save_dir = cfg.dir_GSBS + 'matched_subjects/'
processes = []
subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar

# Load matrix with the number of channels that are overlapping per subject pair
matrix_channel_overlap = np.load(cfg.dir_analysis_results + '5_anticipation/matrix_channel_overlap.npy')
subjects = np.load(cfg.dir_analysis_results + '5_anticipation/matrix_channel_overlap_subjects.npy')
N = len(subjects)

for n1 in range(N):
    for n2 in range(N):
        sub1 = subjects[n1]
        sub2 = subjects[n2]
        if sub1 < sub2 and matrix_channel_overlap[n1,n2] >= 15:
            channels_per_subject = np.load(cfg.dir_analysis_results + '5_anticipation/matching_channels/' + str(sub1) + "vs" + str(sub2) + '.npy', allow_pickle=True).item()

            for this_sub in [sub1, sub2]:
                subject = str(this_sub).zfill(2)

                # Load preGSBS whole-brain
                filename_data = cfg.dir_preGSBS + "preGSBS_wholebrain_" + subject + "_data_" + band + ".npy"
                data = np.load(filename_data, allow_pickle=True).item()
                channels_wholebrain = np.load(cfg.dir_preGSBS + "preGSBS_wholebrain_" + subject + "_channels.npy")

                # Get channel indices
                channel_indices = [chan_idx for chan_idx in range(len(channels_wholebrain)) if channels_wholebrain[chan_idx] in channels_per_subject[this_sub]]

                # Loop through blocks
                for block in cfg.speech_blocks:
                    # Select data
                    data_thisblock = data[block][channel_indices,:]

                    # Run GSBS
                    filename_save = save_dir + "GSBS_" + str(sub1) + "vs" + str(sub2) + "_sub" + subject + "_block" + str(block) + ".npy"
                    if not os.path.exists(filename_save):
                        p = multiprocessing.Process(target=run_GSBS, args=(data_thisblock, filename_save,))
                        processes.append(p)
                        print("Starting GSBS " + filename_save)
                        p.start()

                    # If max number of processes have been reached, finish current jobs
                    if len(processes) >= max_nr_of_processes:
                        for process in processes:
                            process.join()
                        processes = []
