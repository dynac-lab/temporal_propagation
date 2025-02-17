import config as cfg
from funcs import run_GSBS
import numpy as np
import os
import multiprocessing

max_nr_of_processes = 13
band = cfg.band
save_dir = cfg.dir_GSBS
processes = []
subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar

# Loop over subjects
for n in subjects:
    subject = str(n).zfill(2)

    for ROI in ['low', 'high']:
        filename_data = cfg.dir_preGSBS + "preGSBS_ROI-" + ROI + "_" + subject + "_data_" + band + ".npy"
        data = np.load(filename_data, allow_pickle=True).item()

        # Run GSBS per block
        for block in cfg.speech_blocks:
            filename_save = save_dir + "GSBS_" + subject + "_" + band + "_ROI-" + ROI + "_block" + str(block) + ".npy"
            if not os.path.exists(filename_save):
                p = multiprocessing.Process(target=run_GSBS, args=(data[block], filename_save,))
                processes.append(p)
                print("Starting GSBS " + filename_save)
                p.start()

        # If max number of processes have been reached, finish current jobs
        if len(processes) >= max_nr_of_processes:
            for process in processes:
                process.join()
            processes = []