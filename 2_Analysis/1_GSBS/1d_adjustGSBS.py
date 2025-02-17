import config as cfg
from funcs import plot_time_correlation, zeropad_tdistances, plot_line_with_shade,get_peaks
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar
load_dir = cfg.dir_GSBS
band = cfg.band

# Loop through ROIs
for ROI in ['low', 'high']:
    nr_of_states_per_subject_per_block = {}
    ROI_name = 'ROI-' + ROI

    # Loop through subjects
    for n in tqdm(subjects):
        plt.close('all')
        if ROI == 'low' and n == 12:
            # sub-12 does not have low-level channels
            continue

        # Set info
        subject = str(n).zfill(2)
        nr_of_states_per_subject_per_block[subject] = {}
        dir_sub = cfg.dir_analysis_results +  'GSBS/GSBS adjustments/plots_' + 'sub-' + subject + '/'

        # Get mean t-distances
        t_distances = []
        GSBS_objects = {}
        for block in list(cfg.speech_blocks):

            # Load GSBS results
            filename = load_dir + "GSBS_" + subject + "_" + band + "_" + ROI_name + "_block" + str(block) + ".npy"
            GSBS_obj = np.load(filename, allow_pickle=True).item()
            GSBS_objects[block] = GSBS_obj

            # Get t-distance
            t_distances.append(GSBS_obj.tdists)

        t_distances = np.asarray(zeropad_tdistances(t_distances))
        t_mean = np.mean(t_distances, axis=0)
        k_opt = np.argmax(t_mean)
        t_std = np.std(t_distances, axis=0)

        # Per block, identify peaks and define the number of states as the closest peak to k_opt
        for block in list(cfg.speech_blocks):
            k_peaks = get_peaks(GSBS_objects[block].tdists)
            closest_peak_idx = np.argmin(np.abs(k_peaks - k_opt))
            k = k_peaks[closest_peak_idx]
            nr_of_states_per_subject_per_block[subject][block] = k

        # Plot time correlation and t-curve with chosen nstates, pre- and post- adjustments
        for block in cfg.speech_blocks:
            GSBS_obj = GSBS_objects[block]
            f, ax = plt.subplots(2,2,figsize=(20,20))

            # Average t-distance
            plt.sca(ax[0,0])
            plot_line_with_shade(range(len(t_mean)),t_mean, t_std, facecolor='gray')
            plt.ylabel("Mean t-distances across speech blocks")
            plt.xlabel('Number of states')
            plt.title("k_opt = " + str(k_opt) + " (Mean duration of " + str(np.round(30 / k_opt, 3)) + " seconds)")


            # This t-distance
            plt.sca(ax[0,1])
            plt.plot(GSBS_obj.tdists)
            plt.ylabel('t-distance this block')
            plt.xlabel('Number of states')
            plt.title("Block " + str(block))
            plt.vlines(GSBS_obj.nstates, 0, np.max(GSBS_obj.tdists), 'b', label="Original max")
            plt.vlines(k_opt, 0, np.max(GSBS_obj.tdists), 'k', label="k_opt")
            plt.vlines(nr_of_states_per_subject_per_block[subject][block], 0, np.max(GSBS_obj.tdists), 'r', label="k_adjusted")
            plt.legend()

            # Old boundaries
            plt.sca(ax[1,0])
            plot_time_correlation(ax[1,0], GSBS_obj.x, GSBS_obj)
            plt.title("Boundaries original (nstates = " + str(GSBS_obj.nstates) + ")")

            # New boundaries
            plt.sca(ax[1, 1])
            plot_time_correlation(ax[1, 1], GSBS_obj.x, GSBS_obj, nr_of_states_per_subject_per_block[subject][block])
            plt.title("Boundaries adjusted (nstates = " + str(nr_of_states_per_subject_per_block[subject][block]) + ")")

            plt.savefig(dir_sub + "Adjustment_effect_"  + ROI_name + "_block" + str(block))

    # Save nr_of_states_per_subject_per_block for this ROI
    np.save(cfg.dir_analysis_results +  'GSBS/GSBS adjustments/' + 'states_per_subject_' + ROI_name, nr_of_states_per_subject_per_block)