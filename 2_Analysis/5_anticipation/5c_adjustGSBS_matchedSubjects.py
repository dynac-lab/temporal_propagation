import config as cfg
from funcs import zeropad_tdistances, get_peaks
import numpy as np

load_dir = cfg.dir_GSBS
band = cfg.band

save_dir = cfg.dir_analysis_results + '5_anticipation/match_over_delays/'
GSBS_dir = cfg.dir_GSBS + 'matched_subjects/'
matrix_channel_overlap = np.load(cfg.dir_analysis_results + '5_anticipation/matrix_channel_overlap.npy')
subjects = np.load(cfg.dir_analysis_results + '5_anticipation/matrix_channel_overlap_subjects.npy')
N = len(subjects)

def get_k_per_block(pair_key, subject):
    nr_of_states_per_block = {}
    t_distances = []
    GSBS_objects = {}
    for block in list(cfg.speech_blocks):
        # Load GSBS results
        filename = GSBS_dir +  "GSBS_" + pair_key + "_sub" + subject + "_block" + str(block) + ".npy"
        GSBS_obj = np.load(filename, allow_pickle=True).item()
        GSBS_objects[block] = GSBS_obj

        # Get t-distance
        t_distances.append(GSBS_obj.tdists)

    t_distances = np.asarray(zeropad_tdistances(t_distances))
    t_mean = np.mean(t_distances, axis=0)
    k_opt = np.argmax(t_mean)

    # Per block, identify peaks and define the number of states as the closest peak to k_opt
    for block in list(cfg.speech_blocks):
        k_peaks = get_peaks(GSBS_objects[block].tdists)
        closest_peak_idx = np.argmin(np.abs(k_peaks - k_opt))
        k = k_peaks[closest_peak_idx]
        nr_of_states_per_block[block] = k
    return nr_of_states_per_block

# Loop through pairs of subjects
k_per_subjectpair = {}
for n1 in range(N):
    for n2 in  range(N):
        sub1 = subjects[n1]
        sub2 = subjects[n2]
        if matrix_channel_overlap[n1, n2] >= 15 and sub1 < sub2:
            subject1 = str(sub1).zfill(2)
            subject2 = str(sub2).zfill(2)
            pair_key = str(sub1) + "vs" + str(sub2)

            # Get info
            k_per_subjectpair[pair_key] = {}
            k_per_subjectpair[pair_key][subject1] = get_k_per_block(pair_key,subject1)
            k_per_subjectpair[pair_key][subject2] = get_k_per_block(pair_key,subject2)

np.save(save_dir + 'adjusted_number_of_states', k_per_subjectpair)

