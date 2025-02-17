import config as cfg
import numpy as np
from funcs import add_delay, match_per_run, timepoints_to_timeline
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pandas as pd

save_dir = cfg.dir_analysis_results + '5_anticipation/delay_clauses/'
GSBS_dir = cfg.dir_GSBS + 'matched_subjects/'
matrix_channel_overlap = np.load(cfg.dir_analysis_results + '5_anticipation/matrix_channel_overlap.npy')
subjects = np.load(cfg.dir_analysis_results + '5_anticipation/matrix_channel_overlap_subjects.npy')
delays = cfg.speech_delays
adjusted_number_of_states = np.load(cfg.dir_analysis_results + '5_anticipation/' + 'adjusted_number_of_states.npy', allow_pickle=True).item()
blocks = cfg.speech_blocks


def get_match_over_delays(annotations, bounds_subject):

    annotations_padded = copy.deepcopy(annotations)

    # Zero-pad both timelines
    for block in cfg.speech_blocks:
        annotations_padded[block] = np.insert(annotations_padded[block], 0, np.zeros(len(delays)))
        bounds_subject[block] = np.insert(bounds_subject[block], 0, np.zeros(len(delays)))

        annotations_padded[block] = np.append(annotations_padded[block], np.zeros(len(delays)))
        bounds_subject[block] = np.append(bounds_subject[block], np.zeros(len(delays)))

    matches = []
    for delay in delays:
        # Add delay to annotations
        annotations_delayed = copy.deepcopy(annotations_padded)
        for block in cfg.speech_blocks:
            # Add delay to sub1
            annotations_delayed[block] = add_delay(annotations_delayed[block], delay, cfg.GSBS_fs)

            # Make the two timelines of the same length again
            if len(annotations_delayed[block]) < len(bounds_subject[block]):
                bounds_subject[block] = bounds_subject[block][:len(annotations_delayed[block])]
            if len(annotations_delayed[block]) > len(bounds_subject[block]):
                annotations_delayed[block] = annotations_delayed[block][:len(bounds_subject[block])]


        # Compute match
        match_thisdelay = match_per_run(annotations_delayed, bounds_subject)

        # Store match
        matches.append(match_thisdelay)
    return matches


# Get annotations per run
annotations = dict((block, []) for block in blocks)
label = 'clauses'
filename = cfg.dir_annotations + "sound/sound_annotation_" + label + ".tsv"
annotations_file = pd.read_csv(filename, sep='\t')
timepoints_onset = annotations_file.onset.values
timepoints_offset = annotations_file.offset.values
timepoints = np.unique(list(timepoints_onset) + list(timepoints_offset))
for block in blocks:
    idx_bigenough = (timepoints > block * 30)
    idx_smallenough = (timepoints < (block + 1) * 30)
    annotations[block] = timepoints_to_timeline(timepoints[np.logical_and(idx_bigenough, idx_smallenough)] - 30 * block, fs=cfg.GSBS_fs, duration=30)


# Compute optimal delay with clauses per subject pair
matrix_optimal_delay = np.ones_like(matrix_channel_overlap) * np.nan
for n1, sub1 in enumerate(tqdm(subjects)):
    for n2, sub2 in enumerate(subjects):
        if matrix_channel_overlap[n1,n2] >= 15:
            boundaries = {}
            match_over_delay = {}
            optimal_delays = {}
            for this_sub in [sub1, sub2]:
                subject = str(this_sub).zfill(2)

                # Load GSBS results
                boundaries[this_sub] = {}
                for block in cfg.speech_blocks:
                    lowest_sub = min([sub1, sub2])
                    highest_sub = max([sub1, sub2])
                    filename = GSBS_dir +  "GSBS_" + str(lowest_sub) + "vs" + str(highest_sub) + "_sub" + subject + "_block" + str(block) + ".npy"
                    GSBS_obj = np.load(filename, allow_pickle=True).item()
                    boundaries[this_sub][block] = GSBS_obj.get_deltas(adjusted_number_of_states[str(lowest_sub) + "vs" + str(highest_sub)][str(this_sub).zfill(2)][block])

                # Compute match with annotations
                match_over_delay[this_sub] = get_match_over_delays(annotations, boundaries[this_sub])
                optimal_delays[this_sub] = delays[np.argmax(match_over_delay[this_sub])]

            matrix_optimal_delay[n1,n2] = optimal_delays[sub1] - optimal_delays[sub2]


# Save matrix
np.save(cfg.dir_analysis_results + 'RQ4_prediction/' + 'matrix_difference_optimal_delays_with_clauses', matrix_optimal_delay)

# Get relevant values from matrix
n_idx_familiar = [i for i in range(len(subjects)) if subjects[i] in cfg.subjects_analysis_familiar]
n_idx_novel = [i for i in range(len(subjects)) if subjects[i] in cfg.subjects_analysis_novel]

novel_quadrant = matrix_optimal_delay[n_idx_novel,:][:,n_idx_novel]
familiar_quadrant = matrix_optimal_delay[n_idx_familiar,:][:,n_idx_familiar]
f_n = matrix_optimal_delay[n_idx_familiar,:][:,n_idx_novel]
n_f = matrix_optimal_delay[n_idx_novel,:][:,n_idx_familiar]

# Permutation test
nr_permutations = 10000
measure_data = np.nanmedian(f_n)
print("Optimal delay familiar to novel = " + str(measure_data))
measure_null = []
print('Permutation test...')
for iteration in tqdm(np.arange(nr_permutations)):
    # Shuffle subject order
    subjects_shuffled = subjects.copy()
    random.shuffle(subjects_shuffled)

    # Create permuted matrix
    matrix_perm = np.zeros_like(matrix_optimal_delay)
    for n1, sub1 in enumerate(subjects_shuffled):
        for n2, sub2 in enumerate(subjects_shuffled):
            value = matrix_optimal_delay[subjects == sub1, subjects == sub2][0]
            matrix_perm[n1,n2] = value

    # Get measure of this permutation
    f_n_perm = matrix_perm[n_idx_familiar, :][:, n_idx_novel]
    measure_null.append(np.nanmedian(f_n_perm))

# Compute p (one-tailed, bigger)
p = np.sum(measure_null >= measure_data) / nr_permutations
print("p = " + str(p))

# Save p
np.save(cfg.dir_analysis_results + 'RQ4_prediction/' + 'p_value_optimal_delay_with_clauses', p)