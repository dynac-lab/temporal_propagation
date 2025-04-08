import config as cfg
import numpy as np
from funcs import add_delay, match_per_run
import copy
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


save_dir = cfg.dir_analysis_results + 'RQ4_prediction/match_over_delays/'
GSBS_dir = cfg.dir_GSBS + 'matched_subjects/'
matrix_channel_overlap = np.load(cfg.dir_analysis_results + 'RQ4_prediction/matrix_channel_overlap.npy')
subjects = np.load(cfg.dir_analysis_results + 'RQ4_prediction/matrix_channel_overlap_subjects.npy')
delays = np.arange(- 0.609375, 0.61, 1/cfg.GSBS_fs)
adjusted_number_of_states = np.load( cfg.dir_analysis_results + 'RQ4_prediction/' + 'adjusted_number_of_states.npy', allow_pickle=True).item()


def get_match_over_delays(bounds_sub1, bounds_sub2):
    # Determine the seed subject based on the number of boundaries, before adding delays
    nr_bounds_sub1 = np.sum([sum(A) for A in bounds_sub1.values()])
    nr_bounds_sub2 = np.sum([sum(A) for A in bounds_sub2.values()])
    if nr_bounds_sub1 < nr_bounds_sub2:
        seed_sub = 'sub1'
    else:
        seed_sub = 'sub2'

    # Zero-pad both timelines
    for block in cfg.speech_blocks:
        bounds_sub1[block] = np.insert(bounds_sub1[block], 0, np.zeros(len(delays)))
        bounds_sub2[block] = np.insert(bounds_sub2[block], 0, np.zeros(len(delays)))

        bounds_sub1[block] = np.append(bounds_sub1[block], np.zeros(len(delays)))
        bounds_sub2[block] = np.append(bounds_sub2[block], np.zeros(len(delays)))

    matches = []
    for delay in delays:
        # Add delay to first subject
        boundaries_sub1_delayed = copy.deepcopy(bounds_sub1)
        for block in cfg.speech_blocks:
            # Add delay to sub1
            boundaries_sub1_delayed[block] = add_delay(boundaries_sub1_delayed[block], delay, cfg.GSBS_fs)

            # Make the two timelines of the same length again
            if len(boundaries_sub1_delayed[block]) < len(bounds_sub2[block]):
                bounds_sub2[block] = bounds_sub2[block][:len(boundaries_sub1_delayed[block])]
            if len(boundaries_sub1_delayed[block]) > len(bounds_sub2[block]):
                boundaries_sub1_delayed[block] = boundaries_sub1_delayed[block][:len(bounds_sub2[block])]

        # ROI with lowest number of boundaries will be the seed
        if seed_sub == 'sub1':
            boundaries_seed = boundaries_sub1_delayed
            boundaries_other = bounds_sub2
        if seed_sub == 'sub2':
            boundaries_seed = bounds_sub2
            boundaries_other = boundaries_sub1_delayed

        # Compute match
        match_thisdelay = match_per_run(boundaries_seed, boundaries_other)

        # Store match
        matches.append(match_thisdelay)
    return matches

matrix_optimal_delay = np.ones_like(matrix_channel_overlap) * np.nan
for n1, sub1 in enumerate(tqdm(subjects)):
    for n2, sub2 in enumerate(subjects):
        if matrix_channel_overlap[n1,n2] >= 15: # GSBS only done when 15 channels are overlapping
            boundaries = {}
            for this_sub in [sub1, sub2]:
                subject = str(this_sub).zfill(2)
                lowest_sub = min([sub1, sub2])
                highest_sub = max([sub1, sub2])


                # Load GSBS results
                boundaries[this_sub] = {}
                for block in cfg.speech_blocks:
                    nr_of_states = adjusted_number_of_states[str(lowest_sub) + "vs" + str(highest_sub)][str(this_sub).zfill(2)][block]
                    filename = GSBS_dir +  "GSBS_" + str(lowest_sub) + "vs" + str(highest_sub) + "_sub" + subject + "_block" + str(block) + ".npy"
                    GSBS_obj = np.load(filename, allow_pickle=True).item()
                    boundaries[this_sub][block] = GSBS_obj.get_deltas(adjusted_number_of_states[str(lowest_sub) + "vs" + str(highest_sub)][str(this_sub).zfill(2)][block])
            match_over_delay = get_match_over_delays(boundaries[sub1], boundaries[sub2])
            optimal_delay = delays[np.argmax(match_over_delay)]
            matrix_optimal_delay[n1,n2] = optimal_delay

            # Plot match over delay fpr this subject pair
            plt.figure()
            plt.plot(delays, match_over_delay)
            plt.axvline(optimal_delay)
            plt.title(str(sub1) + ' vs ' + str(sub2))
            plt.tight_layout()
            plt.savefig(save_dir + str(sub1) + '_vs_' + str(sub2))
            plt.close('all')

# Plot matrix
plt.figure()
plt.imshow(matrix_optimal_delay, cmap='bwr', vmin=delays[0], vmax=delays[-1], interpolation='none')
plt.yticks(range(len(subjects)), subjects)
plt.xticks(range(len(subjects)), subjects)
for nan in np.argwhere(np.isnan(matrix_optimal_delay)):
    plt.plot(nan[1], nan[0], 'kx')
plt.colorbar()
plt.tight_layout()
plt.savefig(cfg.dir_analysis_results + 'RQ4_prediction/' + 'matrix_optimal_delays')

n_idx_familiar = [i for i in range(len(subjects)) if subjects[i] in cfg.subjects_analysis_familiar]
n_idx_novel = [i for i in range(len(subjects)) if subjects[i] in cfg.subjects_analysis_novel]

novel_quadrant = matrix_optimal_delay[n_idx_novel,:][:,n_idx_novel]
familiar_quadrant = matrix_optimal_delay[n_idx_familiar,:][:,n_idx_familiar]
f_n = matrix_optimal_delay[n_idx_familiar,:][:,n_idx_novel]
n_f = matrix_optimal_delay[n_idx_novel,:][:,n_idx_familiar]

# Save matrix
np.save(cfg.dir_analysis_results + 'RQ4_prediction/' + 'matrix_optimal_delays', matrix_optimal_delay)

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
tail_low_count = np.sum(np.asarray(measure_null) >= measure_data)
tail_high_count = np.sum(np.asarray(measure_null) <= measure_data)

#  Two-tailed, because after implementing GSBS adjustment, measure was below zero
p = (min(tail_low_count, tail_high_count) / nr_permutations)*2
print("p = " + str(p))

np.save(cfg.dir_analysis_results + 'RQ4_prediction/' + 'p_value', p)