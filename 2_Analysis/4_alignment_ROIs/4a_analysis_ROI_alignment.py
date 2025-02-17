import config as cfg
from funcs import add_delay, match_per_run, get_relative_match, shuffle_states_per_run, load_adjusted_strengths
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import os
from scipy.stats import wilcoxon

# Get analysis parameters
blocks = cfg.speech_blocks
delays = cfg.ROI_delays
subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar
save_dir = cfg.dir_analysis_results + "4_alignment_ROIs/"
ROIs = ['low', 'high']

# Remove subject 12 as they did not have a low-level ROI
subjects.remove(12)

def get_matches_between_ROIs_over_delays(boundaries_low_orig, boundaries_high_orig):

    # Copy timelines
    boundaries_low = copy.deepcopy(boundaries_low_orig)
    boundaries_high = copy.deepcopy(boundaries_high_orig)

    # Zeropad timelines
    for block in blocks:
        boundaries_low[block] = np.insert(boundaries_low[block], 0, np.zeros(len(delays)))
        boundaries_high[block] = np.insert(boundaries_high[block], 0, np.zeros(len(delays)))

        boundaries_low[block] = np.append(boundaries_low[block], np.zeros(len(delays)))
        boundaries_high[block] = np.append(boundaries_high[block], np.zeros(len(delays)))

    matches = []
    for delay in delays:
        # Add delay to high ROI
        boundaries_high_delayed = copy.deepcopy(boundaries_high)
        for block in blocks:
            boundaries_high_delayed[block] = add_delay(boundaries_high_delayed[block], delay, cfg.GSBS_fs, cut_end=True)

        nr_bounds_high = np.sum([sum(A) for A in boundaries_high_delayed.values()])
        nr_bounds_low = np.sum([sum(A) for A in boundaries_low.values()])

        # ROI with lowest number of boundaries will be the seed
        if nr_bounds_high < nr_bounds_low:
            boundaries_seed = boundaries_high_delayed
            boundaries_other = boundaries_low
        else:
            boundaries_seed = boundaries_low
            boundaries_other = boundaries_high_delayed

        # Compute match
        match_thisdelay = match_per_run(boundaries_seed, boundaries_other)

        # Store match
        matches.append(match_thisdelay)
    return matches

def get_max_match(boundaries_low, boundaries_high, filename=None):
    matches = get_matches_between_ROIs_over_delays(boundaries_low, boundaries_high)
    max_match = np.max(matches)
    opt_delay = delays[np.argmax(matches)]

    # Save matches over delay if filename is given
    if filename is not None:
        np.save(filename, matches)

    return max_match, opt_delay

def get_stat_match(boundaries_low, boundaries_high, filename=None, to_return=None):
    # Function to do permutation test
    max_match_thisSubject, opt_delay = get_max_match(boundaries_low, boundaries_high,filename=filename)

    if to_return == 'delay':
        return opt_delay
    if to_return == 'match':
        return max_match_thisSubject


# Loop through subjects
matches_raw = {}
matches_relative = {}
measure_per_subject = {}
optimal_delays = {}
for n in subjects:
    subject = str(n).zfill(2)
    print("__________" + subject + "__________")

    # Create subject dir
    dir_sub = save_dir +  'sub-' + subject + '/'
    if not os.path.isdir(dir_sub):
        os.mkdir(dir_sub)

    # Load boundaries
    boundaries_low = load_adjusted_strengths(subject,'low',blocks)
    boundaries_high = load_adjusted_strengths(subject, 'high', blocks)

    # Plot boudnaries
    f, ax = plt.subplots(len(blocks), 1, figsize=(30, 20))
    for block_idx, block in enumerate(blocks):
        plt.sca(ax[block_idx])
        for row, timeline in enumerate([boundaries_low[block],boundaries_high[block]]):
            plt.eventplot(np.where(timeline > 0), orientation='horizontal', lineoffsets=row)
        plt.yticks(np.arange(2), ['low', 'high'])
    plt.xlabel("Timepoint")
    plt.tight_layout()
    plt.savefig(dir_sub + "timelines")

    # Convert to deltas
    for block in blocks:
        boundaries_low[block] = (boundaries_low[block] > 0).astype(int)
        boundaries_high[block] = (boundaries_high[block] > 0).astype(int)


    # Get matches over delays
    match_over_delay_thisSubject = get_matches_between_ROIs_over_delays(boundaries_low, boundaries_high)

    # Get and store optimal delay of this subject
    opt_delay = delays[np.argmax(match_over_delay_thisSubject)]
    optimal_delays[n] = opt_delay

    # Store raw match over delay
    matches_raw[n] = match_over_delay_thisSubject

    # Plot matches over delay
    if n in cfg.subjects_analysis_novel:
        group = 'novel'
    else:
        group = 'familiar'
    plt.figure()
    plt.plot(delays, match_over_delay_thisSubject, c='blue')
    plt.xlabel("Delay of High ROI (s)")
    plt.ylabel("Raw match")
    plt.title("Sub-" + subject + " (" + group + ")")
    plt.axvline(opt_delay, color='blue', linestyle='--')
    plt.savefig(dir_sub + 'match_over_delay')

    # Create null distribution
    curves_null = np.ones((cfg.nr_permutations, len(delays)))
    for iteration in tqdm(range(cfg.nr_permutations)):
        # Shuffle states (of ROI-low)
        boundaries_low_shuffled = shuffle_states_per_run(boundaries_low)

        # Get and store match over delay of this permutation
        matches_permutation = get_matches_between_ROIs_over_delays(boundaries_low_shuffled, boundaries_high)
        curves_null[iteration] = matches_permutation

    # Save null curves
    np.save(dir_sub + 'permutation_curves', curves_null)

    # Compute relative Gaussian match
    mean_null = np.mean(np.max(curves_null, axis=1))
    matches_relative[n] = get_relative_match(match_over_delay_thisSubject, mean_null)
    measure_per_subject[n] = np.max(matches_relative[n])
    print("Measure this subject: " + str(measure_per_subject[n]))

    # Plot relative gaussian
    plt.figure()
    plt.plot(delays, matches_relative[n], c='blue')
    plt.xlabel("Delay of High ROI (s)")
    plt.ylabel("Relative match")
    plt.title("Sub-" + subject + " (" + group + ")")
    plt.axvline(opt_delay, color='blue', linestyle='--')
    plt.axhline(0, color='k')
    plt.savefig(dir_sub + 'match_over_delay_relative')

    plt.close('all')

# Wilcoxon test for amplitude
results = {}
W = wilcoxon(list(measure_per_subject.values()), alternative='greater')
results['stats'] = W.statistic
p = W.pvalue
results['p'] = p

# Save info
np.save(save_dir + 'wilcoxon_results', results)
np.save(save_dir + 'matches_raw_over_ROIdelays', matches_raw)
np.save(save_dir + 'matches_relative_over_ROIdelays', matches_relative)
np.save(save_dir + 'optimal_ROIdelay', optimal_delays)
np.save(save_dir + 'measure_per_subject', measure_per_subject)

# Print some info
print('optimal delays = ' + str(optimal_delays))
print('measure per subject = ' + str(measure_per_subject))
print('p = ' + str(p))


