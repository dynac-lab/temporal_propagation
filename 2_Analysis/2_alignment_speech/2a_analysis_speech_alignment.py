import config as cfg
from funcs import timepoints_to_timeline, add_delay, match_per_run, get_relative_match, shuffle_states_per_run, load_adjusted_strengths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from scipy.stats import wilcoxon




def match_per_annotation_category(data, annotations):
    # Returns a dictionary, one key per label in annotations
    matches = {}
    for label in annotations.keys():
        matches[label] = match_per_run(annotations[label], data)
    return matches

def get_matches_over_delays(boundaries_orig, annotations_orig):
    boundaries = copy.deepcopy(boundaries_orig)
    annotations = copy.deepcopy(annotations_orig)

    # Zero-pad both timelines
    for block in cfg.speech_blocks:
        boundaries[block] = np.insert(boundaries[block], 0, np.zeros(len(delays)))
        boundaries[block] = np.append(boundaries[block], np.zeros(len(delays)))

        for label in annotation_categories:
            annotations[label][block] = np.insert(annotations[label][block], 0, np.zeros(len(delays)))
            annotations[label][block] = np.append(annotations[label][block], np.zeros(len(delays)))


    matches = dict((label, []) for label in annotations.keys())
    for delay in cfg.speech_delays:
        annotations_delayed = copy.deepcopy(annotations)
        for block in cfg.speech_blocks:
            # Add delay to annotations
            for label in cfg.speech_annotation_categories:
                annotations_delayed[label][block] = add_delay(annotations_delayed[label][block], delay, cfg.GSBS_fs, cut_end=True)

            # Make the two timelines of the same length again
            if len(annotations_delayed[annotation_categories[0]][block]) < len(boundaries[block]):
                boundaries[block] = boundaries[block][:len(annotations_delayed[annotation_categories[0]][block])]
            if len(annotations_delayed[annotation_categories[0]][block]) > len(boundaries[block]):
                for label in annotation_categories:
                    annotations_delayed[label][block] = annotations_delayed[label][block][:len(boundaries[block])]

        # Compute match
        match_thisdelay = match_per_annotation_category(boundaries, annotations_delayed)

        # Store match
        for label in cfg.speech_annotation_categories:
            matches[label].append(match_thisdelay[label])
    return matches

def get_max_match(boundaries, annotations, filename=None):
    matches = get_matches_over_delays(boundaries, annotations)
    max_match= {}
    opt_delay = {}
    for label in annotations.keys():
        max_match[label] = np.max(matches[label])
        opt_delay[label] = cfg.speech_delays[np.argmax(matches[label])]

    # Save matches over delay if filename is given
    if filename is not None:
        np.save(filename, matches)

    return max_match, opt_delay


if __name__ == "__main__":
    # Get analysis parameters
    blocks = cfg.speech_blocks
    annotation_categories = cfg.speech_annotation_categories
    delays = cfg.speech_delays
    subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar
    save_dir = cfg.dir_analysis_results + "2_alignment_speech/"
    ROIs = ['low', 'high']

    # Get timings per annotation category per run
    annotations = dict((label, dict((block, []) for block in blocks)) for label in annotation_categories)
    for label in annotation_categories:
        filename = cfg.dir_annotations + "sound/sound_annotation_" + label + ".tsv"
        annotations_file = pd.read_csv(filename, sep='\t')
        timepoints_onset = annotations_file.onset.values
        timepoints_offset = annotations_file.offset.values
        timepoints = np.unique(list(timepoints_onset) + list(timepoints_offset))

        for block in blocks:
            idx_bigenough = (timepoints > block * 30)
            idx_smallenough = (timepoints < (block + 1) * 30)
            annotations[label][block] = timepoints_to_timeline(timepoints[np.logical_and(idx_bigenough, idx_smallenough)] - 30 * block, fs = cfg.GSBS_fs, duration = 30)

    # Plot annotations
    f, ax = plt.subplots(len(blocks), 1, figsize=(30,20))
    for block_idx, block in enumerate(blocks):
        mat = []
        for label in annotation_categories:
            mat.append(annotations[label][block])
        plt.sca(ax[block_idx])
        plt.imshow(mat, interpolation='none', aspect='auto')
        plt.yticks(np.arange(len(annotation_categories)), annotation_categories)
    plt.ylabel("Timepoint")
    plt.xlabel("Delay (s)")
    plt.tight_layout()
    plt.savefig(save_dir + "annotations")

    # Loop through subjects
    curve_data_allSubs = dict((label, dict((ROI, dict((n, []) for n in subjects)) for ROI in ROIs)) for label in annotation_categories)
    curve_relative_data_allSubs = dict((label, dict((ROI, dict((n, []) for n in subjects)) for ROI in ROIs)) for label in annotation_categories)
    curves_permutations_data_allSubs = dict((label, dict((ROI, dict((n, []) for n in subjects)) for ROI in ROIs)) for label in annotation_categories)
    measure_per_sub = dict((label, dict((ROI,{}) for ROI in ROIs)) for label in annotation_categories)
    optimal_delays = {}
    for n_idx, n in enumerate(subjects):
        subject = str(n).zfill(2)
        print("__________" + subject + "__________")
        match_per_delay_permutations = np.ones((len(delays), cfg.nr_permutations))
        optimal_delays[n] = {}

        # Prepare figure
        f, ax = plt.subplots(1,2, figsize=(20,10), sharey=True)
        for i in [0,1]:
            plt.sca(ax[i])
            plt.axhline(0, color='k')
            plt.title('Subject ' + subject + ", " + annotation_categories[i])
            plt.ylabel("Relative Gaussian match")
            plt.xticks(np.arange(len(delays)), delays)

        for ROI in ROIs:
            if n == 12 and ROI == 'low':
                continue

            ROI_name = 'ROI-' + ROI

            # Load GSBS results
            boundaries = load_adjusted_strengths(subject,ROI,blocks)

            # Convert to deltas
            for block in blocks:
                boundaries[block] = (boundaries[block] > 0).astype(int)

            # Get matches over delays
            matches = get_matches_over_delays(boundaries, annotations)

            # Store optimal delay of this ROI
            opt_delay = {}
            for label in annotations.keys():
                opt_delay[label] = cfg.speech_delays[np.argmax(matches[label])]
            optimal_delays[n][ROI] = opt_delay

            # Store
            for label in annotation_categories:
                curve_data_allSubs[label][ROI][n] = matches[label]

            # Permutations
            curves_null = dict((label, np.ones((cfg.nr_permutations, len(delays)))) for label in annotation_categories)
            for iteration in tqdm(range(cfg.nr_permutations)):
                # Shuffle states
                boundaries_shuffled = shuffle_states_per_run(boundaries)

                # Get and store match over delay
                matches_permutation = get_matches_over_delays(boundaries_shuffled, annotations) #filename=dir_sub + 'match_permutation' + str(iteration).zfill(5)
                for label in annotation_categories:
                    curves_null[label][iteration] = matches_permutation[label]

                curves_permutations_data_allSubs[label][ROI][n].append(curves_null[label][iteration])

            # For this subject this ROI, for both annotation categories
            for l_idx, label in enumerate(annotation_categories):
                # Compute relative match
                mean_null = np.mean(np.max(curves_null[label], axis=1))
                rel_match = get_relative_match(matches[label], mean_null)
                curve_relative_data_allSubs[label][ROI][n] = rel_match

                # Get and store measure of interest
                print(np.max(rel_match))
                measure_per_sub[label][ROI][n] = np.max(rel_match)

                # Plot relative match
                plt.sca(ax[l_idx])
                if ROI == 'low':
                    c = 'green'
                if ROI == 'high':
                    c = 'blue'
                plt.plot(delays, rel_match, label=ROI, color=c)
                plt.axvline(opt_delay[label], color=c, linestyle='--')

        # Finalize and save plot
        ax[0].legend()
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(save_dir + 'relative_match_over_delay' + '_sub-' + subject)



    # Wilcoxon signed-rank test
    p_values = dict((label, dict((ROI, 1) for ROI in ROIs)) for label in annotation_categories)
    w_stats =  dict((label, dict((ROI, 1) for ROI in ROIs)) for label in annotation_categories)
    for label in annotation_categories:
        for ROI in ROIs:
            W = wilcoxon(list(measure_per_sub[label][ROI].values()), alternative='greater')
            w_stats[label][ROI] = W.statistic
            p_values[label][ROI] = W.pvalue

    # Save data
    np.save(save_dir + "matches_raw_over_delays", curve_data_allSubs)
    np.save(save_dir + "matches_relative_over_delays", curve_relative_data_allSubs)
    np.save(save_dir + "matches_null_over_delays", curves_permutations_data_allSubs)
    np.save(save_dir + "optimal_delays", optimal_delays)


    np.save(save_dir + "wilcoxon_pvalues", p_values)
    np.save(save_dir + "wilcoxon_statistics", w_stats)

    print("P values:")
    print(p_values)

    print("Statistics:")
    print(w_stats)
