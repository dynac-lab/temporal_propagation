import config as cfg
import numpy as np
from scipy.stats import wilcoxon

if __name__ == "__main__":
    # Get analysis parameters
    blocks = cfg.speech_blocks
    annotation_category = 'clauses'
    save_dir = cfg.dir_analysis_results + "2_alignment_speech/"
    ROIs = ['low', 'high']
    subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar

    optimal_delays = np.load(save_dir + 'optimal_delays.npy', allow_pickle=True).item()
    matches_relative = np.load(save_dir + 'matches_relative_over_delays.npy', allow_pickle=True).item()

    # Loop through subjects
    measure_per_subject = []
    for n in subjects:
        if n == 12:
            continue # Subject 12 does not have low ROI

        if np.sum([np.max(matches_relative['clauses'][ROI][n]) > 0 for ROI in ROIs]) < len(ROIs): # At least one ROI had a below-chance alignment
            print("Skipping " + str(n) + " due to below-level alignment in at least one ROI")
            continue


        optimal_delay_diff = optimal_delays[n]['high'][annotation_category] - optimal_delays[n]['low'][annotation_category]
        measure_per_subject.append(optimal_delay_diff)

    print('Mean measure = ' + str(np.mean(measure_per_subject)))
    print('SD measure = ' + str(np.std(measure_per_subject)))

    # Wilcoxon signed-rank test
    W = wilcoxon(list(measure_per_subject), alternative='two-sided')
    w_stats = W.statistic
    p_values = W.pvalue

    # Save results
    np.save(save_dir + "wilcoxon_delaydiff_pvalues", p_values)
    np.save(save_dir + "wilcoxon_delaydiff_statistics", w_stats)

    print("Measure:")
    print(measure_per_subject)
    print("P values:")
    print(p_values)

    print("Statistics:")
    print(w_stats)
