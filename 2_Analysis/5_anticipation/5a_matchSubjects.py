import config as cfg
import numpy as np
import matplotlib.pyplot as plt
import random

subjects = cfg.subjects_analysis_novel + cfg.subjects_analysis_familiar

# Per subject, count how many channels they have in each BN label
BN_labels = np.asarray(list(cfg.atlas_BN.values())[1:]) # skip label 'Unknown'
channel_count = np.zeros((len(BN_labels), len(subjects)))
for n_idx, n in enumerate(subjects):
    subject = str(n).zfill(2)
    label_per_channel = np.load(cfg.dir_electrode_labels + "sub-" + subject + "/labels_BN_sub" + subject + '.npy', allow_pickle=True).item()
    for ch in label_per_channel.keys():
        if ch not in cfg.bad_channels[subject]:
            label = label_per_channel[ch]
            if label != 'Unknown':
                label_idx = np.where(BN_labels == label)[0][0]
                channel_count[label_idx, n_idx] +=1

# Compute how many channels can be taken together across all subjects
subject_matrix = np.zeros((len(subjects), len(subjects)))
for n1_idx, n1 in enumerate(subjects):
    for n2_idx, n2 in enumerate(subjects):
        these_subjects = channel_count[:,(n1_idx,n2_idx)]
        subject_matrix[n1_idx, n2_idx] = np.sum(np.min(these_subjects, axis=1))

# Fill diagonal with nans
np.fill_diagonal(subject_matrix, np.nan)

# Plot
plt.figure()
plt.imshow(subject_matrix)
for (j,i), label in np.ndenumerate(subject_matrix):
    if i!=j:
        plt.text(i,j,int(label),ha='center', va='center')
plt.yticks(range(len(subjects)), subjects)
plt.xticks(range(len(subjects)), subjects)
plt.tight_layout()
plt.savefig(cfg.dir_analysis_results + '5_anticipation/matrix_channel_overlap')
np.save(cfg.dir_analysis_results + '5_anticipation/matrix_channel_overlap',subject_matrix)
np.save(cfg.dir_analysis_results + '5_anticipation/matrix_channel_overlap_subjects', subjects)

# Loop over all pairs of subjects
for sub1 in subjects:
    for sub2 in subjects:

        # Only look into this pair once
        if int(sub1) < int(sub2):
            n1_idx = np.where(np.asarray(subjects) == sub1)[0][0]
            n2_idx = np.where(np.asarray(subjects) == sub2)[0][0]
            count_these_subjects = channel_count[:,(n1_idx,n2_idx)]
            channel_selection_per_subject = dict((n, []) for n in [sub1,sub2])

            # Loop over all areas in the atlas
            for l_idx, label in enumerate(BN_labels):

                # If both subjects have at least one channel
                if np.min(count_these_subjects[l_idx, :]) > 0:

                    # Per subject, get all channels in this area
                    channels_in_area = dict((n, []) for n in [sub1,sub2])
                    for n in [sub1, sub2]:
                        subject = str(n).zfill(2)
                        label_per_channel = np.load(cfg.dir_electrode_labels + "sub-" + subject + "/labels_BN_sub" + subject + '.npy',allow_pickle=True).item()
                        for chan in label_per_channel.keys():
                            if label_per_channel[chan] == label and chan not in cfg.bad_channels[subject]:
                                channels_in_area[n].append(chan)

                    # Adjust channel selection if one subject has more channels than the other
                    if len(channels_in_area[sub1]) < len(channels_in_area[sub2]):
                        # Add all channels of sub1
                        channel_selection_per_subject[sub1].extend(channels_in_area[sub1])
                        # Select random channel(s) for sub2
                        nr_of_channels = len(channels_in_area[sub1])
                        channel_selection_per_subject[sub2].extend(random.sample(channels_in_area[sub2], nr_of_channels))
                    elif len(channels_in_area[sub1]) > len(channels_in_area[sub2]):
                        # Select random channel(s) for sub1
                        nr_of_channels = len(channels_in_area[sub2])
                        channel_selection_per_subject[sub1].extend(random.sample(channels_in_area[sub1], nr_of_channels))
                        # Add all channels of sub1
                        channel_selection_per_subject[sub2].extend(channels_in_area[sub2])

                    else: # Add all channels of both subjects
                        channel_selection_per_subject[sub1].extend(channels_in_area[sub1])
                        channel_selection_per_subject[sub2].extend(channels_in_area[sub2])

            # Save channel selection
            np.save(cfg.dir_analysis_results + '5_anticipation/matching_channels/' + str(sub1) + "vs" + str(sub2), channel_selection_per_subject)

            # Plot
            for n in [sub1, sub2]:
                subject = str(n).zfill(2)
                fig = plt.figure(figsize=(15, 10))
                ax = fig.add_subplot(projection='3d')
                coords_per_channel = np.load(cfg.dir_electrode_labels + "sub-" + subject + "/coords_per_channel_" + subject + '.npy', allow_pickle=True).item()

                # Get coords included and excluded channels seperately
                coords_incl = np.asarray([coords_per_channel[ch_name] for ch_name in channel_selection_per_subject[n]])
                coords_excl = np.asarray([coords_per_channel[ch_name] for ch_name in coords_per_channel.keys() if ch_name not in channel_selection_per_subject[n] and ch_name not in cfg.bad_channels[subject]])


                # Plot
                ax.scatter(coords_incl[:, 0], coords_incl[:, 1], coords_incl[:, 2], marker='o', edgecolors='black', s=40, label='incl', c='red')
                ax.scatter(coords_excl[:, 0], coords_excl[:, 1], coords_excl[:, 2], marker='o', edgecolors='black', s=40, label='excl', c='blue')


                # Finalize and save
                ax.view_init(0, -180)
                plt.legend()
                plt.tight_layout()
                plt.savefig(cfg.dir_analysis_results + '5_anticipation/matching_channels/' + str(sub1) + "vs" + str(sub2) + "_" + str(subject))
            plt.close('all')




