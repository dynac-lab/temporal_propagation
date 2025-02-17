#!/bin/sh
# Set the path to your data directory
data_dir=XXX

# Set the output directory
output_dir=YYY

# Set subject dir
SUBJECTS_DIR=ZZZ

# Loop through subjects from 1 to 63
for subject_id in {1..63}; do
    subject=sub-$(printf "%02d" $subject_id)  # Format subject ID with leading zeros if needed
	mri_annotation2label --subject $subject --hemi lh --labelbase lh.BN_Atlas.annot --annotation BN_Atlas --surf orig
done
