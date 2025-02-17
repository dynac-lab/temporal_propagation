#!/bin/sh
# Set the path to your data directory
export SUBJECTS_DIR=XXX

# Set the output directory
output_dir=YYY

# Loop through subjects from 1 to 63
for subject_id in {1..63}; do
    subject=sub-$(printf "%02d" $subject_id)  # Format subject ID with leading zeros if needed
    mri_surf2surf --srcsubject fsaverage --trgsubject $subject --hemi lh --sval-annot $SUBJECTS_DIR/fsaverage/label/lh.PALS_B12_Brodmann.annot --tval $SUBJECTS_DIR/$subject/label/lh.PALS_B12_Brodmann.annot
	
	mri_annotation2label --subject $subject --hemi lh --labelbase lh.PALS_B12_Brodmann.annot --annotation PALS_B12_Brodmann --surf orig
done

