#!/bin/sh
# Set the path to your data directory
data_dir=XXX

# Set the output directory
output_dir=YYY

# Set subjects dir
SUBJECTS_DIR=ZZZ

# Loop through subjects from 1 to 63
for subject_id in {1..63}; do
    subject=sub-$(printf "%02d" $subject_id)  # Format subject ID with leading zeros if needed
	echo "Processing $subject..."
	qsub -N "${subject}_BN_atlas" <<EOF
#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -l mem=8GB
#PBS -o ${output_dir}/${subject}_BN_atlas.out
#PBS -e ${output_dir}/${subject}_BN_atlas.err

mris_ca_label -l $SUBJECTS_DIR/$subject/label/lh.cortex.label $subject lh $SUBJECTS_DIR/$subject/surf/lh.sphere.reg $SUBJECTS_DIR/lh.BN_Atlas.gcs $SUBJECTS_DIR/$subject/label/lh.BN_Atlas.annot

EOF
done
