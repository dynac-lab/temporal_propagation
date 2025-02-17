#!/bin/sh
# Set the path to your data directory
data_dir=XXX

# Set the output directory
output_dir=YYY

# Loop through subjects from 1 to 63
for subject_id in {1..63}; do
    subject=sub-$(printf "%02d" $subject_id)  # Format subject ID with leading zeros if needed
    input_file="${data_dir}${subject}/ses-mri3t/anat/${subject}_ses-mri3t_run-1_T1w.nii.gz"

    if [ -f "$input_file" ]; then
        echo "Processing $subject..."
        qsub -N "${subject}_recon-all" <<EOF
#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -l mem=8GB
#PBS -o ${output_dir}/${subject}_recon-all.out
#PBS -e ${output_dir}/${subject}_recon-all.err

recon-all -subject $subject -i "$input_file" -sd $output_dir -cw256 -all

EOF
    else
        echo "${input_file} not found for $subject. Skipping..."
    fi
done
