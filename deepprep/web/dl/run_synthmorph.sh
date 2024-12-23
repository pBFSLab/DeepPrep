#!/bin/bash

# Inputs
bold_preprocess_dir=$1
participant_label=$2
template_space=$3
device=$4

args=()
args+=("--bold_preprocess_dir" "${bold_preprocess_dir}")
args+=("--participant_label" "${participant_label}")
args+=("--template_space" "${template_space}")
args+=("--device" "${device}")

help="Synthmorph args: [bold_preprocess_dir] [participant_label] [template_space] [device]"

if [ $# -le 1 ]; then
  echo "${help}"
  exit 1
fi

echo "INFO: args: ${args[*]}"

# Set environment variable
if [ "${device}" == "cpu" ]; then
  CUDA_VISIBLE_DEVICES=""
elif [ "${device}" == "cuda" ]; then
  CUDA_VISIBLE_DEVICES=0
  echo "CUDA_VISIBLE_DEVICES: 0"
else
  CUDA_VISIBLE_DEVICES="${device}"
  echo "CUDA_VISIBLE_DEVICES: ${device}"
fi
export CUDA_VISIBLE_DEVICES

# Checkout input file
t1_native2mm="${bold_preprocess_dir}"/"${participant_label}"/anat/"${participant_label}"_space-T1w_res-2mm_desc-skull_T1w.nii.gz
norm_native2mm="${bold_preprocess_dir}"/"${participant_label}"/anat/"${participant_label}"_space-T1w_res-2mm_desc-noskull_T1w.nii.gz

[ -e "${t1_native2mm}" ] && echo "t1_native2mm file: ${t1_native2mm} exists!" || { echo "t1_native2mm file: ${t1_native2mm} does not exist, please check!" ; exit 1; }
[ -e "${norm_native2mm}" ] && echo "norm_native2mm file: ${norm_native2mm} exists!" || { echo "norm_native2mm file: ${norm_native2mm} does not exist, please check!" ; exit 1; }

# Run
python3 "/opt/DeepPrep/deepprep/SynthMorph/bold_synthmorph_joint.py" \
--synth_model_path /opt/model/SynthMorph/models \
--synth_script /opt/DeepPrep/deepprep/SynthMorph/mri_synthmorph_joint.py \
--bold_preprocess_dir "${bold_preprocess_dir}" --subject_id "${participant_label}" \
--t1_native2mm "${t1_native2mm}" \
--norm_native2mm "${norm_native2mm}" \
--template_space "${template_space}"