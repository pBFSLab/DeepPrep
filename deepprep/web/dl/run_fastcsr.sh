#!/bin/bash

# Inputs
subjects_dir=$1
participant_label=$2
hemi=$3
device=$4

args=()
args+=("--subjects_dir" "${subjects_dir}")
args+=("--participant_label" "${participant_label}")
args+=("--hemi" "${hemi}")
args+=("--device" "${device}")

help="FastCSR args: [subjects_dir] [participant_label] [hemi] [device]"

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
orig_mgz="${subjects_dir}"/"${participant_label}"/mri/orig.mgz
filled_mgz="${subjects_dir}"/"${participant_label}"/mri/filled.mgz
brainmask_mgz="${subjects_dir}"/"${participant_label}"/mri/brainmask.mgz
aseg_presurf_mgz="${subjects_dir}"/"${participant_label}"/mri/aseg.presurf.mgz

[ -e "${orig_mgz}" ] && echo "orig file: ${orig_mgz} exists!" || { echo "orig file: ${orig_mgz} does not exist, please check!" ; exit 1; }
[ -e "${filled_mgz}" ] && echo "filled file: ${filled_mgz} exists!" || { echo "filled file: ${filled_mgz} does not exist, please check!" ; exit 1; }
[ -e "${brainmask_mgz}" ] && echo "brainmask file: ${brainmask_mgz} exists!" || { echo "brainmask file: ${brainmask_mgz} does not exist, please check!" ; exit 1; }
[ -e "${aseg_presurf_mgz}" ] && echo "aseg presurf file: ${aseg_presurf_mgz} exists!" || { echo "aseg presurf file: ${aseg_presurf_mgz} does not exist, please check!" ; exit 1; }

# Run
python3 "/opt/DeepPrep/deepprep/FastCSR/fastcsr_model_infer.py" \
--fastcsr_subjects_dir "${subjects_dir}" \
--model-path /opt/model/FastCSR \
--subj "${participant_label}" --hemi "${hemi}"

python3 "/opt/DeepPrep/deepprep/FastCSR/levelset2surf.py" \
--fastcsr_subjects_dir "${subjects_dir}" \
--subj "${participant_label}" --hemi "${hemi}" --suffix orig