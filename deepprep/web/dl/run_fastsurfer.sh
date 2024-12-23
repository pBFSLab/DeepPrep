#!/bin/bash

# Inputs
subjects_dir=$1
participant_label=$2
device=$3

args=()
args+=("--subjects_dir" "${subjects_dir}")
args+=("--participant_label" "${participant_label}")
args+=("--device" "${device}")

help="FastSurfer args: [subjects_dir] [participant_label] [device]"

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
inputfile="${subjects_dir}"/"${participant_label}"/mri/orig.mgz
[ -e "${inputfile}" ] && echo "orig file: ${inputfile} exists!" || { echo "orig file: ${inputfile} does not exist, please check!" ; exit 1; }

# Run
python3 "/opt/DeepPrep/deepprep/FastSurfer/FastSurferCNN/eval.py" \
--in_name "${subjects_dir}"/"${participant_label}"/mri/orig.mgz \
--out_name "${subjects_dir}"/"${participant_label}"/mri/aparc.DKTatlas+aseg.deep.mgz \
--conformed_name "${subjects_dir}"/"${participant_label}"/mri/conformed.mgz \
--order 1 \
--network_sagittal_path /opt/DeepPrep/deepprep/FastSurfer/checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
--network_coronal_path /opt/DeepPrep/deepprep/FastSurfer/checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
--network_axial_path /opt/DeepPrep/deepprep/FastSurfer/checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
--batch_size 1 --simple_run --run_viewagg_on check

python3 "/opt/DeepPrep/deepprep/FastSurfer/recon_surf/reduce_to_aseg.py" \
--input "${subjects_dir}"/"${participant_label}"/mri/aparc.DKTatlas+aseg.deep.mgz \
--output "${subjects_dir}"/"${participant_label}"/mri/aseg.auto_noCCseg.mgz \
--outmask "${subjects_dir}"/"${participant_label}"/mri/mask.mgz \
--fixwm