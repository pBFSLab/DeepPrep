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

help="SUGAR args: [subjects_dir] [participant_label] [hemi] [device]"

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
hemi_sulc="${subjects_dir}"/"${participant_label}"/surf/"${hemi}".sulc
hemi_curv="${subjects_dir}"/"${participant_label}"/surf/"${hemi}".curv
hemi_sphere="${subjects_dir}"/"${participant_label}"/surf/"${hemi}".sphere

[ -e "${hemi_sulc}" ] && echo "hemi sulc file: ${hemi_sulc} exists!" || { echo "hemi sulc file: ${hemi_sulc} does not exist, please check!" ; exit 1; }
[ -e "${hemi_curv}" ] && echo "hemi curv file: ${hemi_curv} exists!" || { echo "hemi curv file: ${hemi_curv} does not exist, please check!" ; exit 1; }
[ -e "${hemi_sphere}" ] && echo "hemi sphere file: ${hemi_sphere} exists!" || { echo "hemi sphere file: ${hemi_sphere} does not exist, please check!" ; exit 1; }

# Run
python3 "/opt/DeepPrep/deepprep/SUGAR/predict.py" \
--fsd /opt/freesurfer \
--model_path /opt/model/SUGAR/model_files \
--hemi "${hemi}" --sid "${participant_label}" \
--sd "${subjects_dir}" --device "${device}"