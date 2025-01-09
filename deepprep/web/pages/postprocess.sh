#!/bin/bash
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

args=("$@")
echo "INFO: args: ${args[*]}"

cpus=""
memory=""
freesurfer_home="/opt/freesurfer"
fs_license_file=""
subjects_dir=""
deepprep_home="/opt/DeepPrep"
ignore_error=""
debug=""

help="DeepPrep args:
deepprep-docker [bids_dir] [output_dir] [{participant}]
                [--fs_license_file PATH] [--subjects_dir PATH] [--confounds_index_file Path]
                [--task_id '[task1 task2 task3 ...]'] [--participant_label '[001 002 003 ...]']
                [--skip_frame 0] [--repetition_time 0]
                [--spaces '[None MNI152NLin6Asym MNI152NLin2009cAsym fsnative fsaverage fsaverage6 ...]']
                [--cpus 10] [--memory 20]
                [--skip_bids_validation]
                [--ignore_error] [--resume]
"

bids_dir=$1
output_dir=$2
shift
shift
shift

args=()
args+=("--bids_dir")
args+=("${bids_dir}")
args+=("--output_dir")
args+=("${output_dir}")
args=("${args[@]}" "$@")

# Parse command line options
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --deepprep_home)
      deepprep_home="$2"
      echo "Input --deepprep_home : ${deepprep_home}"
      shift
      ;;
    --fs_license_file)
      fs_license_file="$2"
      echo "Input --fs_license_file : ${fs_license_file}"
      shift
      ;;
    --subjects_dir)
      subjects_dir="$2"
      echo "Input --subjects_dir : ${subjects_dir}"
      shift
      ;;
    --cpus)
      cpus="$2"
      echo "Input --cpus : ${cpus}"
      shift
      ;;
    --memory)
      memory="$2"
      echo "Input --memory : ${memory} GB"
      shift
      ;;
    --ignore_error)
      ignore_error="True"
      echo "Input --ignore_error : ${ignore_error}"
      ;;
    --resume)
      resume="True"
      echo "Input --resume : ${resume}"
      args+=("-resume")
      ;;
    --debug)
      debug="True"
      echo "Input --debug : ${debug}"
      ;;
    -h|-help|--help)
      echo "${help}"
      exit 0
      ;;
  esac
  shift
done

if [ ! -d "${deepprep_home}" ]; then
  echo "ERROR: deepprep_home is not exists : ${deepprep_home}"
  exit 1
fi

if [ -n "${freesurfer_home}" ]; then
  export FREESURFER_HOME="${freesurfer_home}"
  echo "FREESURFER_HOME is set to: $FREESURFER_HOME"
fi
if [ -z "${freesurfer_home}" ]; then
  echo "ERROR: freesurfer_home is empty : ${freesurfer_home}.Please provide --freesurfer_home parameter or set FREESURFER_HOME environment variable."
  exit 1
fi
if [ ! -d "${freesurfer_home}" ]; then
  echo "ERROR: freesurfer_home is not exists : ${freesurfer_home}"
  exit 1
fi
if [ -z "FSF_OUTPUT_FORMAT" ]; then
    echo "source ${freesurfer_home}/SetUpFreeSurfer.sh"
    source "${freesurfer_home}/SetUpFreeSurfer.sh"
fi
if [ -z "${fs_license_file}" ]; then
  echo "WARNNING: You should replace license.txt path with your own FreeSurfer license! You can get your license file for free from https://surfer.nmr.mgh.harvard.edu/registration.html"
  echo "WARNNING: Then add  --fs_license_file <your license file path> ."
  fs_license_file="${deepprep_home}/deepprep/FreeSurfer/license.txt"
fi
if [ ! -f "${fs_license_file}" ]; then
  echo "ERROR: fs_license_file is not exists : ${fs_license_file}"
  exit 1
fi
export FS_LICENSE=${fs_license_file}

nextflow_work_dir="${output_dir}/WorkDir/nextflow"  # output_dir/WorkDir/nextflow
qc_dir="${output_dir}/QC"  # output_dir/QC

if [ ! -d "${nextflow_work_dir}" ]; then
  mkdir -p "${nextflow_work_dir}"
  echo "INFO: create  nextflow WorkDir: ${nextflow_work_dir}"
else
  echo "INFO: existed nextflow WorkDir: ${nextflow_work_dir}"
fi
if [ ! -d "${output_dir}/WorkDir/home" ]; then
  mkdir -p "${output_dir}/WorkDir/home"
  echo "INFO: create  nextflow WorkDir: ${output_dir}/WorkDir/home"
fi
if [ ! -d "${output_dir}/WorkDir/tmp" ]; then
  mkdir -p "${output_dir}/WorkDir/tmp"
  echo "INFO: create  nextflow WorkDir: ${output_dir}/WorkDir/tmp"
fi

nextflow_file="${deepprep_home}/deepprep/rest/denoise/postprocess.nf"
common_config="${deepprep_home}/deepprep/rest/denoise/postprocess.config"
if [ -n "${debug}" ]; then
  echo "DEBUG: nextflow_file : ${nextflow_file}"
  echo "DEBUG: common_config : ${common_config}"
fi
if [ ! -f "${common_config}" ]; then
  echo "ERROR: config_file is not exists : ${common_config}"
  exit 1
fi

run_config="${nextflow_work_dir}/run.config"
echo "INFO: run_config : ${run_config}"
cat "${common_config}" > "${run_config}"

if [ -n "${cpus}" ]; then
  sed -i "s@//cpus=@    cpus=${cpus}@g" "${run_config}"
fi

if [ -n "${memory}" ]; then
  sed -i "s@//memory=@    memory='${memory} GB'@g" "${run_config}"
fi

if [ -n "${ignore_error}" ]; then
  sed -i "s@//errorStrategy@    errorStrategy@g" "${run_config}"
fi

cd "${nextflow_work_dir}" && \
nextflow run "${nextflow_file}" \
-c "${run_config}" \
-w "${nextflow_work_dir}" \
-with-report "${qc_dir}/report.html" \
-with-timeline "${qc_dir}/timeline.html" \
"${args[@]}"
