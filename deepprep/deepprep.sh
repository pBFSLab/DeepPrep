#!/bin/bash
args=("$@")
echo "INFO: args: ${args[*]}"

config_file=""
executor="local"
cpus=""
memory=""
freesurfer_home="/opt/freesurfer"
fs_license_file=""
deepprep_home="/opt/DeepPrep"
container=""
ignore_error=""
debug=""

help="DeepPrep args:
deepprep-docker [bids_dir] [output_dir] [{participant}] [--bold_task_type TASK_LABEL]
                [--fs_license_file PATH] [--participant-label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
                [--subjects_dir PATH] [--executor {local cluster}]
                [--anat_only] [--bold_only] [--bold_sdc] [--bold_confounds]
                [--bold_surface_spaces '[fsnative fsaverage fsaverage6 ...]']
                [--bold_template_space {MNI152NLin6Asym MNI152NLin2009cAsym}] [--bold_template_res {02 03...}]
                [--device { {auto 0 1 2...} cpu}] [--gpu_compute_capability {8.6}]
                [--cpus 10] [--memory 5]
                [--ignore_error] [--resume]
"

if [ $# -lt 5 ]; then
  echo "ERROR: args less than required."
  echo "${help}"
  exit 1
fi

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
    --config_file)
      config_file="$2"
      echo "Input --config_file : ${config_file}"
      shift
      ;;
    --container)
      container="$2"
      echo "Input --container : ${container}"
      shift
      ;;
    --executor)
      executor="$2"
      echo "Input --executor : ${executor}"
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

# 定义目录路径
nextflow_work_dir="${output_dir}/WorkDir/nextflow"  # output_dir/WorkDir/nextflow
qc_dir="${output_dir}/QC"  # output_dir/QC

# 判断目录是否存在
if [ ! -d "${nextflow_work_dir}" ]; then
  # 目录不存在，进行创建
  mkdir -p "${nextflow_work_dir}"
  echo "INFO: create  nextflow WorkDir: ${nextflow_work_dir}"
else
  echo "INFO: existed nextflow WorkDir: ${nextflow_work_dir}"
fi

nextflow_file="${deepprep_home}/deepprep/nextflow/deepprep.nf"
common_config="${deepprep_home}/deepprep/nextflow/deepprep.common.config"
local_config="${deepprep_home}/deepprep/nextflow/deepprep.local.config"
if [ -z "${config_file}" ]; then
  config_file="${local_config}"
fi
if [ -n "${debug}" ]; then
  echo "DEBUG: nextflow_file : ${nextflow_file}"
  echo "DEBUG: common_config : ${common_config}"
  echo "DEBUG: local_config : ${local_config}"
  echo "DEBUG: config_file : ${config_file}"
fi

run_config="${nextflow_work_dir}/run.config"
echo "INFO: run_config : ${run_config}"

cat "${common_config}" > "${run_config}"
cat "${config_file}" >> "${run_config}"

if [ ! -d "${deepprep_home}" ]; then
  echo "ERROR: deepprep_home is not exists : ${deepprep_home}"
  exit 1
fi
sed -i "s@/opt/DeepPrep@${deepprep_home}@g" "${run_config}"

if [ -z "${freesurfer_home}" ]; then
  echo "ERROR: freesurfer_home is empty : ${freesurfer_home}"
  exit 1
fi
sed -i "s@\${freesurfer_home}@${freesurfer_home}@g" "${run_config}"

if [ -z "${fs_license_file}" ]; then
  echo "ERROR: No Input --fs_license_file : ${fs_license_file}"
  exit 1
fi
if [ ! -f "${fs_license_file}" ]; then
  echo "ERROR: fs_license_file is not exists : ${fs_license_file}"
  exit 1
fi
sed -i "s@\${fs_license_file}@${fs_license_file}@g" "${run_config}"

if [ -n "${cpus}" ]; then
  sed -i "s@//cpus=@    cpus=${cpus}@g" "${run_config}"
fi

if [ -n "${memory}" ]; then
  sed -i "s@//memory=@    memory='${memory} GB'@g" "${run_config}"
fi

if [ -n "${ignore_error}" ]; then
  sed -i "s@//errorStrategy@    errorStrategy@g" "${run_config}"
fi

if [ "${executor}" = "local" ]; then
  if pgrep redis-server > /dev/null
  then
    echo "INFO: Redis is already running."
  else
    echo "INFO: Starting Redis..."
    nohup redis-server > /dev/null 2>&1 &
    echo "INFO: Redis is running."
  fi
  if [ ! -d "${freesurfer_home}" ]; then
    echo "ERROR: freesurfer_home is not exists : ${freesurfer_home}"
    exit 1
  fi
  source "${freesurfer_home}/SetUpFreeSurfer.sh"
  export FS_LICENSE=${fs_license_file}
else
  if [ -z "${container}" ]; then
  echo "ERROR: No Input --container : ${container}"
  exit 1
  fi
  if [ ! -f "${container}" ]; then
    echo "ERROR: container file is not exists : ${container}"
    exit 1
  fi
  sed -i "s@\${nextflow_work_dir}@${nextflow_work_dir}@g" "${run_config}"
  sed -i "s@\${container}@${container}@g" "${run_config}"
fi

cd "${nextflow_work_dir}" && \
nextflow run "${nextflow_file}" \
-c "${run_config}" \
-w "${nextflow_work_dir}" \
-with-report "${qc_dir}/report.html" \
-with-timeline "${qc_dir}/timeline.html" \
"${args[@]}"
