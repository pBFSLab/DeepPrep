#!/bin/bash
args=("$@")
echo "INFO: args: ${args[*]}"

config_file=""
executor="local"
cpus=""
memory=""
freesurfer_home="/usr/local/freesurfer"
fs_license_file=""
deepprep_home="/opt/DeepPrep"
ignore_error=""
debug=""

help="DeepPrep args:
deepprep-docker [bids_dir] [output_dir] [{participant}] [--bold_task_type TASK_LABEL]
                [--fs_license_file PATH] [--participant-label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
                [--subjects_dir PATH] [--executor {local cluster}]
                [--anat_only] [--bold_only] [--bold_sdc] [--bold_confounds]
                [--bold_surface_spaces '[fsnative fsaverage fsaverage6 ...]']
                [--bold_template_space {MNI152NLin6Asym MNI152NLin2009cAsym}] [--bold_template_res {02 03...}]
                [--device {auto {0 1 2...} cpu}] [--gpu_compute_capability {8.6}]
                [--cpus 10] [--memory 5]
                [--freesurfer_home PATH] [--deepprep_home PATH] [--templateflow_home PATH]
                [--ignore_error]
                [-resume]
"

if [ $# -lt 3 ]; then
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
    --freesurfer_home)
      freesurfer_home="$2"
      echo "Input --freesurfer_home : ${freesurfer_home}"
      shift
      ;;
    --freesurfer_license)
      freesurfer_license="$2"
      echo "Input --freesurfer_license : ${freesurfer_license}"
      shift
      ;;
    -c|--config_file)
      config_file="$2"
      echo "Input --config_file : ${config_file}"
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
    --debug)
      debug="True"
      echo "Input --debug : ${debug}"
      ;;
    -h|--help)
      echo "${help}"
      exit 0
      ;;
  esac
  shift
done

if [ "${executor}" = "local" ]; then
  if pgrep redis-server > /dev/null
  then
    echo "INFO: Redis is already running."
  else
    echo "INFO: Starting Redis..."
    nohup redis-server > /dev/null 2>&1 &
    echo "INFO: Redis is running."
  fi
fi

source "${freesurfer_home}/SetUpFreeSurfer.sh"

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

if [ -n "${deepprep_home}" ]; then
  echo "INFO: deepprep_home : ${deepprep_home}"
  sed -i "s@/opt/DeepPrep@${deepprep_home}@g" "${run_config}"
fi

if [ -n "${freesurfer_home}" ]; then
  echo "INFO: freesurfer_home : ${freesurfer_home}"
  sed -i "s@/usr/local/freesurfer@${freesurfer_home}@g" "${run_config}"
fi

if [ -n "${fs_license_file}" ]; then
  echo "INFO: fs_license_file : ${fs_license_file}"
  export FS_LICENSE=${fs_license_file}
  sed -i "s@\${freesurfer_home}/license.txt@${fs_license_file}@g" "${run_config}"
fi

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
