#!/bin/bash
echo "$@"

if pgrep redis-server > /dev/null
then
  echo "INFO: Redis is already running."
else
  echo "INFO: Starting Redis..."
  nohup redis-server > /dev/null 2>&1 &
  echo "INFO: Redis is running."
fi

source /usr/local/freesurfer/SetUpFreeSurfer.sh

# 定义目录路径
if [ "$1" != "--bids_dir" ]; then
    echo "$1 != --bids_dir"
    exit 1
fi
if [ "$3" != "--output_dir" ]; then
    echo "$3 != --output_dir"
    exit 1
fi
bids_dir=$2
output_dir=$4
nextflow_work_dir="${output_dir}/WorkDir/nextflow"  # output_dir/WorkDir/nextflow
qc_dir="${output_dir}/QC"  # output_dir/WorkDir/nextflow

# 判断目录是否存在
if [ ! -d "$nextflow_work_dir" ]; then
  # 目录不存在，进行创建
  mkdir -p "$nextflow_work_dir"
  echo "INFO: create  nextflow WorkDir: $nextflow_work_dir"
else
  echo "INFO: existed nextflow WorkDir: $nextflow_work_dir"
fi

cd "$nextflow_work_dir" && \
nextflow run /opt/DeepPrep/deepprep/nextflow/deepprep.nf \
-c /opt/DeepPrep/deepprep/nextflow/nextflow.docker.local.config \
-w "${nextflow_work_dir}" \
-with-report "${qc_dir}/report.html" \
-with-timeline "${qc_dir}/timeline.html" \
"$@"
