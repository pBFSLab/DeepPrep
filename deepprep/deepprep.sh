#!/bin/bash
if pgrep redis-server > /dev/null
then
  echo "Redis is already running."
else
  echo "Starting Redis..."
  nohup redis-server > /dev/null 2>&1 &
  echo "Redis is running."
fi

source /usr/local/freesurfer/SetUpFreeSurfer.sh

# 定义目录路径
work_dir="/nextflow/workDir"

# 判断目录是否存在
if [ ! -d "$work_dir" ]; then
  # 目录不存在，进行创建
  mkdir "$work_dir"
  echo "create workDir successfully"
else
  echo "workDir already exist"
fi
echo "$work_dir"

cd "$work_dir"
echo "$@"
nextflow "$@"
