service redis-server start
service redis-server status
source /usr/local/freesurfer/SetUpFreeSurfer.sh

# 定义目录路径
work_dir="/mnt/workDir"

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
