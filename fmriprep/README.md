# aws s3下载依赖和dokcer编译脚本
aws s3 sync s3://backupng/anning/fmriprep_dependencies /$HOME/WorkSpace/fmriprep/fmriprep_dependencies

# docker
## 安装
docker pull nipreps/fmriprep:21.0.2
cd fmriprep/fmriprep_dependencies/docker
docker build -t fmriprep .

## 运行
docker run --rm -it -v /mnt/ngshare/PersonData/anning/OrigData/ds004128/:/data:ro     -v /mnt/ngshare/PersonData/anning/OrigData/ds004128_out/:/out fmriprep     /data /out participant --skip_bids_validation

# 脚本
默认的安装路径 /opt
## 安装
修改 fmriprep_install.sh 中的依赖路径为下载依赖的路径

chmod +x fmriprep_install.sh

sudo ./fmriprep_install.sh

## 加载环境变量
cd /$HOME/WorkSpace/fmriprep/fmriprep_dependencies
source .fmriprep

## 安装python环境
pip install sentry-sdk fmriprep
npm install -g bids-validator

## 运行
fmriprep /mnt/ngshare/PersonData/anning/OrigData/ds004128 /mnt/ngshare/PersonData/anning/OrigData/ds004128_out participant --skip_bids_validation