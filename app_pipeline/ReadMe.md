## App pipeline
本代码是简化App pipeline，主要为preprocess、res_proj阶段，用于速度测试及与DeepPrep做对比测试。

### 运行配置

文件下载
```shell
cd app_pipeline
aws s3 sync s3://backupng/weiwei/DeepPrep/app_pipeline/resources ./resources
aws s3 sync s3://backupng/weiwei/DeepPrep/app_pipeline/data ./data
# mri_vol2vol_ext
sudo aws s3 cp s3://backupng/weiwei/DeepPrep/app_pipeline/bin/mri_vol2vol_ext $FREESURFER_HOME/bin/mri_vol2vol_ext
sudo chmod 755 $FREESURFER_HOME/bin/mri_vol2vol_ext
```
reources：程序运行依赖文件

data：测试数据
### 运行环境
```shell
conda create --name App python=3.6
conda activate App
pip install -r requirements.txt
```
### 程序入口
```
sample_App_pipeline.py
```