# build image
docker build --progress=plain -t deepprep:dev .

# run docker image to test scheduler.py and nextflow
docker run -it --gpus all -v /mnt/ngshare:/mnt/ngshare -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt deepprep:dev /bin/bash

cd /root/workspace/DeepPrep/deepprep
python3   scheduler.py  --bids-dir     /mnt/ngshare/temp/MSC_One     --recon-output-dir     /mnt/ngshare/temp/MSC_One_Recon     --bold-output-dir     /mnt/ngshare/temp/MSC_One_BoldPreprocess     --cache-dir     /mnt/ngshare/temp/MSC_One_Workflow     --bold-task-type     motor

cd /root/workspace/DeepPrep/deepprep/nextflow
nextflow 

# 