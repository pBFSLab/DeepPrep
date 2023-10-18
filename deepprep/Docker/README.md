# build image
docker buildx build --progress=plain -t deepprep:dev-22.04 -f Dockerfile_22.04 .

# run docker image to test scheduler.py and nextflow
docker run -it --gpus all -v /mnt/ngshare:/mnt/ngshare -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt deepprep:dev /bin/bash
docker run -it --gpus all -v /mnt/ngshare:/mnt/ngshare -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt deepprep:runtime-ubuntu22.04 /bin/bash
docker run -it --gpus all -v /mnt/ngshare:/mnt/ngshare -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt deepprep:runtime-ubuntu20.04 /bin/bash

# test scheduler
cd /root/workspace/DeepPrep/deepprep
python3   scheduler.py  --bids-dir     /mnt/ngshare/temp/MSC_One     --recon-output-dir     /mnt/ngshare/temp/MSC_One_Recon     --bold-output-dir     /mnt/ngshare/temp/MSC_One_BoldPreprocess     --cache-dir     /mnt/ngshare/temp/MSC_One_Workflow     --bold-task-type     motor

# test nextflow
cd /root/workspace/DeepPrep/deepprep/nextflow 
nextflow run deepprep.nf

# Singularity
singularity build --notest DeepPrep_runtime_ubuntu20.04.sif singularity_ubuntu20.04.def
singularity build --notest DeepPrep_runtime_ubuntu22.04.sif singularity_ubuntu22.04.def