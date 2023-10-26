# build image
docker buildx build --progress=plain -t deepprep:runtime-ubuntu22.04 -f ubuntu22.04.Dockerfile .
docker buildx build --progress=plain -t deepprep:runtime-ubuntu20.04 -f ubuntu20.04.Dockerfile .

# run docker contaniter and nextflow
docker run -it --gpus all -v /mnt/ngshare:/mnt/ngshare -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt deepprep:runtime-ubuntu22.04 /bin/bash
docker run -it --gpus all -v /mnt/ngshare:/mnt/ngshare -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt deepprep:runtime-ubuntu20.04 /bin/bash

service start redis-server
cd /root/workspace/DeepPrep/deepprep/nextflow 
nextflow run -c nextflow.docker.config deepprep.nf

# run docker image and nextflow run
docker run -it --gpus all \
    -v /mnt/ngshare:/mnt/ngshare \
    -v /mnt/ngshare/temp/MSC_Docker_nf_workDir:/mnt/workDir \
    -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep \
    -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model \
    -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt \
    deepprep:runtime-ubuntu20.04 /opt/nextflow.sh run /root/workspace/DeepPrep/deepprep/nextflow/deepprep.nf \
    -c /root/workspace/DeepPrep/deepprep/nextflow/nextflow.docker.config
    

# exec docker container and nextflow run -resume
docker exec -it \
    <container_id#Warning:change this in the actual situation> \
    bash /opt/nextflow.sh run /root/workspace/DeepPrep/deepprep/nextflow/deepprep.nf \
    -c /root/workspace/DeepPrep/deepprep/nextflow/nextflow.docker.config
    

# Singularity build
sudo singularity build --notest DeepPrep_runtime_ubuntu20.04.sif singularity_ubuntu20.04.def
sudo singularity build --notest DeepPrep_runtime_ubuntu22.04.sif singularity_ubuntu22.04.def

# Singularity test
singularity exec -e --nv -B /mnt/ngshare:/mnt/ngshare \
    -B /mnt/ngshare/temp/MSC_Singularity_nf_workDir:/mnt/workDir \
    -B /home/anning/workspace/DeepPrep/deepprep/nextflow:/mnt/nextflow \
    /home/anning/workspace/DeepPrep/deepprep/Docker/DeepPrep_runtime_ubuntu20.04.sif \
    /opt/nextflow.sh run -resume /mnt/nextflow/deepprep.nf -c /mnt/nextflow/nextflow.singularity.local.config

# remove docker tmp cache
docker system df
docker builder prune
