# build image
docker buildx build --progress=plain -t deepprep:runtime-ubuntu22.04 -f ubuntu22.04.Dockerfile .
docker buildx build --progress=plain -t deepprep:runtime-ubuntu20.04 -f ubuntu20.04.Dockerfile .

# run docker contaniter and nextflow
docker run -it --gpus all -v /mnt/ngshare:/mnt/ngshare -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt deepprep:runtime-ubuntu22.04 /bin/bash
docker run -it --gpus all -v /mnt/ngshare:/mnt/ngshare -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt deepprep:runtime-ubuntu20.04 /bin/bash

service start redis-server
cd /root/workspace/DeepPrep/deepprep/nextflow 
nextflow run -c nextflow.docker.config deepprep.nf

# run docker image and nextflow 
docker run -it --gpus all \
    -v /mnt/ngshare:/mnt/ngshare \
    -v /mnt/ngshare/temp/MSC_Docker_nf_workDir:/mnt/workDir \
    -v /home/anning/workspace/DeepPrep:/root/workspace/DeepPrep \
    -v /home/anning/workspace/DeepPrep/deepprep/model:/usr/share/deepprep/model \
    -v /usr/local/freesurfer/license.txt:/usr/local/freesurfer/license.txt \
    deepprep:runtime-ubuntu20.04 /opt/nextflow.sh run -c /root/workspace/DeepPrep/deepprep/nextflow/nextflow.docker.config \
    /root/workspace/DeepPrep/deepprep/nextflow/deepprep.nf

# re-run
docker exec -it \
    <container_id#change with the actual situation> \
    bash /opt/nextflow.sh run -c /root/workspace/DeepPrep/deepprep/nextflow/nextflow.docker.config \
    /root/workspace/DeepPrep/deepprep/nextflow/deepprep.nf

# Singularity build
singularity build --notest DeepPrep_runtime_ubuntu20.04.sif singularity_ubuntu20.04.def
singularity build --notest DeepPrep_runtime_ubuntu22.04.sif singularity_ubuntu22.04.def

# Singularity test
singularity exec -e --nv -B /mnt/ngshare:/mnt/ngshare \
    -B /mnt/ngshare/temp/Nextflow_workDir:/mnt/workDir \
    -B /home/anning/workspace/DeepPrep/deepprep/nextflow:/mnt/nextflow \
    singularity_ubuntu20.04.def /opt/nextflow.sh run /mnt/nextflow/deepprep.nf -c /mnt/nextflow/nextflow.singularity.config

# remove docker tmp cache
docker system df
docker builder prune
