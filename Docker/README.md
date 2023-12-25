# Docker

export DEEPPREP_IMAGE_OUTPUT_PATH=/mnt/ngshare/SeaFile/Seafile/DeepPrep_Docker_Singularity
export DEEPPREP_VERSION='v0.0.10ubuntu22.04'

## build Docker image
docker buildx build --progress=plain -t deepprep:${DEEPPREP_VERSION} -f Docker/ubuntu22.04.Dockerfile .
docker save deepprep:${DEEPPREP_VERSION} -o ${DEEPPREP_IMAGE_OUTPUT_PATH}/deepprep_${DEEPPREP_VERSION}.tar.gz

## build Singularity image
sed -i "2c\From: deepprep:${DEEPPREP_VERSION}" Docker/singularity_ubuntu22.04.def
sed -i "36c\    Version ${DEEPPREP_VERSION}" Docker/singularity_ubuntu22.04.def
sudo singularity build --notest ${DEEPPREP_IMAGE_OUTPUT_PATH}/deepprep_${DEEPPREP_VERSION}.sif Docker/singularity_ubuntu22.04.def

## remove docker tmp cache (opt)
docker system df
docker builder prune

## run Docker contaniter and nextflow
export DEEPPREP_WORKDIR=$HOME/DEEPPREP_WORKDIR  # (required)
export DEEPPREP_RUN_CONFIG=${DEEPPREP_WORKDIR}/nextflow.docker.local.config  # (required)
export FREESURFER_LICENSE=/usr/local/freesurfer/license.txt  # (required)
mkdir ${DEEPPREP_WORKDIR}

export DEEPPREP_VERSION=v0.0.8ubuntu22.04  # (required)
export BIDS_DATASET=UKB1  # (required)
export DEEPPREP_RESULT_DIR=${DEEPPREP_WORKDIR}/${BIDS_DATASET}_${DEEPPREP_VERSION}D
mkdir ${DEEPPREP_RESULT_DIR}
mkdir ${DEEPPREP_RESULT_DIR}/nextflow_workDir

docker run -it --rm --gpus all --entrypoint /bin/bash \
-v ${DEEPPREP_WORKDIR}/${BIDS_DATASET}:/BIDS_DATASET \
-v ${DEEPPREP_RESULT_DIR}:/DEEPPREP_RESULT_DIR \
-v ${DEEPPREP_RESULT_DIR}/nextflow_workDir:/nextflow/workDir \
-v ${FREESURFER_LICENSE}:/usr/local/freesurfer/license.txt \
-v /home/anning/workspace/DeepPrep/deepprep:/deepprep \
-v ${DEEPPREP_RUN_CONFIG}:/nextflow.docker.local.config \
deepprep:${DEEPPREP_VERSION}

/deepprep/Docker/deepprep.sh run /deepprep/nextflow/deepprep.nf \
-resume \
-c /nextflow.docker.local.config --bids_dir /BIDS_DATASET \
--subjects_dir /DEEPPREP_RESULT_DIR/Recon \
--bold_preprocess_path /DEEPPREP_RESULT_DIR/BOLD \
--qc_result_path /DEEPPREP_RESULT_DIR/QC \
-with-report /DEEPPREP_RESULT_DIR/QC/report.html \
-with-timeline /DEEPPREP_RESULT_DIR/QC/timeline.html \
--bold_task_type rest --anat_only True

## Singularity test
export DEEPPREP_WORKDIR=$HOME/anning/DEEPPREP_WORKDIR  # (required)
export DEEPPREP_RUN_CONFIG=/lustre/grp/lhslab/sunzy/anning/workspace/DeepPrep/deepprep/nextflow/nextflow.docker.local.config  # (required)
export FREESURFER_LICENSE=/lustre/grp/lhslab/sunzy/freesurfer/license.txt  # (required)
mkdir ${DEEPPREP_WORKDIR}

export DEEPPREP_VERSION=v0.0.10ubuntu22.04  # (required)
export BIDS_DATASET=UKB1  # (required)
export DEEPPREP_RESULT_DIR=${DEEPPREP_WORKDIR}/${BIDS_DATASET}_${DEEPPREP_VERSION}S
mkdir ${DEEPPREP_RESULT_DIR}
mkdir ${DEEPPREP_RESULT_DIR}/nextflow_workDir

singularity exec -e --nv \
-B ${DEEPPREP_WORKDIR}/${BIDS_DATASET}:/BIDS_DATASET \
-B ${DEEPPREP_RESULT_DIR}:/DEEPPREP_RESULT_DIR \
-B ${DEEPPREP_RESULT_DIR}/nextflow_workDir:/nextflow/workDir \
-B ${FREESURFER_LICENSE}:/usr/local/freesurfer/license.txt \
-B ${DEEPPREP_RUN_CONFIG}:/nextflow.docker.local.config \
-B /lustre/grp/lhslab/sunzy/anning/workspace/DeepPrep/deepprep:/deepprep \
${DEEPPREP_WORKDIR}/deepprep_${DEEPPREP_VERSION}.sif \
/deepprep/deepprep.sh run /deepprep/nextflow/deepprep.nf \
-resume \
-c /nextflow.docker.local.config --bids_dir /BIDS_DATASET \
--subjects_dir /DEEPPREP_RESULT_DIR/Recon \
--bold_preprocess_path /DEEPPREP_RESULT_DIR/BOLD \
--qc_result_path /DEEPPREP_RESULT_DIR/QC \
-with-report /DEEPPREP_RESULT_DIR/QC/report.html \
-with-timeline /DEEPPREP_RESULT_DIR/QC/timeline.html \
--bold_task_type rest

# Dev
service start redis-server
cd /root/workspace/DeepPrep/deepprep/nextflow 
nextflow run -c nextflow.docker.config deepprep.nf
