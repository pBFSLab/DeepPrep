
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

## Run singularity
singularity exec -e --nv \
-B ${BIDS_DATASET_DIR}:/BIDS_DATASET \
-B ${DEEPPREP_RESULT_DIR}:/DEEPPREP_RESULT_DIR \
-B ${DEEPPREP_RESULT_DIR}/nextflow_workDir:/nextflow/workDir \
-B ${FREESURFER_LICENSE}:/usr/local/freesurfer/license.txt \
-B ${DEEPPREP_RUN_CONFIG}:/nextflow.docker.local.config \
-B ${DEEPPREP_HOME}:/deepprep \
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

## run in HPC
nextflow run /lustre/grp/lhslab/sunzy/anning/workspace/DeepPrep/deepprep/nextflow/deepprep.nf \
-resume \
-c /lustre/grp/lhslab/sunzy/anning/workspace/DeepPrep/deepprep/nextflow/nextflow.singularity.hpc.config \
--bids_dir /lustre/grp/lhslab/sunzy/BIDS/MSC \
--subjects_dir /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/MSC_v0.0.10ubuntu22.04H/Recon \
--bold_preprocess_path /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/MSC_v0.0.10ubuntu22.04H/BOLD \
--qc_result_path /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/MSC_v0.0.10ubuntu22.04H/QC \
-with-report /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/MSC_v0.0.10ubuntu22.04H/QC/report.html \
-with-timeline /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/MSC_v0.0.10ubuntu22.04H/QC/timeline.html \
--bold_task_type rest

## Dev
### Clone DeepPrep Code
if first clone:
    git clone --recursive
else:
    git submodule update --init --recursive

### local 
export DEEPPREP_HOME=${HOME}/workspace/DeepPrep/deepprep
export FREESURFER_LICENSE=${FREESURFER_HOME}/license.txt  # (required)
export DEEPPREP_WORKDIR=${HOME}/DEEPPREP_WORKDIR  # (required)
export DEEPPREP_RUN_CONFIG=${DEEPPREP_HOME}/nextflow/nextflow.docker.local.config  # (required)

### deepprep version and dataset
export DEEPPREP_VERSION=v0.0.12ubuntu22.04  # (required)
export BIDS_DATASET_DIR=${DEEPPREP_WORKDIR}/ds004498  # (required)
export BIDS_DATASET_NAME=ds004498  # (required)
export DEEPPREP_RESULT_DIR=${DEEPPREP_WORKDIR}/${BIDS_DATASET_NAME}_${DEEPPREP_VERSION}D

mkdir ${DEEPPREP_WORKDIR}
mkdir ${DEEPPREP_RESULT_DIR}
mkdir ${DEEPPREP_RESULT_DIR}/nextflow_workDir

### docker run 
docker run -it --rm --gpus all --entrypoint /bin/bash \
-v ${BIDS_DATASET_DIR}:/BIDS_DATASET \
-v ${DEEPPREP_RESULT_DIR}:/DEEPPREP_RESULT_DIR \
-v ${DEEPPREP_RESULT_DIR}/nextflow_workDir:/nextflow/workDir \
-v ${FREESURFER_LICENSE}:/usr/local/freesurfer/license.txt \
-v ${DEEPPREP_HOME}:/deepprep \
-v ${DEEPPREP_RUN_CONFIG}:/nextflow.docker.local.config \
deepprep:${DEEPPREP_VERSION}

### run code
/deepprep/deepprep.sh run /deepprep/nextflow/deepprep.nf \
-resume \
-c /nextflow.docker.local.config --bids_dir /BIDS_DATASET \
--subjects_dir /DEEPPREP_RESULT_DIR/Recon \
--bold_preprocess_path /DEEPPREP_RESULT_DIR/BOLD \
--qc_result_path /DEEPPREP_RESULT_DIR/QC \
-with-report /DEEPPREP_RESULT_DIR/QC/report.html \
-with-timeline /DEEPPREP_RESULT_DIR/QC/timeline.html \
--bold_task_type rest --bold_only True

## local Dev
service start redis-server
cd /root/workspace/DeepPrep/deepprep/nextflow 
nextflow run -c nextflow.docker.config deepprep.nf
