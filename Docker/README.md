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
