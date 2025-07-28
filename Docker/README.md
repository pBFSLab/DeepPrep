# Docker

## local http server
export HTTP_HOME=${HOME}/http_server
mkdir -p ${HTTP_HOME}/DeepPrep/deps
wget --content-disposition -P ${HTTP_HOME}/DeepPrep/deps/ https://download.anning.info/ninganme-public/DeepPrep/deps/cp310_whl.zip && unzip ${HTTP_HOME}/DeepPrep/deps/cp310_whl.zip -d ${HTTP_HOME}/DeepPrep/deps/
cd ${HTTP_HOME} && sudo python3 -m http.server 80

export HTTP_HOME=/mnt/ngshare/SeaFile/Seafile/DeepPrep
cd ${HTTP_HOME} && sudo python3 -m http.server 80

## build Docker image
export DEEPPREP_HOME=${HOME}/DeepPrep
mkdir -p ${DEEPPREP_HOME}/deepprep/model
wget --content-disposition -P ${DEEPPREP_HOME}/deepprep/ https://download.anning.info/ninganme-public/DeepPrep/deps/model.zip && unzip ${DEEPPREP_HOME}/deepprep/model.zip -d ${DEEPPREP_HOME}/deepprep/model
cd ${DEEPPREP_HOME}
export DEEPPREP_VERSION='25.1.1'
docker buildx build --progress=plain -t pbfslab/deepprep:${DEEPPREP_VERSION} -f Docker/ubuntu22.04.Dockerfile .

docker save pbfslab/deepprep:${DEEPPREP_VERSION} -o ${DEEPPREP_IMAGE_OUTPUT_PATH}/deepprep_${DEEPPREP_VERSION}.tar.gz

## build Singularity image
```
apt-cache madison docker-ce | awk '{ print $3 }'
VERSION_STRING=5:24.0.7-1~ubuntu.20.04~focal
sudo apt-get install docker-ce=$VERSION_STRING docker-ce-cli=$VERSION_STRING containerd.io docker-buildx-plugin docker-compose-plugin
```

export DEEPPREP_IMAGE_OUTPUT_PATH=/mnt/ngshare/SeaFile/Seafile/DeepPrep_Docker_Singularity/DeepPrep_${DEEPPREP_VERSION} && \
mkdir -p ${DEEPPREP_IMAGE_OUTPUT_PATH} && \
sudo singularity build --notest ${DEEPPREP_IMAGE_OUTPUT_PATH}/deepprep_${DEEPPREP_VERSION}.sif Docker/singularity_ubuntu22.04.def

export DEEPPREP_IMAGE_OUTPUT_PATH=/mnt/ngshare/SeaFile/Seafile/DeepPrep_Docker_Singularity/DeepPrep_${DEEPPREP_VERSION} && \
mkdir -p ${DEEPPREP_IMAGE_OUTPUT_PATH} && \
sudo singularity build ${DEEPPREP_IMAGE_OUTPUT_PATH}/deepprep_${DEEPPREP_VERSION}.sif docker-daemon://pbfslab/deepprep:${DEEPPREP_VERSION}

## remove docker tmp cache (opt)
docker system df
docker builder prune
git clone --recursive --single-branch -b ${DEEPPREP_VERSION} https://github.com/pBFSLab/DeepPrep.git /opt/DeepPrep
