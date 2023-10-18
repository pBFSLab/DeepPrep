# syntax = docker/dockerfile:1.5

## Start from this Docker image
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

## Install * in Docker image
RUN apt-get update && apt-get upgrade -y
RUN apt-get --no-install-recommends -y install vim python3 python3-pip python3-dev python3-venv python2 git
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip

## Install time zone
ENV TZ=Etc/UTC
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata

## Install FreeSurfer
RUN apt-get -y install bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libjpeg62-dev libglu1-mesa
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:linuxuprising/libpng12
RUN apt-get update && apt-get -y install libpng12-0
RUN add-apt-repository -r ppa:linuxuprising/libpng12
COPY freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz /opt
RUN tar -C /usr/local -xzvf /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && rm /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz
RUN sed -i '1c\#!/usr/bin/env python2' /usr/local/freesurfer/bin/slicedelay
ENV FREESURFER_HOME=/usr/local/freesurfer

## Install workbench
COPY workbench-linux64-v1.5.0.zip /opt
RUN unzip /opt/workbench-linux64-v1.5.0.zip -d /usr/local && rm /opt/workbench-linux64-v1.5.0.zip
ENV PATH="$PATH:/usr/local/workbench/bin_linux64"

## Install Python lib
RUN --mount=type=cache,target=/root/.cache/pip pip3 install nighres
RUN --mount=type=cache,target=/root/.cache/pip pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN --mount=type=cache,target=/root/.cache/pip pip3 install tensorflow==2.5.0
RUN --mount=type=cache,target=/root/.cache/pip pip3 install git+https://github.com/Deep-MI/LaPy.git@v1.0.1
RUN --mount=type=cache,target=/root/.cache/pip pip3 install git+https://github.com/voxelmorph/voxelmorph.git@9d5e429084c33e07bd9ab41d6cb9beab8d4be62b
COPY requirements.txt /opt
RUN pip3 install -r /opt/requirements.txt
RUN pip3 install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html

COPY pyg /opt

RUN cd /opt && pip3 install pyg_lib-0.1.0+pt112cu113-cp38-cp38-linux_x86_64.whl torch_cluster-1.6.0+pt112cu113-cp38-cp38-linux_x86_64.whl torch_scatter-2.1.0+pt112cu113-cp38-cp38-linux_x86_64.whl torch_sparse-0.6.15+pt112cu113-cp38-cp38-linux_x86_64.whl torch_spline_conv-1.2.1+pt112cu113-cp38-cp38-linux_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/pip pip3 install torch-geometric==2.2.0

RUN --mount=type=cache,target=/root/.cache/pip pip3 install numpy==1.19.5 pandas==1.3.4

RUN apt-get -y install libmagickwand-dev

## Install openjdk
RUN apt-get -y install openjdk-11-jdk
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/lib/server"

## Install nextflow
RUN apt-get -y install wget && wget -qO- https://get.nextflow.io | bash && chmod +x nextflow && mv nextflow /usr/bin

## clean pip and apt tmp file
RUN pip3 cache purge
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

## echo to .bashrc
RUN echo "source /usr/local/freesurfer/SetUpFreeSurfer.sh\n" >> ~/.bashrc
