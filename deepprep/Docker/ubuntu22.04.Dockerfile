# syntax = docker/dockerfile:1.5

## Start from this Docker image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

## Install * in Docker image
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    # Python
RUN apt-get update && apt-get --no-install-recommends -y install wget vim git python3 python3-pip python3-dev python2 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Install time zone
ENV TZ=Etc/UTC
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    # FreeSurfer
RUN apt-get update && apt-get --no-install-recommends -y install software-properties-common bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libglu1-mesa \
    # wand for svg
    libmagickwand-dev \
    # tksurfer
    libxmu6 \
    # open jdk
    openjdk-11-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN add-apt-repository ppa:linuxuprising/libpng12 && apt-get update && apt-get -y install libpng12-0 && add-apt-repository -r ppa:linuxuprising/libpng12 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && python -m pip install --upgrade pip && pip3 cache purge

## Install FreeSurfer
#COPY freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz /opt
#RUN wget -P /opt/ http://30.30.30.204:8000/Docker/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && tar -C /usr/local -xzvf /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && rm /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz
RUN wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/cd34b3a9b3d54983b33b/?dl=1 && tar -C /usr/local -xzvf /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && rm /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz
RUN sed -i '1c\#!/usr/bin/env python2' /usr/local/freesurfer/bin/slicedelay
ENV FREESURFER_HOME=/usr/local/freesurfer

## Install workbench
#COPY workbench-linux64-v1.5.0.zip /opt
#RUN wget -P /opt/ http://30.30.30.204:8000/Docker/workbench-linux64-v1.5.0.zip && unzip /opt/workbench-linux64-v1.5.0.zip -d /usr/local && rm /opt/workbench-linux64-v1.5.0.zip
RUN wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/edc6781c767a46d1af00/?dl=1 && unzip /opt/workbench-linux64-v1.5.0.zip -d /usr/local && rm /opt/workbench-linux64-v1.5.0.zip
ENV PATH="$PATH:/usr/local/workbench/bin_linux64"

## Install openjdk
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/lib/server"
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

### Install Python lib
#RUN git clone https://github.com/nighres/nighres.git --branch release-1.5.0 --single-branch && \
#    apt-get -y --no-install-recommends install python3-jcc && \
#    cd nighres && ln -sf /usr/lib/jvm/java-11-openjdk-amd64 /usr/lib/jvm/default-java && \
#    bash build.sh && pip3 install .
#RUN wget -P /opt http://30.30.30.204:8000/Docker/nighres-1.5.0-cp310-cp310-linux_x86_64.whl && pip3 install /opt/nighres-1.5.0-cp310-cp310-linux_x86_64.whl && pip3 cache purge
RUN wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/3e7ce1a2bd4e4af18078/?dl=1 && pip3 install /opt/nighres-1.5.0-cp310-cp310-linux_x86_64.whl && pip3 cache purge
RUN pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && pip3 cache purge
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html && pip3 cache purge
RUN pip3 install git+https://github.com/Deep-MI/LaPy.git@v1.0.1 && pip3 cache purge
RUN pip3 install git+https://github.com/voxelmorph/voxelmorph.git@9d5e429084c33e07bd9ab41d6cb9beab8d4be62b && pip3 cache purge
# becase sagereg torch-geometric <= 2.2.0
RUN pip3 install tensorflow==2.11.1 torch-geometric==2.2.0 nnunet==1.7.1 \
    sh==2.0.6 dynaconf==3.2.3 \
    fvcore \
    wand==0.6.11 \
    bids==0.0 nipype==1.8.6 niworkflows==1.9.0 SimpleITK==2.3.0 nitime==0.10.1 \
    open3d==0.17.0 \
    && pip3 cache purge
RUN pip3 install --no-index --no-cache-dir pytorch3d==0.7.4 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html  && pip3 cache purge

### Install nextflow
RUN wget -qO- https://get.nextflow.io | bash && chmod +x nextflow && mv nextflow /usr/bin && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

### Install Redis
RUN apt-get update && apt-get --no-install-recommends -y install redis-server && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN sed -i '147c\supervised systemd' /etc/redis/redis.conf
RUN pip3 install python-redis-lock  && pip3 cache purge

### Install SynthMorph
RUN git clone https://github.com/adalca/neurite && cd neurite && pip3 install .  && pip3 cache purge && rm -rf neurite

### echo to .bashrc
RUN echo "source /usr/local/freesurfer/SetUpFreeSurfer.sh" >> ~/.bashrc

COPY deepprep /deepprep
RUN chmod +x /deepprep/Docker/deepprep.sh

### CMD
ENTRYPOINT ["/bin/bash", "/deepprep/Docker/deepprep.sh", "run", "/deepprep/nextflow/deepprep.nf"]
