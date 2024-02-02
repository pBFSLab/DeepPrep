# syntax = docker/dockerfile:1.5

## Start from this Docker image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

## Install * in Docker image
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Python
RUN apt-get update && apt-get --no-install-recommends -y install wget curl vim git python3 python3-pip python3-dev python2 rsync && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install time zone
ENV TZ=Etc/UTC
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && \
    # FreeSurfer
    apt-get --no-install-recommends -y install software-properties-common bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libglu1-mesa \
    # wand for svg
    libmagickwand-dev \
    # tksurfer
    libxmu6 \
    # redis
    redis-server \
    # open jdk
    openjdk-11-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN add-apt-repository ppa:linuxuprising/libpng12 && apt-get update && apt-get -y install libpng12-0 && add-apt-repository -r ppa:linuxuprising/libpng12 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && python -m pip install --upgrade pip && pip3 cache purge && rm -rf /tmp/* /var/tmp/*
RUN apt purge python3-blinker -y && apt autoremove -y

### Install Python lib
#RUN git clone https://github.com/nighres/nighres.git --branch release-1.5.0 --single-branch && \
#    apt-get -y --no-install-recommends install python3-jcc && \
#    cd nighres && ln -sf /usr/lib/jvm/java-11-openjdk-amd64 /usr/lib/jvm/default-java && \
#    bash build.sh && pip3 install .
RUN #pip3 config set global.extra-index-url "https://download.pytorch.org/whl/cu118"
RUN --mount=type=cache,target=/root/.cache \
    pip3 install \
    # https://data.pyg.org/whl/torch-2.0.1+cu118.html
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    # https://download.pytorch.org/whl/cu118
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
    # https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html
    pytorch3d==0.7.4 \
    # nighres-1.5.0-cp310-cp310-linux_x86_64.whl
    nighres==1.5.0  \
    open3d==0.17.0 \
    sh==2.0.6 dynaconf==3.2.3 wand==0.6.11 \
    tensorflow==2.11.1 \
    # SUGAR torch-geometric <= 2.2.0
    torch-geometric==2.2.0 \
    nnunet==1.7.1 \
    bids==0.0 nipype==1.8.6 niworkflows==1.10.0 SimpleITK==2.3.0 nitime==0.10.1 \
    git+https://github.com/NingAnMe/fmriprep.git@deepprep smriprep==0.13.2 sdcflows scipy==1.11.4 \
    git+https://github.com/Deep-MI/LaPy.git@v1.0.1 \
    git+https://github.com/voxelmorph/voxelmorph.git@9d5e429084c33e07bd9ab41d6cb9beab8d4be62b \
    git+https://github.com/NingAnMe/mriqc.git@deepprep \
    # CUDA
    tensorrt==8.6.1 onnxruntime==1.16.3 cuda-python==12.3.0 \
    # SynthMorph
    git+https://github.com/adalca/neurite \
    --trusted-host 30.30.30.204 -f http://30.30.30.204/cp310_whl \
    && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

### mriqc
RUN pip3 install mriqc-learn==0.0.2 --no-deps && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

### ONNX TensorRT
RUN cp /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so.8 /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so.7 && \
    cp /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer_plugin.so.8 /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer_plugin.so.7
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/tensorrt_libs"

## Install openjdk
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/lib/server"
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

### Install nextflow
RUN mkdir -p /opt/nextflow/bin && cd /opt/nextflow/bin && wget -qO- https://get.nextflow.io | bash && chmod 755 nextflow && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV PATH="/opt/nextflow/bin:${PATH}"

### bids-validator
RUN curl -sL https://deb.nodesource.com/setup_20.x | sudo -E bash - && \
    apt purge libnode-dev -y && apt autoremove -y && apt install -y nodejs  && npm install -g bids-validator && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && npm cache clean --force

### Install Redis
RUN sed -i '147c\supervised systemd' /etc/redis/redis.conf
RUN pip3 install python-redis-lock  && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

## Install FreeSurfer
RUN wget --content-disposition -P /opt/ http://30.30.30.204/DeepPrep_new/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && tar -C /opt -xzvf /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && rm /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz
#RUN wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/4863e1ddb58d416bb6d3/?dl=1 && tar -C /opt -xzvf /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && rm /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz
ENV FREESURFER_HOME=/opt/freesurfer
RUN echo "source ${FREESURFER_HOME}/SetUpFreeSurfer.sh" >> ~/.bashrc

## Install workbench
RUN wget -P /opt/ http://30.30.30.204/workbench-linux64-v1.5.0.zip && unzip /opt/workbench-linux64-v1.5.0.zip -d /opt && rm /opt/workbench-linux64-v1.5.0.zip
#RUN wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/edc6781c767a46d1af00/?dl=1 && unzip /opt/workbench-linux64-v1.5.0.zip -d /opt && rm /opt/workbench-linux64-v1.5.0.zip
ENV PATH="/opt/workbench/bin_linux64:${PATH}"

### ANTs  2.3.1
#RUN apt-get update && apt-get --no-install-recommends -y install build-essential && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
#RUN cd ~ && \
#    git clone https://github.com/cookpa/antsInstallExample.git && \
#    cd antsInstallExample && \
#    bash installANTs.sh && \
#    cp -r install /opt/ANTs && \
#    cd ~ && rm -r antsInstallExample
RUN wget --content-disposition -P /opt/ http://30.30.30.204/DeepPrep_new/ANTs_linux-ubuntu22-amd64_2.3.1.tar.gz && tar -C /opt -xzvf /opt/ANTs_linux-ubuntu22-amd64_2.3.1.tar.gz && rm /opt/ANTs_linux-ubuntu22-amd64_2.3.1.tar.gz
#RUN wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/ff5d799180524fe9a7b7/?dl=1 && tar -C /opt -xzvf /opt/ANTs_linux-ubuntu22-amd64_2.3.1.tar.gz && rm /opt/ANTs_linux-ubuntu22-amd64_2.3.1.tar.gz
ENV ANTSPATH="/opt/ANTs"
ENV PATH="${ANTSPATH}/bin:${PATH}"

### FSL 6.0.5.1
RUN apt-get update && apt-get --no-install-recommends -y install libopenblas-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget --content-disposition -P /opt/ http://30.30.30.204/DeepPrep_new/FSL_6.0.5.1.tar.gz && tar -C /opt -xzvf /opt/FSL_6.0.5.1.tar.gz && rm /opt/FSL_6.0.5.1.tar.gz
#RUN wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/22e8940b945b49758a76/?dl=1 && tar -C /opt -xzvf /opt/FSL_6.0.5.1.tar.gz && rm /opt/FSL_6.0.5.1.tar.gz
ENV FSLDIR="/opt/fsl"
ENV FSLOUTPUTTYPE="NIFTI"
ENV PATH="${FSLDIR}/bin:${PATH}"

#### AFNI 23.3.14
#RUN cd /opt && \
#    wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/0c88034741084793bd56/?dl=1 && \
#    wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/413153c2166440fc9394/?dl=1 && \
#    bash /opt/OS_notes.linux_ubuntu_22_64_a_admin.txt 2>&1 | tee o.ubuntu_22_a.txt && \
#    tcsh /opt/OS_notes.linux_ubuntu_22_64_b_user.tcsh 2>&1 | tee o.ubuntu_22_b.txt && \
#    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
#    apt remove python3-matplotlib -y && apt autoremove -y && \
#    cd ~ && mv /root/abin /opt && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget --content-disposition -P /opt/ http://30.30.30.204/DeepPrep_new/AFNI_linux-ubuntu-16-64_23.3.14.tar.gz && tar -C /opt -xzvf /opt/AFNI_linux-ubuntu-16-64_23.3.14.tar.gz && rm /opt/AFNI_linux-ubuntu-16-64_23.3.14.tar.gz
#RUN wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/9bc695e553554cd180b6/?dl=1 && tar -C /opt -xzvf /opt/AFNI_linux-ubuntu-16-64_23.3.14.tar.gz && rm /opt/AFNI_linux-ubuntu-16-64_23.3.14.tar.gz
ENV PATH="/opt/abin:${PATH}"

### mriqc
#COPY --from=freesurfer/synthstrip@sha256:f19578e5f033f2c707fa66efc8b3e11440569facb46e904b45fd52f1a12beb8b /freesurfer/models/synthstrip.1.pt /opt/freesurfer/models/synthstrip.1.pt
RUN mkdir ${FREESURFER_HOME}/models && wget --content-disposition -P ${FREESURFER_HOME}/models http://30.30.30.204/model/SynthStrip/synthstrip.1.pt  # synthstrip.1.pt
#RUN mkdir ${FREESURFER_HOME}/models && wget --content-disposition -P ${FREESURFER_HOME}/models http://30.30.30.141:8080/f/cb60226d1e8f497b9fb5/?dl=1  # synthstrip.1.pt

### default template
RUN python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin6Asym', desc=None, resolution=2, suffix='T1w', extension='nii.gz')"

COPY deepprep/model /opt/DeepPrep/deepprep/model
COPY deepprep/FastCSR /opt/DeepPrep/deepprep/FastCSR
COPY deepprep/SageReg /opt/DeepPrep/deepprep/SageReg
COPY deepprep/FastSurfer /opt/DeepPrep/deepprep/FastSurfer
COPY deepprep/SynthMorph /opt/DeepPrep/deepprep/SynthMorph
COPY deepprep/nextflow /opt/DeepPrep/deepprep/nextflow
COPY deepprep/deepprep.sh /opt/DeepPrep/deepprep/deepprep.sh
RUN chmod 755 /opt/DeepPrep/deepprep/deepprep.sh && chmod 755 /opt/DeepPrep/deepprep/nextflow/bin/*.py
ENV PATH="/opt/DeepPrep/deepprep/nextflow/bin:${PATH}"

### CMD
ENTRYPOINT ["/opt/DeepPrep/deepprep/deepprep.sh"]
