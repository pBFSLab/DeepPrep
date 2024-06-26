# syntax = docker/dockerfile:1.5

## Start from this Docker image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

## Install * in Docker image
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Python
RUN apt-get update && apt-get --no-install-recommends -y install wget curl vim git python3 python3-pip python3-dev python2 rsync build-essential && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install time zone
ENV TZ=Etc/UTC
RUN apt-get update && apt-get --no-install-recommends -y install tzdata && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# FreeSurfer
RUN apt-get update && apt-get --no-install-recommends -y install software-properties-common bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libglu1-mesa && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get update && \
    apt-get --no-install-recommends -y install \
    # wand for svg
    libmagickwand-dev \
    # tksurfer
    libxmu6 \
    # redis
    redis-server \
    # open jdk
    openjdk-11-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN add-apt-repository ppa:linuxuprising/libpng12 && apt-get update && apt-get --no-install-recommends -y install libpng12-0 && add-apt-repository -r ppa:linuxuprising/libpng12 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
    bids==0.0 nipype==1.8.6 niworkflows==1.10.0 matplotlib==3.8.4 SimpleITK==2.3.0 nitime==0.10.1 \
    git+https://github.com/NingAnMe/fmriprep.git@deepprep smriprep==0.13.2 sdcflows scipy==1.11.4 \
    git+https://github.com/Deep-MI/LaPy.git@v1.0.1 \
    git+https://github.com/voxelmorph/voxelmorph.git@ca28315d0ba24cd8946ac4f6ed081e049e5264fe \
    git+https://github.com/NingAnMe/mriqc.git@deepprep \
    # SynthMorph
    git+https://github.com/adalca/neurite@682f828b7b5fa652d7205c894c7fe667f1a26251 surfa==0.6.0 \
    --trusted-host 30.30.30.204 -f http://30.30.30.204/cp310_whl \
    && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

### mriqc
RUN pip3 install mriqc-learn==0.0.2 --no-deps && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

## Install openjdk
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/lib/server"
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

### Install nextflow
RUN mkdir -p /opt/nextflow/bin && cd /opt/nextflow/bin && wget -qO- https://get.nextflow.io | bash && chmod 755 nextflow && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV PATH="/opt/nextflow/bin:${PATH}"

### bids-validator
RUN curl -sL https://deb.nodesource.com/setup_20.x | sudo -E bash - && \
    apt purge libnode-dev -y && apt autoremove -y && apt-get --no-install-recommends -y install nodejs  && npm install -g bids-validator && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && npm cache clean --force

### Install Redis
RUN sed -i '147c\supervised systemd' /etc/redis/redis.conf
RUN pip3 install python-redis-lock  && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

### aws cli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && sudo ./aws/install && rm -rf aws awscliv2.zip

## Install FreeSurfer
RUN wget --content-disposition -P /opt/ http://30.30.30.204/DeepPrep_new/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && tar -C /opt -xzvf /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && rm /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz
ENV FREESURFER_HOME=/opt/freesurfer
RUN echo "source ${FREESURFER_HOME}/SetUpFreeSurfer.sh" >> ~/.bashrc

### FSL 6.0.5.1
RUN apt-get update && apt-get --no-install-recommends -y install libopenblas-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget --content-disposition -P /opt/ http://30.30.30.204/DeepPrep_new/FSL_6.0.5.1.tar.gz && tar -C /opt -xzvf /opt/FSL_6.0.5.1.tar.gz && rm /opt/FSL_6.0.5.1.tar.gz
ENV FSLDIR="/opt/fsl"
ENV FSLOUTPUTTYPE="NIFTI_GZ"
ENV PATH="${FSLDIR}/bin:${PATH}"

## Install workbench
RUN wget -P /opt/ http://30.30.30.204/workbench-linux64-v1.5.0.zip && unzip /opt/workbench-linux64-v1.5.0.zip -d /opt && rm /opt/workbench-linux64-v1.5.0.zip
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
ENV ANTSPATH="/opt/ANTs/bin"
ENV PATH="${ANTSPATH}:${PATH}"
ENV ANTS_RANDOM_SEED=14193

#### AFNI 24.0.0
#RUN cd /opt && \
#    wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/0c88034741084793bd56/?dl=1 && \
#    wget --content-disposition -P /opt/ http://30.30.30.141:8080/f/413153c2166440fc9394/?dl=1 && \
#    bash /opt/OS_notes.linux_ubuntu_22_64_a_admin.txt 2>&1 | tee o.ubuntu_22_a.txt && \
#    tcsh /opt/OS_notes.linux_ubuntu_22_64_b_user.tcsh 2>&1 | tee o.ubuntu_22_b.txt && \
#    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
#    apt remove python3-matplotlib -y && apt autoremove -y && \
#    cd ~ && mv /root/abin /opt && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget --content-disposition -P /opt/ http://30.30.30.204/DeepPrep_new/AFNI_linux-ubuntu-22-64_24.0.00.tar.gz && tar -C /opt -xzvf /opt/AFNI_linux-ubuntu-22-64_24.0.00.tar.gz && rm /opt/AFNI_linux-ubuntu-22-64_24.0.00.tar.gz
RUN wget --content-disposition -P /opt/ http://30.30.30.204/DeepPrep_new/libxp6_1.0.2-2_amd64.deb && dpkg -i /opt/libxp6_1.0.2-2_amd64.deb && rm /opt/libxp6_1.0.2-2_amd64.deb
ENV PATH="/opt/abin:${PATH}"
RUN apt-get update && apt-get --no-install-recommends -y install libxpm-dev libxft2 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

### mriqc
#COPY --from=freesurfer/synthstrip@sha256:f19578e5f033f2c707fa66efc8b3e11440569facb46e904b45fd52f1a12beb8b /freesurfer/models/synthstrip.1.pt /opt/freesurfer/models/synthstrip.1.pt
RUN mkdir ${FREESURFER_HOME}/models && wget --content-disposition -P ${FREESURFER_HOME}/models http://30.30.30.204/model/SynthStrip/synthstrip.1.pt  # synthstrip.1.pt

### matplotlib
RUN pip3 install matplotlib==3.8.4  && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

### default template
RUN python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin6Asym', desc=None, resolution=2, suffix='T1w', extension='nii.gz')"
RUN python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin2009cAsym', desc=None, resolution=2, suffix='T1w', extension='nii.gz')"
RUN python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin2009cAsym', desc='brain', resolution=2, suffix='mask', extension='nii.gz')"
RUN python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin2009cAsym', desc='fMRIPrep', resolution=2, suffix='boldref', extension='nii.gz')"
RUN python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin2009cAsym', label='brain', resolution=1, suffix='probseg', extension='nii.gz')"

COPY deepprep/model/FastCSR /opt/model/FastCSR
COPY deepprep/model/SUGAR /opt/model/SUGAR
COPY deepprep/model/SynthMorph /opt/model/SynthMorph

# Dev
#COPY deepprep/FastCSR /opt/DeepPrep/deepprep/FastCSR
#COPY deepprep/SUGAR /opt/DeepPrep/deepprep/SUGAR
#COPY deepprep/FastSurfer /opt/DeepPrep/deepprep/FastSurfer
#COPY deepprep/SynthMorph /opt/DeepPrep/deepprep/SynthMorph
#COPY deepprep/nextflow /opt/DeepPrep/deepprep/nextflow
#COPY deepprep/deepprep.sh /opt/DeepPrep/deepprep/deepprep.sh
# release
ENV DEEPPREP_VERSION="24.1.0"
ENV DEEPPREP_HASH="a3f8f03"
RUN git clone --recursive --single-branch -b ${DEEPPREP_VERSION} https://github.com/pBFSLab/DeepPrep.git /opt/DeepPrep \
    && cd /opt/DeepPrep && git checkout ${DEEPPREP_HASH} && rm -r /opt/DeepPrep/.git

RUN chmod 755 /opt/DeepPrep/deepprep/deepprep.sh && chmod 755 /opt/DeepPrep/deepprep/nextflow/bin/*.py
ENV PATH="/opt/DeepPrep/deepprep/nextflow/bin:${PATH}"

### source ${FREESURFER_HOME}/SetUpFreeSurfer.sh
ENV FSLDISPLAY=/usr/bin/display
ENV FREESURFER=${FREESURFER_HOME}
ENV FSL_DIR="${FSLDIR}"
ENV OS=Linux
ENV MINC_BIN_DIR=${FREESURFER_HOME}/mni/bin
ENV FSFAST_HOME=${FREESURFER_HOME}/fsfast
ENV MNI_DATAPATH=${FREESURFER_HOME}/mni/data
ENV FS_OVERRIDE=0
ENV FUNCTIONALS_DIR=${FREESURFER_HOME}/sessions
ENV MINC_LIB_DIR=${FREESURFER_HOME}/mni/lib
ENV FMRI_ANALYSIS_DIR=${FREESURFER_HOME}/fsfast
ENV MNI_DIR=${FREESURFER_HOME}/mni
ENV PERL5LIB=${FREESURFER_HOME}/mni/share/perl5
ENV MNI_PERL5LIB=${FREESURFER_HOME}/mni/share/perl5
ENV LOCAL_DIR=${FREESURFER_HOME}/local
ENV FIX_VERTEX_AREA=""
ENV FSLCONVERT=/usr/bin/convert
ENV SUBJECTS_DIR=${FREESURFER_HOME}/subjects
ENV FSF_OUTPUT_FORMAT=nii.gz
ENV FSL_BIN=${FSLDIR}/bin
ENV PATH="${FREESURFER_HOME}/bin:${FREESURFER_HOME}/fsfast/bin:${FREESURFER_HOME}/tktools:${FSLDIR}/bin:${FREESURFER_HOME}/mni/bin:${PATH}"

## CMD
ENTRYPOINT ["/opt/DeepPrep/deepprep/deepprep.sh"]
