# syntax = docker/dockerfile:1.5

FROM ubuntu:jammy-20240627.1 as baseimage

ARG DEPENDENCE_URL="http://localhost"
#ARG DEPENDENCE_URL="https://download.anning.info/ninganme-public"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl vim git ca-certificates rsync build-essential \
        python3 python3-pip python3-dev python2 \
        && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# FreeSurfer 7.2.0
FROM baseimage as freesurfer
RUN wget --content-disposition -P /opt/ ${DEPENDENCE_URL}/DeepPrep/deps/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && tar -C /opt -xzvf /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && rm /opt/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz
COPY --from=freesurfer/synthstrip@sha256:f19578e5f033f2c707fa66efc8b3e11440569facb46e904b45fd52f1a12beb8b /freesurfer/models/synthstrip.1.pt /opt/freesurfer/models/synthstrip.1.pt

# FSL 6.0.5.1
FROM baseimage as fsl
RUN wget --content-disposition -P /opt/ ${DEPENDENCE_URL}/DeepPrep/deps/FSL_6.0.5.1.tar.gz && tar -C /opt -xzvf /opt/FSL_6.0.5.1.tar.gz && rm /opt/FSL_6.0.5.1.tar.gz

# Workbench 1.5.0
FROM baseimage as workbench
RUN apt-get update && apt-get install -y --no-install-recommends unzip
RUN wget -P /opt/ ${DEPENDENCE_URL}/DeepPrep/deps/workbench-linux64-v1.5.0.zip && unzip /opt/workbench-linux64-v1.5.0.zip -d /opt && rm /opt/workbench-linux64-v1.5.0.zip

# AFNI 24.0.0
##RUN cd /opt && \
##    wget --content-disposition -P /opt/ ${DEPENDENCE_URL}/DeepPrep/deps/f/0c88034741084793bd56/?dl=1 && \
##    wget --content-disposition -P /opt/ ${DEPENDENCE_URL}/DeepPrep/deps/f/413153c2166440fc9394/?dl=1 && \
##    bash /opt/OS_notes.linux_ubuntu_22_64_a_admin.txt 2>&1 | tee o.ubuntu_22_a.txt && \
##    tcsh /opt/OS_notes.linux_ubuntu_22_64_b_user.tcsh 2>&1 | tee o.ubuntu_22_b.txt && \
##    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
##    apt remove python3-matplotlib -y && apt autoremove -y && \
##    cd ~ && mv /root/abin /opt && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
FROM baseimage as afni
RUN wget --content-disposition -P /opt/ ${DEPENDENCE_URL}/DeepPrep/deps/AFNI_linux-ubuntu-22-64_24.0.00.tar.gz && tar -C /opt -xzvf /opt/AFNI_linux-ubuntu-22-64_24.0.00.tar.gz && rm /opt/AFNI_linux-ubuntu-22-64_24.0.00.tar.gz
RUN wget --content-disposition -P /opt/ ${DEPENDENCE_URL}/DeepPrep/deps/libxp6_1.0.2-2_amd64.deb

# ANTs 2.3.1
##RUN apt-get update && apt-get install -y --no-install-recommends build-essential && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
##RUN cd ~ && \
##    git clone https://github.com/cookpa/antsInstallExample.git && \
##    cd antsInstallExample && \
##    bash installANTs.sh && \
##    cp -r install /opt/ANTs && \
##    cd ~ && rm -r antsInstallExample
FROM baseimage as ants
RUN wget --content-disposition -P /opt/ ${DEPENDENCE_URL}/DeepPrep/deps/ANTs_linux-ubuntu22-amd64_2.3.1.tar.gz && tar -C /opt -xzvf /opt/ANTs_linux-ubuntu22-amd64_2.3.1.tar.gz && rm /opt/ANTs_linux-ubuntu22-amd64_2.3.1.tar.gz

# Nextflow
FROM baseimage as nextflow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # open jdk
    openjdk-11-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
## Install openjdk
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/lib/server" \
    JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
RUN mkdir -p /opt/nextflow/bin && cd /opt/nextflow/bin && wget -qO- https://get.nextflow.io | bash && \
    chmod 755 nextflow && ./nextflow && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN /opt/nextflow/bin/nextflow

# micromamba
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as micromamba

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
ENV MAMBA_ROOT_PREFIX="/opt/conda"
RUN micromamba create -y -n deepprep -c conda-forge python=3.10.14 pip
RUN micromamba shell init -s bash && echo "micromamba activate deepprep" >> $HOME/.bashrc

# UV_USE_IO_URING for apparent race-condition (https://github.com/nodejs/node/issues/48444)
# Check if this is still necessary when updating the base image.
ENV PATH="/opt/conda/envs/deepprep/bin:$PATH" \
    UV_USE_IO_URING=0
#ENV PATH="/opt/node-v20.18.1-linux-x64/bin:$PATH"
#RUN wget --content-disposition -P /opt/ https://nodejs.org/dist/v20.18.1/node-v20.18.1-linux-x64.tar.xz && tar -C /opt -xvf /opt/node-v20.18.1-linux-x64.tar.xz && rm /opt/node-v20.18.1-linux-x64.tar.xz
#RUN npm install -g svgo@^3.2.0 bids-validator@^1.14.0 && rm -r ~/.npm
#RUN cp /usr/bin/bids-validator /opt/conda/envs/deepprep/bin

### Install Python lib
#RUN git clone https://github.com/nighres/nighres.git --branch release-1.5.0 --single-branch && \
#    apt-get -y --no-install-recommends install python3-jcc && \
#    cd nighres && ln -sf /usr/lib/jvm/java-11-openjdk-amd64 /usr/lib/jvm/default-java && \
#    bash build.sh && pip3 install .
#RUN pip3 config set global.extra-index-url "https://download.pytorch.org/whl/cu118"
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip3 install \
    # https://data.pyg.org/whl/torch-2.0.1+cu118.html
    pyg_lib==0.3.1+pt20cu118 torch_scatter==2.1.2+pt20cu118 torch_sparse==0.6.18+pt20cu118 torch_cluster==1.6.3+pt20cu118 torch_spline_conv==1.2.2+pt20cu118 \
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
    git+https://github.com/NingAnMe/fmriprep.git@deepprep smriprep==0.13.2 sdcflows==2.8.1 scipy==1.11.4 \
    git+https://github.com/Deep-MI/LaPy.git@v1.0.1 \
    git+https://github.com/voxelmorph/voxelmorph.git@ca28315d0ba24cd8946ac4f6ed081e049e5264fe \
    git+https://github.com/NingAnMe/mriqc.git@deepprep \
    --trusted-host localhost -f http://localhost/DeepPrep/deps/cp310_whl \
    && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

# SynthMorph
RUN pip3 install numpy==1.26.4 && pip3 cache purge && rm -rf /tmp/* /var/tmp/*
RUN pip3 install git+https://github.com/freesurfer/surfa.git@ded5f1d3d90e223050ab7792ac4760c3242e43c7 && pip3 cache purge && rm -rf /tmp/* /var/tmp/*
RUN pip uninstall neurite -y && pip3 install git+https://github.com/adalca/neurite@682f828b7b5fa652d7205c894c7fe667f1a26251 && pip3 cache purge && rm -rf /tmp/* /var/tmp/*
# matplotlib
#RUN pip3 install matplotlib==3.8.4 && pip3 cache purge && rm -rf /tmp/* /var/tmp/*
# mriqc
RUN pip3 install mriqc-learn==0.0.2 --no-deps && pip3 cache purge && rm -rf /tmp/* /var/tmp/*
# python-redis-lock
RUN pip3 install python-redis-lock==4.0.0  && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

RUN pip3 install streamlit==1.38.0 protobuf==3.20  && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

RUN pip3 install git+https://github.com/netneurolab/neuromaps.git@ae3c88a60746c0137dc81b15130a12a25946252b && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

RUN pip3 install pyvista==0.45.2 && pip3 cache purge && rm -rf /tmp/* /var/tmp/*

# Start from this Docker image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

##
#COPY Docker/source.list /etc/apt/sources.list

ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

## Install * in Docker image
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install time zone
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y --no-install-recommends curl rsync tzdata && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure PPAs for libpng12 and libxp6. FreeSurfer and AFNI
RUN curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xEA8CACC073C3DB2A" | gpg --dearmor -o /usr/share/keyrings/linuxuprising.gpg && \
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xA1301338A3A48C4A" | gpg --dearmor -o /usr/share/keyrings/zeehio.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/linuxuprising.gpg] https://ppa.launchpadcontent.net/linuxuprising/libpng12/ubuntu jammy main" > /etc/apt/sources.list.d/linuxuprising.list && \
    echo "deb [signed-by=/usr/share/keyrings/zeehio.gpg] https://ppa.launchpadcontent.net/zeehio/libxp/ubuntu jammy main" > /etc/apt/sources.list.d/zeehio.list
#RUN GNUPGHOME=/tmp gpg --keyserver hkps://keyserver.ubuntu.com --no-default-keyring --keyring /usr/share/keyrings/linuxuprising.gpg --recv 0xEA8CACC073C3DB2A \
#    && GNUPGHOME=/tmp gpg --keyserver hkps://keyserver.ubuntu.com --no-default-keyring --keyring /usr/share/keyrings/zeehio.gpg --recv 0xA1301338A3A48C4A \
#    && echo "deb [signed-by=/usr/share/keyrings/linuxuprising.gpg] https://ppa.launchpadcontent.net/linuxuprising/libpng12/ubuntu jammy main" > /etc/apt/sources.list.d/linuxuprising.list \
#    && echo "deb [signed-by=/usr/share/keyrings/zeehio.gpg] https://ppa.launchpadcontent.net/zeehio/libxp/ubuntu jammy main" > /etc/apt/sources.list.d/zeehio.list
#RUN curl -sSL -o linuxuprising.asc 'https://keyserver.ubuntu.com/pks/lookup?search=0xEA8CACC073C3DB2A&fingerprint=on&op=index' && apt-key add linuxuprising.asc
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # wand for svg
    libmagickwand-dev \
    # tksurfer
    libxmu6 \
    # redis
    redis-server \
    # open jdk
    openjdk-11-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

## Install openjdk
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/lib/server" \
    JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

#### Install Redis
RUN sed -i '147c\supervised systemd' /etc/redis/redis.conf
RUN sed -i.bak '/^save / s/^/#/' /etc/redis/redis.conf

#### bids-validator /usr/bin/bids-validator
ENV UV_USE_IO_URING=0
ENV PATH="/opt/node-v20.18.1-linux-x64/bin:$PATH"
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget --content-disposition -P /opt/ https://nodejs.org/dist/v20.18.1/node-v20.18.1-linux-x64.tar.xz && tar -C /opt -xvf /opt/node-v20.18.1-linux-x64.tar.xz && rm /opt/node-v20.18.1-linux-x64.tar.xz
RUN npm install -g svgo@^3.2.0 bids-validator@^1.14.0 && rm -r ~/.npm

## aws cli
RUN apt-get update && apt-get install -y --no-install-recommends unzip && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -rf aws awscliv2.zip

## Install FreeSurfer
COPY --from=freesurfer /opt/freesurfer /opt/freesurfer
RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    binutils \
    libgomp1 \
    perl \
    psmisc \
    tcsh \
    libpng12-0 \
    libglu1-mesa && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV FREESURFER_HOME=/opt/freesurfer
## source ${FREESURFER_HOME}/SetUpFreeSurfer.sh
ENV OS=Linux \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT=nii.gz \
    FREESURFER=${FREESURFER_HOME}
ENV SUBJECTS_DIR=${FREESURFER_HOME}/subjects \
    FUNCTIONALS_DIR=${FREESURFER_HOME}/sessions \
    MNI_DIR=${FREESURFER_HOME}/mni \
    LOCAL_DIR=${FREESURFER_HOME}/local \
    FSFAST_HOME=${FREESURFER_HOME}/fsfast \
    FMRI_ANALYSIS_DIR=${FREESURFER_HOME}/fsfast \
    MINC_BIN_DIR=${FREESURFER_HOME}/mni/bin \
    MINC_LIB_DIR=${FREESURFER_HOME}/mni/lib \
    MNI_DATAPATH=${FREESURFER_HOME}/mni/data
ENV PERL5LIB=${FREESURFER_HOME}/mni/share/perl5 \
    MNI_PERL5LIB=${FREESURFER_HOME}/mni/share/perl5
ENV PATH="${FREESURFER_HOME}/tktools:${FREESURFER_HOME}/bin:${FREESURFER_HOME}/fsfast/bin:${FREESURFER_HOME}/mni/bin:${PATH}"

## MRIQC
##COPY --from=freesurfer/synthstrip@sha256:f19578e5f033f2c707fa66efc8b3e11440569facb46e904b45fd52f1a12beb8b /freesurfer/models/synthstrip.1.pt /opt/freesurfer/models/synthstrip.1.pt
##RUN mkdir ${FREESURFER_HOME}/models && wget --content-disposition -P ${FREESURFER_HOME}/models ${DEPENDENCE_URL}/DeepPrep/deps/model/SynthStrip/synthstrip.1.pt  # synthstrip.1.pt

## FSL 6.0.5.1
COPY --from=fsl /opt/fsl /opt/fsl
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV FSLDIR="/opt/fsl"
ENV FSLDISPLAY=/usr/bin/display \
    PYTHONNOUSERSITE=1 \
    FSL_DIR="${FSLDIR}" \
    FSL_BIN=${FSLDIR}/bin \
    FSLCONVERT=/usr/bin/convert \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"
ENV PATH="${FSLDIR}/bin:${PATH}"

### Install workbench
COPY --from=workbench /opt/workbench /opt/workbench
ENV PATH="/opt/workbench/bin_linux64:${PATH}"

## ANTs  2.3.1
COPY --from=ants /opt/ANTs /opt/ANTs
ENV ANTSPATH="/opt/ANTs/bin"
ENV ANTS_RANDOM_SEED=14193
ENV PATH="${ANTSPATH}:${PATH}"

##### AFNI 24.0.0
COPY --from=afni /opt/abin /opt/abin
##RUN wget --content-disposition -P /opt/ ${DEPENDENCE_URL}/DeepPrep/deps/libxp6_1.0.2-2_amd64.deb && dpkg -i /opt/libxp6_1.0.2-2_amd64.deb && rm /opt/libxp6_1.0.2-2_amd64.deb
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxpm-dev \
    libpng12-0 \
    libxp6 \
    libxft2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV PATH="/opt/abin:${PATH}"

## Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users deepprep
WORKDIR /home/deepprep
ENV HOME="/home/deepprep"

## Install nextflow
COPY --from=nextflow /opt/nextflow /opt/nextflow
RUN /opt/nextflow/bin/nextflow
RUN find $HOME/.nextflow -type d -exec chmod go=u {} + && \
    find $HOME/.nextflow -type f -exec chmod go=u {} +
ENV PATH="/opt/nextflow/bin:${PATH}" \
    NXF_OFFLINE='true'

## Python
COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=micromamba /opt/conda/envs/deepprep /opt/conda/envs/deepprep
ENV MAMBA_ROOT_PREFIX="/opt/conda"
RUN micromamba shell init -s bash && \
    echo "micromamba activate deepprep" >> $HOME/.bashrc
ENV PATH="/opt/conda/envs/deepprep/bin:$PATH" \
    CPATH="/opt/conda/envs/deepprep/include:$CPATH" \
    LD_LIBRARY_PATH="/opt/conda/envs/deepprep/lib:$LD_LIBRARY_PATH"

#### default template
RUN python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin6Asym', desc=None, resolution=1, suffix='T1w', extension='nii.gz')" && \
    python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin6Asym', desc=None, resolution=2, suffix='T1w', extension='nii.gz')" && \
    python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin2009cAsym', desc=None, resolution=1, suffix='T1w', extension='nii.gz')" && \
    python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin2009cAsym', desc=None, resolution=2, suffix='T1w', extension='nii.gz')" && \
    python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin2009cAsym', desc='brain', resolution=2, suffix='mask', extension='nii.gz')" && \
    python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin2009cAsym', desc='fMRIPrep', resolution=2, suffix='boldref', extension='nii.gz')" && \
    python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin2009cAsym', label='brain', resolution=1, suffix='probseg', extension='nii.gz')"

#### default template for CIFTI
RUN python3 -c "from neuromaps.datasets import fetch_fsaverage; fetch_fsaverage(density='41k')" && \
    python3 -c "from neuromaps.datasets import fetch_fslr; fetch_fslr(density='32k')"
RUN python3 -c "import templateflow.api as tflow; tflow.get('MNI152NLin6Asym', resolution='02', suffix='dseg', atlas='HCP', raise_empty=True)" && \
    python3 -c "import templateflow.api as tflow; tflow.get('fsaverage', density='41k', suffix='sphere', raise_empty=True)" && \
    python3 -c "import templateflow.api as tflow; tflow.get('fsLR', density='32k', suffix='midthickness', raise_empty=True)" && \
    python3 -c "import templateflow.api as tflow; tflow.get('fsLR', density='32k', suffix='dparc', desc='nomedialwall', raise_empty=True)"

RUN find $HOME/.cache/templateflow -type d -exec chmod go=u {} + && \
    find $HOME/.cache/templateflow -type f -exec chmod go=u {} +

COPY deepprep/model/FastCSR /opt/model/FastCSR
COPY deepprep/model/SUGAR /opt/model/SUGAR
COPY deepprep/model/SynthMorph /opt/model/SynthMorph

# Dev
COPY deepprep/FreeSurfer /opt/freesurfer
COPY deepprep/FastCSR /opt/DeepPrep/deepprep/FastCSR
COPY deepprep/SUGAR /opt/DeepPrep/deepprep/SUGAR
COPY deepprep/FastSurfer /opt/DeepPrep/deepprep/FastSurfer
COPY deepprep/SynthMorph /opt/DeepPrep/deepprep/SynthMorph
COPY deepprep/nextflow /opt/DeepPrep/deepprep/nextflow
COPY deepprep/web /opt/DeepPrep/deepprep/web
COPY deepprep/qc /opt/DeepPrep/deepprep/qc
COPY deepprep/rest /opt/DeepPrep/deepprep/rest
COPY deepprep/deepprep.sh /opt/DeepPrep/deepprep/deepprep.sh
# release
ENV DEEPPREP_VERSION="25.1.1"

RUN chmod 755 /opt/DeepPrep/deepprep/deepprep.sh && chmod 755 /opt/DeepPrep/deepprep/nextflow/bin/*.py
RUN chmod 755 /opt/DeepPrep/deepprep/web/pages/*.sh && chmod 755 /opt/DeepPrep/deepprep/rest/denoise/bin/*.py

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} +

#RUN /opt/conda/envs/deepprep/bin/pip3 install streamlit==1.38.0 protobuf==3.20  && pip3 cache purge && rm -rf /tmp/* /var/tmp/*
EXPOSE 8501
## CMD
ENTRYPOINT ["/opt/DeepPrep/deepprep/deepprep.sh"]
