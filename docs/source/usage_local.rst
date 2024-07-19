.. include:: links.rst


-------------------
Usage Notes (Local)
-------------------


===============
The BIDS Format
===============

DeepPrep is able to end-to-end preprocess anatomical and functional MRI data for different data size ranging
from a single participant to a HUGE dataset. It is also flexible to run the anatomical part or functional part
that requires a complete Recon folder to be specified. The DeepPrep workflow takes the directory of the dataset
that is to be processed as the input, which is required to be in the valid BIDS format.
It is highly recommended that you validate your dataset with this free, online `BIDS Validator`_.

For more information about BIDS and BIDS-Apps, please check the `NiPreps portal`_.


======================
The FreeSurfer License
======================
DeepPrep uses FreeSurfer tools and thus requires a valid license.

    To obtain a FreeSurfer license, simply register for free at
    https://surfer.nmr.mgh.harvard.edu/registration.html.

Please make sure that a valid license file is passed into DeepPrep.
For example, if the license is stored in the ``$HOME/freesurfer/license.txt`` file on
the host system, the ``<fs_license_file>`` in command ``-v <fs_license_file>:/fs_license.txt`` should be replaced with the valid path: ::

    $ -v $HOME/freesurfer/license.txt:/fs_license.txt



=================
Docker User Guide
=================


Command-Line Arguments
======================

DeepPrep: Deep learning empowered preprocessing workflow 24.1.1:

.. code-block:: none

   usage: deepprep-docker [bids_dir] [output_dir] [{participant}] [--bold_task_type '[task1 task2 task3 ...]']
                          [--fs_license_file PATH] [--participant_label '[001 002 003 ...]']
                          [--subjects_dir PATH] [--skip_bids_validation]
                          [--anat_only] [--bold_only] [--bold_sdc] [--bold_confounds]
                          [--bold_surface_spaces '[None fsnative fsaverage fsaverage6 ...]']
                          [--bold_volume_space {None MNI152NLin6Asym MNI152NLin2009cAsym}] [--bold_volume_res {02 03...}]
                          [--device { {auto 0 1 2...} cpu}]
                          [--cpus 10] [--memory 20]
                          [--ignore_error] [--resume]


Sample Docker Command
======================

.. code-block:: none
    :linenos:

    $ docker run -it --rm --gpus all \
                 -v <bids_dir>:/input \
                 -v <output_dir>:/output \
                 -v <fs_license_file>:/fs_license.txt \
                 pbfslab/deepprep:24.1.1 \
                 /input \
                 /output \
                 participant \
                 --bold_task_type rest \
                 --fs_license_file /fs_license.txt

**Let's dig into the mandatory commands**
    + ``<bids_dir>`` - refers to the directory of the input dataset, which should be in `BIDS format`_.
    + ``<output_dir>`` - refers to the directory for the outputs of DeepPrep.
    + ``<fs_license_file>`` - the directory of a valid FreeSurfer License.
    + ``deepprep:24.1.1`` - the latest version of the Docker image. You can specify the version by ``deepprep:<version>``.
    + ``participant`` - refers to the analysis level.
    + ``--bold_task_type`` - the task label of BOLD images (i.e. ``rest``, ``motor``).

**Dig further (optional commands)**
    + ``--participant_label`` - the subject ID you want to process, i.e. ``'sub-001 sub-002'``. Otherwise, all the subjects in the ``<bids_dir>`` will be processed by default.
    + ``--subjects_dir`` - the output directory of *Recon* files, default is ``<output_dir>/Recon``.
    + ``--skip_bids_validation`` - with this flag, the BIDS format validation of the input dataset will be skipped.
    + ``--anat_only`` - with this flag, only the *anatomical* images will be processed.
    + ``--bold_only`` - with this flag, only the *functional* images will be processed, where *Recon* files are pre-requested.
    + ``--bold_sdc`` - applies susceptibility distortion correction (SDC), default is ``True``.
    + ``--bold_confounds`` - generates confounds derived from BOLD fMRI, such as head motion variables and global signals; the default is ``True``.
    + ``--bold_surface_spaces`` - specifies surface template spaces, i.e. ``'fsnative fsaverage fsaverage[3-6]'``, default is ``'fsaverage6'``. (*Note:* the space names must be quoted using single quotation marks)
    + ``--bold_volume_space`` - specifies an available volumetric space from `TemplateFlow`_, default is ``MNI152NLin6Asym``.
    + ``--bold_volume_res`` - specifies the spatial resolution of the corresponding template space from `TemplateFlow`_, default is ``02``.
    + ``--device`` - specifies the device. The default is ``auto``, which automatically selects a GPU device; ``0`` specifies the first GPU device; ``cpu`` refers to CPU only.
    + ``--cpus`` - refers to the maximum CPUs for usage, which should be integer values > 0.
    + ``--memory`` - refers to the maximum memory resources for usage, which should be integer values > 0.
    + ``--ignore_error`` - ignores the errors that occurred during processing.
    + ``--resume`` - allows the DeepPrep pipeline to start from the last exit point.


Quick Start
-----------

Get started with a ``test_sample``, `download here`_.

The BIDS formatted sample contains one subject with one anatomical image and two functional images.

1. Run with GPU (**recommended**)

.. code-block:: none
    :linenos:

    $ docker run -it --rm --gpus all \
                 -v ~/test_sample:/input \
                 -v ~/deepprep_output:/output \
                 -v ~/license.txt:/fs_license.txt \
                 pbfslab/deepprep:24.1.1 \
                 /input \
                 /output \
                 participant \
                 --bold_task_type 6cat \
                 --fs_license_file /fs_license.txt

**Docker arguments**
    + ``-it`` - (optional) starts the container in an interactive mode.
    + ``--rm`` - (optional) the container will be removed upon exit.
    + ``--gpus all`` - (optional) assign all the available GPUs on the local host to the container. *This flag is highly recommended*.
    + ``-v`` - mounts your local directories to the directories inside the container. The input directories should be in *absolute path* to avoid any mistakes.


2. Run with CPU only

.. code-block:: none
    :linenos:

    $ docker run -it --rm \
                 -v ~/test_sample:/input \
                 -v ~/deepprep_output:/output \
                 -v ~/license.txt:/fs_license.txt \
                 pbfslab/deepprep:24.1.1 \
                 /input \
                 /output \
                 participant \
                 --bold_task_type 6cat \
                 --fs_license_file /fs_license.txt \
                 --device cpu

**DeepPrep arguments**
    + ``--device cpu`` - refers to CPU only.



======================
Singularity User Guide
======================

Before You Start
========================

1. Download the Singularity

.. code-block:: none

    $ curl -C - -O https://download.anning.info/ninganme-public/DeepPrep/SingularityImage/deepprep_24.1.0.sif

Then you will get: ``<saved_path>/deepprep_24.1.0.sif``

The Singularity can be executed in a manner similar to the Docker command.


2. Follow the **warning message** in :doc:`installation` to set up your environment.


3.  Make sure you have ``<output_dir>`` created before executing the Singularity.


Sample Singularity Command
==========================

1. Here's a sample command only relies on CPU.

.. code-block:: none
    :linenos:

    $ singularity run --cleanenv \
                 -B <bids_dir>:/input \
                 -B <output_dir>:/output \
                 -B <fs_license_file>:/fs_license.txt \
                 <saved_path>/deepprep_24.1.0.sif \
                 /input \
                 /output \
                 participant \
                 --bold_task_type rest \
                 --fs_license_file /fs_license.txt \
                 --device cpu

2. If you wish to use GPU as well, add the ``--nv`` and ``--device all`` like:

.. code-block:: none
    :linenos:

    $ singularity run --cleanenv --nv \
                 -B <bids_dir>:/input \
                 -B <output_dir>:/output \
                 -B <fs_license_file>:/fs_license.txt \
                 <saved_path>/deepprep_24.1.0.sif \
                 /input \
                 /output \
                 participant \
                 --bold_task_type rest \
                 --fs_license_file /fs_license.txt \
                 --device all



.. container:: congratulation

   **Congratulations! You are all set!**