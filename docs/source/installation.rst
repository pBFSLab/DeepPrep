.. include:: links.rst

============
Installation
============

Run with Docker (step-by-step)
---------------------------------

DeepPrep provides a Docker image as the recommended way to get started.

.. note::
    **Required Environment**
        + Graphics Driver VRAM: >= 24GB
        + Disk: >= 20G
        + RAM: >= 16GB
        + Swap space: >=16G
        + Ubuntu:  >= 20.04
        + NVIDIA Driver: >= 11.8

1. Install Docker if you don't have one (`Docker Installation Page`_).

.. _Docker Installation Page: https://www.docker.com/get-started/


2. Test Docker with the ``hello-world`` image:

.. code-block:: python
    :linenos:

    $ docker run -it --rm hello-world

The following message should appear:

::

    Hello from Docker!
    This message shows that your installation appears to be working correctly.

    To generate this message, Docker took the following steps:
     1. The Docker client contacted the Docker daemon.
     2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
        (amd64)
     3. The Docker daemon created a new container from that image which runs the
        executable that produces the output you are currently reading.
     4. The Docker daemon streamed that output to the Docker client, which sent it
        to your terminal.

    To try something more ambitious, you can run an Ubuntu container with:
     $ docker run -it ubuntu bash

    Share images, automate workflows, and more with a free Docker ID:
     https://hub.docker.com/

    For more examples and ideas, visit:
     https://docs.docker.com/get-started/

3. Please make sure GPUs are accessible by adding the flag ``--gpus all``:

.. code-block:: python
    :linenos:

    $ docker run -it --gpus all --rm hello-world

The same output as before is expected.

4. Pull the Docker image

.. code-block:: python
    :linenos:

    $ docker pull deepprep:latest

5. Sample Docker command

.. code-block:: bash
    :linenos:

    $ docker run -it --rm --gpus all \
    -v <bids_dir>:/BIDS_DATASET \
    -v <output_dir>:/DEEPPREP_RESULT_DIR \
    -v <freesurfer_license>:/usr/local/freesurfer/license.txt \
    deepprep:latest \
    <bids_dir> <output_dir> participant \
    --bold_task_type rest \
    --fs_license_file freesurfer_license \
    --bold_surface_spaces 'fsnative fsaverage6' \
    --bold_template_space MNI152NLin6Asym \
    --bold_template_res 02 \
    -resume

**Let's dig into the mandatory commands**
    + ``--gpus all`` - (Docker argument) assigns all the available GPUs on the local host to the container.
    + ``-v`` - (Docker argument) flag mounts your local directories to the directories inside the container. The input directories should be in *absolute path* to avoid any confusions.
    + ``<bids_dir>`` - refers to the directory of the input dataset, which should be in `BIDS format`_.
    .. _BIDS format: https://bids-specification.readthedocs.io/en/stable/index.html
    + ``<output_dir>`` - refers to the directory for the outputs of DeepPrep.
    + ``<freesurfer_license>`` - the directory of a valid FreeSurfer License.
    + ``deepprep:latest`` - the latest version of the Docker image. One can specify the version by ``deepprep:<version>``.
    + ``participant`` - refers to the analysis level.
    + ``--bold_task_type`` - the task label of BOLD images (i.e. rest, motor).

**Dig further (optional commands)**
    + ``-it`` - (Docker argument) starts the container in an interactive mode.
    + ``--rm`` - (Docker argument) the container will be removed when exit.
    + ``--subjects_dir`` - the output directory of *Recon* files, the default directory is ``<output_dir>/Recon``.
    + ``--participant_label`` - the subject id you want to process, otherwise all the subjects in the ``<bids_dir>`` will be processed.
    + ``--anat_only`` - with this flag, only the *anatomical* images will be processed.
    + ``--bold_only`` - with this flag, only the *functional* images will be processed, where *Recon* files are pre-requested.
    + ``--bold_sdc`` - with this flag, susceptibility distortion correction (SDC) will be applied.
    + ``--bold_confounds`` - with this flag, confounds will be generated.
    + ``--bold_surface_spaces`` - specifies surfaces spaces, i.e. 'fsnative fsaverage fsaverage6'. (*Note:* the space names must be quoted using single quotation marks.)
    + ``--bold_template_space`` - specifies an available template space from `TemplateFlow`_, i.e. MNI152NLin6Asym.
    .. _TemplateFlow: https://www.templateflow.org/browse/
    + ``--bold_template_res`` - specifies the resolution of the corresponding template space from `TemplateFlow`_, i.e. 02.
    + ``--device`` - specifies the device, i.e. cpu.
    + ``--gpu_compute_capability`` - refers to the GPU compute capability, you can find yours `here`_.
    .. _here: https://developer.nvidia.com/cuda-gpus
    + ``--cpus`` - refers to the maximum CPUs for usage.
    + ``--memory`` - refers to the maximum memory resources for usage.
    + ``--freesurfer_home`` - the directory of the FreeSurfer home.
    + ``--deepprep_home`` - the directory of the DeepPrep home.
    + ``--templateflow_home`` - the directory of the TemplateFlow home.
    + ``--ignore_error`` - ignores the errors occurred during processing.
    + ``-resume`` - allows the DeepPrep pipeline starts from the last exit point.
