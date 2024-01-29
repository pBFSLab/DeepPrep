.. include:: links.rst

============
Installation
============

Running with Docker step-by-step
---------------------------------

DeepPrep provides a docker image as the recommended way to get started.

1. Install Docker if you don't have one (`Docker Installation Page`_).

.. _Docker Installation Page: https://www.docker.com/get-started/


2. Test Docker with the ``hello-world`` image:

.. code-block:: python
    :linenos:

    $ docker run -it --rm hello-world

The following message should appear:

.. code-block:: python
    :linenos:

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
    -v <input_dir>:/BIDS_DATASET \
    -v <output_dir>:/DEEPPREP_RESULT_DIR \
    -v <freesurfer_license>:/usr/local/freesurfer/license.txt \
    -v <config_file>:/nextflow.docker.local.config \
    deepprep:latest \
    -with-report /DEEPPREP_RESULT_DIR/QC/report.html \
    -with-timeline /DEEPPREP_RESULT_DIR/QC/timeline.html \
    -resume \
    -c /nextflow.docker.local.config \
    --bids_dir /BIDS_DATASET \
    --bold_task_type rest

**Let's dig into the mandatory commands**
    + ``--gpus all`` assigns all the available GPUs on the local host to the container.
    + ``-v`` flag mounts your local directories to the directories inside the container. The input directories should be in *absolute path* to avoid any confusions.
    + The ``<input_dir>`` should be in `bids format`_.
    .. _bids format: https://bids-specification.readthedocs.io/en/stable/index.html
    + ``<freesurfer_license>`` is the directory of a valid FreeSurfer License.
    + ``deepprep:latest`` is the docker image.

**Dig further (optional commands)**
    + ``-it`` starts the container in an interactive mode.
    + ``--rm`` the container will be removed when exit.
    + ``-resume`` allows the deepprep pipeline starts from the last exit point.
    + ``--bold_task_type`` is the task type of BOLD images (i.e. rest, motor).
    + ``--subjects`` is the subject id you want to process, otherwise all the subjects in the ``<input_dir>`` will be processed.
    + ``--subjects_dir`` is the output directory of *Recon* files, the default directory is ``<output_dir>/Recon``.
    + ``--anat_only True`` only the anatomical images will be processed, default is ``False``.
    + ``--bold_only True`` only the functional images will be processed, where *Recon* files are pre-requests. The default is ``False``.
    + ``--bold_sdc`` applies Susceptibility Distortion Correction (SDC) on BOLD images, default is ``True``.
