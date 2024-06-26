.. include:: links.rst


---------------------
Usage Notes (Cluster)
---------------------

=====================
Process in a Nutshell
=====================

1. log into HPC or cloud platforms that have **nodes** with resource application permissions.
2. Download the Singularity image to the **shared directory** on the cluster platform (or pull the Docker image directly).
3. Install java > 11 and Nextflow > 23.
4. Update the configuration file to meet the requirements for the specific platform.
5. Execute DeepPrep Singularity/Docker.




========================
Platform and Login Nodes
========================

 - The specified login nodes are typically applicable on HPC or cloud platforms, allowing users to submit jobs to the platform.

 - Job schedulers vary widely across HPC platforms, i.e. SLURM, PBS, SGE, etc. Thus, the commands for submitting jobs are different, i.e. ``sbatch`` for SLURM, ``qsub`` for PBS, etc.

 - Also, different services are used across various cloud platforms to submit jobs, i.e. AWS uses AWS Batch service, and Google Cloud uses Google Cloud Batch service.




=============================================
Download the Singularity image
=============================================
The Singularity image is typically used on HPC since users do not have permission to access the root directory.
Meanwhile, the Docker image is commonly used on cloud platforms.

.. warning::
    The Singularity image should be downloaded to a **shared directory** ``<shared_storage_path>``.



Download the DeepPrep sif image
-------------------------------------------------

.. code-block:: none

    $ curl -C - -O https://download.anning.info/ninganme-public/DeepPrep/SingularityImage/deepprep_24.1.0.sif

Then you will get: ``<shared_storage_path>/deepprep_24.1.0.sif``



=====================================
Download the Docker image (DockerHub)
=====================================
The Singularity image is typically used on AWS etc.

Download the DeepPrep image from the DockerHub
----------------------------------------------

.. code-block:: none

    $ docker pull pbfslab/deepprep:24.1.0


When its done, you will find the Docker image by this command ``docker image ls``, the ``REPOSITORY`` is ``pbfslab/deepprep`` with a ``TAG: 24.1.0``.




======================
Install Nextflow >= 23
======================

Please install `Nexflow`_ if you don't have one or the version < 23.

Check the version of Nextflow with ``nextflow -version``:

.. code-block:: none
    :emphasize-lines: 4

    nextflow -version

      N E X T F L O W
      version 23.04.3 build 5875
      created 11-08-2023 18:37 UTC (12-08-2023 02:37 CDT)
      cite doi:10.1038/nbt.3820
      http://nextflow.io


Sample --- Install on Ubuntu
----------------------------

**Java**

The Cluster implemented in DeepPrep is based on Nextflow, which requires JAVA > 11.
Please install `Java`_ if it doesn't exist or the version < 11. (`How to install Java`_)

Check the version of Java with ``java -version``:

.. code-block:: none
    :emphasize-lines: 3

    java -version

        openjdk version "11.0.21" 2023-10-17
        OpenJDK Runtime Environment (build 11.0.21+9-post-Ubuntu-0ubuntu120.04)
        OpenJDK 64-Bit Server VM (build 11.0.21+9-post-Ubuntu-0ubuntu120.04, mixed mode, sharing)


**Nextflow**

Install Nextflow:

.. code-block:: none

    $ export NEXTFLOW_BIN=/opt/nextflow/bin
    $ mkdir -p ${NEXTFLOW_BIN} && cd ${NEXTFLOW_BIN} && wget -qO- https://get.nextflow.io | bash
    $ chmod 755 ${NEXTFLOW_BIN}/nextflow
    $ export PATH=${NEXTFLOW_BIN}:${PATH}

Double check the Nextflow version >= 23 with ``nextflow -version``.





========================================
Update the Configuration file (with GPU)
========================================

.. warning::

    It is **mandatory** to update the DeepPrep configuration file to meet the requirements of your cluster platform.

DeepPrep is developed based on Nextflow, enabling a scalable computation on the `platforms list`_.

Let's first walk through the detailed introduction of executing DeepPrep on SLURM.
Following that, samples for SLURM, PBS, and AWS will be provided.


This is the configuration file used on SLURM with GPU Driver:

.. code-block:: none
    :emphasize-lines: 26-30

    //deepprep.slurm.gpu.config

    singularity.enabled = true
    singularity.autoMounts = false
    singularity.runOptions = '-e \
    --home ${output_dir}/WorkDir/home \
    --env TEMP=${output_dir}/WorkDir/tmp \
    --env TMP=${output_dir}/WorkDir/tmp \
    --env TMPDIR=${output_dir}/WorkDir/tmp \
    -B ${bids_dir} \
    -B ${output_dir} \
    -B ${subjects_dir} \
    -B ${fs_license_file}:/opt/freesurfer/license.txt \
    '

    process {
    //errorStrategy = 'ignore'

        executor = 'slurm'

        queue = 'cpu1,cpu2,fat'

        clusterOptions = { " --chdir=${nextflow_work_dir}" }

        container = "${container}"

        withLabel: with_gpu {
            queue = 'gpu2'
            clusterOptions = { " --chdir=${nextflow_work_dir} --gres=gpu:1" }
            singularity.runOptions = '-e --nv \
                --home ${output_dir}/WorkDir/home \
                --env TEMP=${output_dir}/WorkDir/tmp \
                --env TMP=${output_dir}/WorkDir/tmp \
                --env TMPDIR=${output_dir}/WorkDir/tmp \
                -B ${bids_dir} \
                -B ${output_dir} \
                -B ${subjects_dir} \
                -B ${fs_license_file}:/opt/freesurfer/license.txt \
                '
         }
    }


**Explanation**
    + ``singularity.enabled = true`` - Execution is based on the Singularity image. This command can be replaced with ``docker.enabled = true`` on cloud platforms if Docker image is used.
    + ``singularity.autoMounts = false`` - When set to ``true``, the host directories will be automatically mounted in the container upon execution. It relies on the ``user bin control``, which is enabled by default in Singularity installation. However, DeepPrep does not need this function; thus, it is set to ``false``.
    + ``singularity.runOptions`` - The personalized setting to execute the DeepPrep Singularity image. *Do NOT modify*.

    + ``process`` - defines the parameters for each process in DeepPrep.
    + ``executor = 'slurm'`` - indicates the executed environment is SLURM.
    + ``queue = 'cpu1,cpu2,fat'`` - specifies the resource from ``queue`` to be allocated. A list of available ``queue`` will be returned from the command ``sinfo``, and users need to *UPDATE this setting* with available resources from the list.
    + ``clusterOptions = { " --chdir=${nextflow_work_dir}" }`` - other personalized settings on the cluster, where ``--chdir=${nextflow_work_dir}`` is the personalized working directory.
    + ``container = "${container}"`` - the specified container to use.

**For GPU users**
    + ``withLabel: with_gpu`` - the personalized GPU setting.
    + ``queue = 'gpu2'`` - indicates the resource to be allocated from the GPU queue named `'gpu2'`. *UPDATE this setting* with available GPU queue (check with ``sinfo``).
    + ``clusterOptions = { " --gres=gpu:1" }`` - specifies the resource required for a job submission, ``gpu:1`` indicates one GPU.
    + ``singularity.runOptions = '--nv'`` - The GPU environment will be enabled upon execution.


.. note::
    Personalize the ``queue`` settings ONLY should work well for all the SLURM-based HPC platforms.

The configuration file is ``DeepPrep/deepprep/nextflow/cluster/deepprep.slurm.gpu.config``.


Run DeepPrep with GPU
---------------------

1. Assign an *absolute path* to ``${TEST_DIR}``.

.. code-block:: none

    $ export TEST_DIR=<shared_storage_path>

2. Download the DeepPrep code.

::

    $ cd ${TEST_DIR}
    $ git clone https://github.com/pBFSLab/DeepPrep.git && cd DeepPrep && git checkout 24.1.0

3. Run DeepPrep.

Pass *absolute paths* to avoid any mistakes.

.. code-block:: none
    :emphasize-lines: 14

    export FS_LICENSE=<freesurfer_license_file>
    export BIDS_PATH=${TEST_DIR}/<bids_path>
    export OUTPUT_PATH=${TEST_DIR}/<output_path>

    ${TEST_DIR}/DeepPrep/deepprep/deepprep.sh \
    ${BIDS_PATH} \
    ${OUTPUT_PATH} \
    participant \
    --bold_task_type <task_label> \
    --deepprep_home ${TEST_DIR}/DeepPrep \
    --fs_license_file ${FS_LICENSE} \
    --executor cluster \
    --container ${TEST_DIR}/deepprep_24.1.0.sif \
    --config_file ${TEST_DIR}/DeepPrep/deepprep/nextflow/cluster/deepprep.slurm.gpu.config

**Add the following arguments to execute on clusters**

.. code-block:: none

    --executor cluster
    --container ${TEST_DIR}/deepprep_24.1.0.sif
    --config_file ${TEST_DIR}/DeepPrep/deepprep/nextflow/cluster/deepprep.slurm.gpu.config



========================================
Update the Configuration file (CPU only)
========================================

To execute DeepPrep on CPU, you can use the previously stated configuration file (on GPU) by removing the ``withLabel: with_gpu ...`` section.

Shown as below:

.. code-block:: none

    //deepprep.slurm.cpu.config

    singularity.enabled = true
    singularity.autoMounts = false
    singularity.runOptions = '-e \
        --home ${output_dir}/WorkDir/home \
        --env TEMP=${output_dir}/WorkDir/tmp \
        --env TMP=${output_dir}/WorkDir/tmp \
        --env TMPDIR=${output_dir}/WorkDir/tmp \
        -B ${bids_dir} \
        -B ${output_dir} \
        -B ${subjects_dir} \
        -B ${fs_license_file}:/opt/freesurfer/license.txt \
    '

    process {
    //errorStrategy = 'ignore'

        executor = 'slurm'

        queue = 'cpu1,cpu2,fat'

        clusterOptions = { " --chdir=${nextflow_work_dir}" }

        container = "${container}"
    }



The configuration file is ``DeepPrep/deepprep/nextflow/cluster/deepprep.slurm.cpu.config``.


Run DeepPrep with CPU only
--------------------------

To execute DeepPrep only on CPUs, add the ``--device cpu`` command and modify the ``--config_file`` to the CPU version ``deepprep.slurm.cpu.config``.

Pass *absolute paths* to avoid any mistakes.

Shown as below:

.. code-block:: none
    :emphasize-lines: 14-15

    export FS_LICENSE=<freesurfer_license_file>
    export BIDS_PATH=${TEST_DIR}/<bids_path>
    export OUTPUT_PATH=${TEST_DIR}/<output_path>

    ${TEST_DIR}/DeepPrep/deepprep/deepprep.sh \
    ${BIDS_PATH} \
    ${OUTPUT_PATH} \
    participant \
    --bold_task_type <task_label> \
    --deepprep_home ${TEST_DIR}/DeepPrep \
    --fs_license_file ${FS_LICENSE} \
    --executor cluster \
    --container ${TEST_DIR}/deepprep_24.1.0.sif \
    --config_file ${TEST_DIR}/DeepPrep/deepprep/nextflow/cluster/deepprep.slurm.cpu.config \
    --device cpu



==========================
Samples on three platforms
==========================

Get started with a ``test_sample``, `download here`_.

The BIDS formatted sample contains one subject with one anatomical image and two functional images.


SLURM
-----

**CPU Only**

.. code-block:: none

    export TEST_DIR=<test_dir>
    ${TEST_DIR}/DeepPrep/deepprep/deepprep.sh \
    ${TEST_DIR}/test_sample \
    ${TEST_DIR}/test_sample_DeepPrep_cpu \
    participant \
    --bold_task_type rest \
    --deepprep_home ${TEST_DIR}/DeepPrep \
    --fs_license_file ${TEST_DIR}/DeepPrep/license.txt \
    --executor cluster \
    --container ${TEST_DIR}/deepprep_24.1.0.sif \
    --config_file ${TEST_DIR}/DeepPrep/deepprep/nextflow/cluster/deepprep.slurm.cpu.config \
    --device cpu \
    --debug \
    --resume


.. code-block:: none

    //deepprep.slurm.cpu.config

    singularity.enabled = true
    singularity.autoMounts = false
    singularity.runOptions = '-e \
        --home ${output_dir}/WorkDir/home \
        --env TEMP=${output_dir}/WorkDir/tmp \
        --env TMP=${output_dir}/WorkDir/tmp \
        --env TMPDIR=${output_dir}/WorkDir/tmp \
        -B ${bids_dir} \
        -B ${output_dir} \
        -B ${subjects_dir} \
        -B ${fs_license_file}:/opt/freesurfer/license.txt \
    '

    process {
    //errorStrategy = 'ignore'

        executor = 'slurm'

        queue = 'cpu1,cpu2,fat'

        clusterOptions = { " --chdir=${nextflow_work_dir}" }

        container = "${container}"
    }


**With GPU**

.. code-block:: none

    export TEST_DIR=<test_dir>
    ${TEST_DIR}/DeepPrep/deepprep/deepprep.sh \
    ${TEST_DIR}/test_sample \
    ${TEST_DIR}/test_sample_DeepPrep_gpu \
    participant \
    --bold_task_type rest \
    --deepprep_home ${TEST_DIR}/DeepPrep \
    --fs_license_file ${TEST_DIR}/DeepPrep/license.txt \
    --executor cluster \
    --container ${TEST_DIR}/deepprep_24.1.0.sif \
    --config_file ${TEST_DIR}/DeepPrep/deepprep/nextflow/cluster/deepprep.slurm.gpu.config \
    --debug \
    --resume

.. code-block:: none

    //deepprep.slurm.gpu.config

    singularity.enabled = true
    singularity.autoMounts = false
    singularity.runOptions = '-e \
    --home ${output_dir}/WorkDir/home \
    --env TEMP=${output_dir}/WorkDir/tmp \
    --env TMP=${output_dir}/WorkDir/tmp \
    --env TMPDIR=${output_dir}/WorkDir/tmp \
    -B ${bids_dir} \
    -B ${output_dir} \
    -B ${subjects_dir} \
    -B ${fs_license_file}:/opt/freesurfer/license.txt \
    '

    process {
    //errorStrategy = 'ignore'

        executor = 'slurm'

        queue = 'cpu1,cpu2,fat'

        clusterOptions = { " --chdir=${nextflow_work_dir}" }

        container = "${container}"

        withLabel: with_gpu {
            queue = 'gpu2'
            clusterOptions = { " --chdir=${nextflow_work_dir} --gres=gpu:1" }
            singularity.runOptions = '-e --nv \
                --home ${output_dir}/WorkDir/home \
                --env TEMP=${output_dir}/WorkDir/tmp \
                --env TMP=${output_dir}/WorkDir/tmp \
                --env TMPDIR=${output_dir}/WorkDir/tmp \
                -B ${bids_dir} \
                -B ${output_dir} \
                -B ${subjects_dir} \
                -B ${fs_license_file}:/opt/freesurfer/license.txt \
                '
         }
    }



PBS
---

**CPU Only**

.. code-block:: none

    export TEST_DIR=<test_dir>
    ${TEST_DIR}/DeepPrep/deepprep/deepprep.sh \
    ${TEST_DIR}/test_sample \
    ${TEST_DIR}/test_sample_DeepPrep_cpu \
    participant \
    --bold_task_type rest \
    --deepprep_home ${TEST_DIR}/DeepPrep \
    --fs_license_file ${TEST_DIR}/DeepPrep/license.txt \
    --executor cluster \
    --container ${TEST_DIR}/deepprep_24.1.0.sif \
    --config_file ${TEST_DIR}/DeepPrep/deepprep/nextflow/cluster/deepprep.pbfs.cpu.config \
    --device cpu \
    --debug \
    --resume


.. code-block:: none

    // deepprep.pbs.cpu.config

    singularity.enabled = true
    singularity.autoMounts = false
    singularity.runOptions = '-e \
        --home ${output_dir}/WorkDir/home \
        --env TEMP=${output_dir}/WorkDir/tmp \
        --env TMP=${output_dir}/WorkDir/tmp \
        --env TMPDIR=${output_dir}/WorkDir/tmp \
        -B ${bids_dir} \
        -B ${output_dir} \
        -B ${subjects_dir} \
        -B ${fs_license_file}:/opt/freesurfer/license.txt \
    '

    process {
    //errorStrategy = 'ignore'

        executor = 'pbspro'

        time = '2h'

    //    queue = 'bigmem'
    //    clusterOptions = { ' ' }

        container = '${container}'
    }


**With GPU**

.. code-block:: none

    export TEST_DIR=<test_dir>
    ${TEST_DIR}/DeepPrep/deepprep/deepprep.sh \
    ${TEST_DIR}/test_sample \
    ${TEST_DIR}/test_sample_DeepPrep_gpu \
    participant \
    --bold_task_type rest \
    --deepprep_home ${TEST_DIR}/DeepPrep \
    --fs_license_file ${TEST_DIR}/DeepPrep/license.txt \
    --executor cluster \
    --container ${TEST_DIR}/deepprep_24.1.0.sif \
    --config_file ${TEST_DIR}/DeepPrep/deepprep/nextflow/cluster/deepprep.pbfs.gpu.config \
    --device auto \
    --debug \
    --resume


.. code-block:: none

    // deepprep.pbs.gpu.config

    singularity.enabled = true
    singularity.autoMounts = false
    singularity.runOptions = '-e \
        --home ${output_dir}/WorkDir/home \
        --env TEMP=${output_dir}/WorkDir/tmp \
        --env TMP=${output_dir}/WorkDir/tmp \
        --env TMPDIR=${output_dir}/WorkDir/tmp \
        -B ${bids_dir} \
        -B ${output_dir} \
        -B ${subjects_dir} \
        -B ${fs_license_file}:/opt/freesurfer/license.txt \
    '

    process {
    //errorStrategy = 'ignore'

        executor = 'pbspro'

        time = '2h'

    //    queue = 'bigmem'
    //    clusterOptions = { " " }

        container = '${container}'

        withLabel: with_gpu {
            module = 'cuda/11.8.0-gcc/9.5.0'
            queue = 'gpu_k240'
            clusterOptions = { ' -l select=1:ncpus=8:mem=20gb:ngpus=1:gpu_model=p100,walltime=00:20:0' }
            singularity.runOptions = '-e --nv \
                --home ${output_dir}/WorkDir/home \
                --env TEMP=${output_dir}/WorkDir/tmp \
                --env TMP=${output_dir}/WorkDir/tmp \
                --env TMPDIR=${output_dir}/WorkDir/tmp \
                -B ${bids_dir} \
                -B ${output_dir} \
                -B ${subjects_dir} \
                -B ${fs_license_file}:/opt/freesurfer/license.txt \
            '
        }
    }



AWS
---

*In Process*
