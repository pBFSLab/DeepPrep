.. include:: links.rst


-------------------
Usage Notes (GUI)
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

.

.. _gui-guide:

=================
GUI User Guide
=================

.. note::

   For the sake of simplicity in usage, DeepPre has released three GUIs: Preprocessing, Postprocessing, and Quick QC.


**Welcome to the DeepPrep Graph User Interface (GUI)!**

We now provide our users with a user-friendly GUI to run:

1. Preprocessing pipelines for T1-weighted (T1w) and BOLD images.

2. The subsequent BOLD postprocessing steps include bandpass filtering, regression, and optional smoothing.

3. A quick QC report to visualize the results!

=============================
Preprocessing of T1w & BOLD
=============================

DeepPrep is a preprocessing pipeline that can flexibly handle anatomical and functional MRI data end-to-end, accommodating various sizes from a single participant to LARGE datasets.
Both the anatomical and functional parts can be run separately. However, preprocessed Recon is a mandatory prerequisite for executing the functional process.
The DeepPrep workflow takes the directory of the dataset to be processed as input, which is required to be in a valid BIDS format.


Main Interface Overview
=======================
Upon launching, you will see the main interface.

Here’s a breakdown of key components:

    + ``select a process`` - This section allows users to decide which process to execute, either anatomical preprocessing pipeline, or functional preprocessing pipeline, or both. Selecting ``All`` to run the entire DeepPrep pipeline (both anatomical and functional). Selecting ``T1w only`` to run the anatomical part. To run functional part only by selecting ``BOLD only``, but make sure you have the preprocessed recon data ready, and input the path in the ``Recon Result Path`` section.
    + ``select a device`` - Indicating the device you would like to use by selecting ``GPU`` or ``CPU``. The default is ``auto``, which will automatically selects a device to execute.
    + ``BIDS Path`` - The path to the input dataset, which is required to be in a standard BIDS format.
    + ``output Path`` - The path to save the preprocessed results, which will be saved in a standard BIDS format.
    + ``Recon Result Path (optional)`` - This section is only mandatory if user selects ``BOLD only`` in the ``select a process`` section. Otherwise, it is okay to leave this space blank, and the default path to store the results of anatomical data is ``<output_dir>/Recon``.
    + ``FreeSurfer license file path`` - We highly recommend users to replace the default FreeSurfer license file with yours, which can be downloaded for free from https://surfer.nmr.mgh.harvard.edu/registration.html
    + ``BOLD task type`` - The task type of the BOLD data. If there are multiple types (i.e. rest and motor), quote all the types and separate them with spaces like ``'rest motor'``.
    + ``CIFTI format`` - Selecting this section to have outputs in cifti format.
    + ``select the surface spaces: (optional)`` - Selecting all the surface spaces you would like to have, the default is ``fsaverage6``, other available options are ``fsnative``, ``fsaverage3`` to ``fsaverage5``.
    + ``select a normalized volume space: (optional)`` - Selecting a standard volume space, the default is ``MNI152NLin6Asym``.
    + ``skip n frames of BOLD data`` - The number of frames you would like to skip in BOLD data, the default is ``2``.


After you having filling up this page, a docker command will automatically generated at the end of this page. Click ``Run`` to execute this command.

**DO NOT LEAVE THIS PAGE UNTIL THE PROCESS IS FINISHED!**

======================
Postprocessing of BOLD
======================
At present, this program is designed to process resting-state functional magnetic resonance imaging (rs-fMRI) data. The processed data can be utilized for calculating functional connectivity (FC) maps, individualized brain functional parcellation, or other relevant analyses.

For task-based functional magnetic resonance imaging (task-fMRI) data, it is recommended to employ alternative tools for subsequent analysis, such as Statistical Parametric Mapping (SPM).

**processing steps:**

Surface space: bandpass -> regression -> smooth (optional)

Volume space:  bandpass -> regression -> smooth (optional)


Main Interface Overview
=======================
Upon launching, you will see the main interface.

Here’s a breakdown of key components:

    + ``Preprocessing Result Path`` - The path to the preprocessed files, which are saved in a standard BIDS format by default.
    + ``Output Path`` - The path to save the postprocessed results, which will be saved in a standard BIDS format.
    + ``FreeSurfer license file path`` - We highly recommend users to replace the default FreeSurfer license file with yours, which can be downloaded for free from https://surfer.nmr.mgh.harvard.edu/registration.html
    + ``Confounds File Path`` - A text file that lists all the confounds names, which will be used in regression. Feel free to replace this with your own confounds lists.
    + ``BOLD task type`` - The task type of the BOLD data. If there are multiple types (i.e. rest and motor), quote all the types and separate them with spaces like ``'rest motor'``.
    + ``skip n frames of BOLD data`` - The number of frames you would like to skip in BOLD data, the default is ``2``.
    + ``fwhm`` - The smooth kernel in mm.
    + ``bandpass filter`` - The default range is ``0.01-0.08``.

After you having filling up this page, a docker command will automatically generated at the end of this page. Click ``Run`` to execute this command.

**DO NOT LEAVE THIS PAGE UNTIL THE PROCESS IS FINISHED!**

========
Quick QC
========
This page allows you to quickly perform quality control (QC) on your BOLD data.
Input the path first, and then click the ``Run`` button. Once the process is complete, click ``Show`` to view the results.
More QC functions will be online, stay tuned!

Main Interface Overview
=======================
Upon launching, you will see the main interface.

Here’s a breakdown of key components:

    + ``BIDS Path`` - The path to the input dataset, which is required to be in a standard BIDS format.
    + ``Output Path`` - The path to save the quick QC results.
    + ``FreeSurfer license file path`` - We highly recommend users to replace the default FreeSurfer license file with yours, which can be downloaded for free from https://surfer.nmr.mgh.harvard.edu/registration.html

After you having filling up this page, a docker command will automatically generated at the end of this page. Click ``Run`` to execute this command. Once the process is finished, users can click the ``Show`` button to view the results.

**DO NOT LEAVE THIS PAGE UNTIL THE PROCESS IS FINISHED!**

.. container:: congratulation

   **Congratulations! You are all set!**