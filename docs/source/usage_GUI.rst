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


Preprocessing of T1w & BOLD
=============================

DeepPrep is a preprocessing pipeline that can flexibly handle anatomical and functional MRI data end-to-end, accommodating various sizes from a single participant to LARGE datasets.
Both the anatomical and functional parts can be run separately. However, preprocessed Recon is a mandatory prerequisite for executing the functional process.
The DeepPrep workflow takes the directory of the dataset to be processed as input, which is required to be in a valid BIDS format.

Experiment with it: `Preprocessing GUI`_

Postprocessing of BOLD
======================
At present, this program is designed to process resting-state functional magnetic resonance imaging (rs-fMRI) data. The processed data can be utilized for calculating functional connectivity (FC) maps, individualized brain functional parcellation, or other relevant analyses.

For task-based functional magnetic resonance imaging (task-fMRI) data, it is recommended to employ alternative tools for subsequent analysis, such as Statistical Parametric Mapping (SPM).

**processing steps:**

Surface space: bandpass -> regression -> smooth (optional)

Volume space:  bandpass -> regression -> smooth (optional)

Experiment with it: `Postprocessing GUI`_

Quick QC
======================
This page allows you to quickly perform quality control (QC) on your BOLD data.
Input the path first, and then click the ``Run`` button. Once the process is complete, click ``Show`` to view the results.
More QC functions will be online, stay tuned!

Experiment with it: `QuickQC GUI`_



.. container:: congratulation

   **Congratulations! You are all set!**