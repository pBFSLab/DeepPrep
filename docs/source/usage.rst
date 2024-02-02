-----------
Usage Notes
-----------

=============================
Execution and the BIDS format
=============================

The DeepPrep workflow takes as principal input the path of the dataset that is to be processed.
The input dataset is required to be in valid BIDS format, it is possible to include only a T1w
image to run only the anat part,or just include a BOLD image and only run the func part
(the complete Recon needs to be specified).We highly recommend that you validate your dataset
with the free, online `BIDS Validator`_.

.. _BIDS Validator: http://bids-standard.github.io/bids-validator/

Further information about BIDS and BIDS-Apps can be found at the `NiPreps portal`_.

.. _NiPreps portal: https://www.nipreps.org/apps/framework/


======================
Command-Line Arguments
======================

Deepprep: Deep PREProcessing workflows v0.0.14::

   usage: deepprep [--bids_dir] [--subjects_dir] [--output_dir] [--subjects]
                    [--anat_only] [--bold_only] [--bold_task_type]
                    [--bold_skip_frame] [--bold_sdc] [--bold_confounds]
                   [--bold_spaces] [--bold_template_space] [--bold_template_resolution]

====================
Positional Arguments
====================

**bids_dir**
   The root folder of a BIDS valid dataset
**subjects_dir**
    The output path for the outcomes of anatomical preprocessing
**output_dir**
    The output path for the outcomes of function preprocessing
**subjects**
    Run with specified subjects
**anat_only**
    Run anatomical workflows only
**bold_only**
    Run BOLD workflows only
**bold_task_type**
    Select a specific task to be processed
**bold_skip_frame**
    Number of skipped frames
**bold_sdc**
    Whether to perform step sdc
**bold_confounds**
    None
**bold_spaces**
    None
**bold_template_space**
    None
**bold_template_resolution**
    None

======================
The FreeSurfer license
======================
DeepPrep uses FreeSurfer tools, which require a license to run.

To obtain a FreeSurfer license, simply register for free at
https://surfer.nmr.mgh.harvard.edu/registration.html.

When using manually-prepared environments or singularity, FreeSurfer will search for a license key
file first using the  ``$FS_LICENSE`` environment variable and then in the default path to the license key
file ``($FREESURFER_HOME/license.txt)``.

It is possible to run the docker container pointing the image to a local path where a valid license file
is stored. For example, if the license is stored in the ``$HOME/.licenses/freesurfer/license.txt`` file on
the host system, ``$FREESURFER_LICENSE=$HOME/.licenses/freesurfer/license.txt``: ::

    $ docker run -it -v ${FREESURFER_LICENSE}:/usr/local/freesurfer/license.txt
