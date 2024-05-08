
Welcome to DeepPrep's documentation!
====================================

Installation
------------

For instructions on installing and using DeepPrep, please refer to the documentation page.
https://deepprep.readthedocs.io/en/latest/installation.html

About
-----

Neuroimaging has entered the era of big data. However, the advancement of preprocessing pipelines falls behind the rapid expansion of data volume, causing significant computational challenges. Here, we present DeepPrep, a pipeline empowered by deep learning and workflow manager. Evaluated on over 55,000 scans, DeepPrep demonstrates a 11-fold acceleration, exceptional scalability, and robustness compared to the current state-of-the-art pipeline, providing a promising solution to meet the scalability requirements of neuroimaging.
The following is efficient workflow of the DeepPrep pipeline:

.. image:: https://github.com/pBFSLab/DeepPrep/raw/40446b5408cdf01cad7dcf44d092cda2130ffd69/docs/source/_static/fig1.svg
   :align: center

|

A computationally efficient and scalable neuroimaging pipeline is empowered by deep-learning algorithms and workflow managers.

 | a) The neuroimaging pipeline leverages deep learning algorithms, including FastSurfer, FastCSR, SUGAR, and SynthMorph, to replace the most time-intensive modules present in conventional pipelines. This substitution enables the achievement of highly efficient and robust brain tissue segmentation, cortical surface reconstruction, cortical surface registration, and volumetric spatial normalization. The current version of the pipeline supports both anatomical and functional MRI preprocessing in both volumetric and cortical surface spaces.
 | b) The preprocessing pipeline is organized into multiple relatively independent yet interdependent processes. Imaging data adhering to the BIDS format are preprocessed through a structured workflow managed by Nextflow, an open-source workflow manager designed for life sciences. Nextflow efficiently schedules task processes and allocates computational resources across diverse infrastructures, encompassing local computers, HPC clusters, and cloud computing environments. The pipeline yields standard preprocessed imaging and derivative files, DeepPrep quality control reports, and runtime reports as its outputs.

Outperformance in application to large-sample and clinical datasets:

.. image:: https://github.com/pBFSLab/DeepPrep/raw/40446b5408cdf01cad7dcf44d092cda2130ffd69/docs/source/_static/fig2.svg
   :align: center

|

DeepPrep achieves over 10-fold acceleration and shows robustness in processing clinical samples.

 | a) DeepPrep successfully processed 1189 participants’ scans from the UKB dataset on a workstation over one week, a remarkable 11-fold increase compared to fMRIPrep’s processing of 107 participants.
 | b) Average processing time per participant is 8.48 minutes for DeepPrep and 92.60 minutes for fMRIprep.
 | c) In the HPC context, preprocessing time of fMRIPrep can be reduced by allocating additional computational resources, yet with higher costs associated with CPU hours. DeepPrep offers flexibility in resource allocation, tailoring computational resources to the specific requirements of each task process, resulting in reliable costs and efficient processing time for individual participants. The cost of DeepPrep is at least 20 times lower than that of fMRIprep.
 | d) Robustness of DeepPrep is assessed in preprocessing clinical samples from Stroke, Tumor, and DoC datasets, who failed to be processed by FreeSurfer. DeepPrep successfully completed preprocessing in 100% of patients, with 75.5% of patients being correctly preprocessed. Meanwhile, fMRIprep’s success rate was 69.8% for completion and 50.9% for correct preprocessing.
 | e) Preprocessing errors are categorized into four types, including brain tissue segmentation, cortical surface reconstruction, cortical surface registration, and volumetric spatial normalization. Four representative cases with preprocessing errors are presented, illustrating obvious brain lesions or imaging noises in the original images. fMRIPrep yields an inaccurate brain mask when skull stripping, failed to reconstruct cortical surfaces, exhibited misalignment in surface parcellation in the pre- and post-central gyrus, and produced inappropriate volumetric normalization in perilesional region. In contrast, DeepPrep successfully and accurately processed these cases.

Upcoming improvements
---------------------
1. Cifti: DeepPrep will now produce outputs in cifti format.
2. MP2RAGE: DeepPrep will be compatible with MP2RAGE.
3. High-resolution images: We're expanding DeepPrep's support to include high-resolution images (i.e. 7T).
4. DeepQC: DeepPrep will feature an automated quality control process called DeepQC for both sMRI and fMRI visualization.


Citation
--------

License
--------

   Copyright 2023 The DeepPrep Developers

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
