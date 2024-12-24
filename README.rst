
Welcome to DeepPrep's documentation!
====================================


.. note::

    DeepPrep has just released the newest version **pbfslab/deepprep:24.1.2**. Check out the details in the ``What's New`` section!
    https://deepprep.readthedocs.io/en/latest/changes.html#id1


Installation
------------

For instructions on installing and using DeepPrep, please refer to the documentation page.
https://deepprep.readthedocs.io/en/latest/installation.html

About
-----

Neuroimaging has entered the era of big data. However, the advancement of preprocessing pipelines falls behind the rapid expansion of data volume, causing significant computational challenges. Here, we present DeepPrep, a pipeline empowered by deep learning and workflow manager. Evaluated on over 55,000 scans, DeepPrep demonstrates a 11-fold acceleration, exceptional scalability, and robustness compared to the current state-of-the-art pipeline, providing a promising solution to meet the scalability requirements of neuroimaging.
The following is efficient workflow of the DeepPrep pipeline:

.. image:: https://download.anning.info/ninganme-public/DeepPrep/docs/source/24.1.1/_static/readme/fig1.svg
   :align: center

|

A computationally efficient and scalable neuroimaging pipeline is empowered by deep-learning algorithms and workflow managers.

 | a) The neuroimaging pipeline leverages deep learning algorithms, including FastSurfer, FastCSR, SUGAR, and SynthMorph, to replace the most time-intensive modules present in conventional pipelines. This substitution enables the achievement of highly efficient and robust brain tissue segmentation, cortical surface reconstruction, cortical surface registration, and volumetric spatial normalization. The current version of the pipeline supports both anatomical and functional MRI preprocessing in both volumetric and cortical surface spaces.
 | b) The preprocessing pipeline is organized into multiple relatively independent yet interdependent processes. Imaging data adhering to the BIDS format are preprocessed through a structured workflow managed by Nextflow, an open-source workflow manager designed for life sciences. Nextflow efficiently schedules task processes and allocates computational resources across diverse infrastructures, encompassing local computers, HPC clusters, and cloud computing environments. The pipeline yields standard preprocessed imaging and derivative files, DeepPrep quality control reports, and runtime reports as its outputs.

Outperformance in application to large-sample and clinical datasets:

.. image:: https://download.anning.info/ninganme-public/DeepPrep/docs/source/24.1.1/_static/readme/fig2.png
   :align: center

|

DeepPrep achieves over 10-fold acceleration and shows robustness in processing clinical samples.

 | a) DeepPrep achieves 10.1 times faster than fMRIPrep in processesing a single subject sequentially on a local workstation. Error bars represent standard deviations.
 | b) DeepPrep (blue bars) processed 1146 subjects in batches, which is 10.4 times more efficient than fMRIPrep (gray bars).
 | c) In the HPC processing, fMRIPrep shows a trade-off curve between preprocessing time and hardware expense associated with CPU hours (the gray line). DeepPrep provides reliable costs and efficient processing time for individual subjects (the blue dot), with expense at least 5.8 times lower than fMRIPrep
 | d) The robustness of DeepPrep was evaluated by preprocessing 53 intractable clinical samples. DeepPrep successfully completed preprocessing in 100% of patients, with 58.5% of patients being accurately preprocessed, significantly higher than fMRIPrep’s rates of 69.8% and 30.2%, respectively.
 | e) Preprocessing errors are categorized into three types, with three representative clinical samples shown. fMRIPrep yields an inaccurate brain mask when skull stripping, failed to reconstruct cortical surfaces, and exhibited misalignment in surface parcellation in the pre- and post-central gyrus. In contrast, DeepPrep successfully and accurately processed these cases.

Upcoming improvements
---------------------
- [X] CIFTI: DeepPrep will produce outputs in CIFTI format.
- [ ] MP2RAGE: DeepPrep will be compatible with MP2RAGE.
- [ ] High-resolution images: We're expanding DeepPrep's support to include high-resolution images (i.e. 7T).
- [ ] DeepQC: DeepPrep will feature an automated quality control process called DeepQC for both sMRI and fMRI visualization.


Citation
--------
Ren, J.\*, An, N.\*, Lin, C., Zhang, Y., Sun, Z., Zhang, W., Li, S., Guo, N., Cui, W., Hu, Q. and Wang, W., 2024. DeepPrep: An accelerated, scalable, and robust pipeline for neuroimaging preprocessing empowered by deep learning. bioRxiv, pp.2024-03.

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
