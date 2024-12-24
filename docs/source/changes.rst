.. include:: links.rst

----------
What's new
----------

.. _latest-note:

24.1.2 (December 23, 2024)
-------------------------

1. **Update:** BOLD preprocessing results now include CIFTI format.
2. **Update:** Dockerfile for building Docker image have been upgraded to use online resources.


24.1.1 (July 19, 2024)
-------------------------

1. **Update:** The minimum logical CPU core is 4, and the minimum RAM requirement is 12GB.
2. **Update:** Speed up, by optimizing several processes in DeepPrep to be multiprocessing.
3. **Doc:** Add step-by-step Singularity user guide in :ref:`singularity-guide`.
4. **Doc:** List all the training datasets used in Deep Learning models (FastSurfer, FastCSR, SUGAR, SynthMorph) in :ref:`DL-trainingsets`.
5. **Doc:** Add confounds descriptions in :ref:`outputs-confounds`.


24.1.0 (May 27, 2024)
-----------------------

The released ``pbfslab/deepprep:24.1.0`` improves the overall volumetric registration quality,
reduces the BOLD-to-template registration smoothness, and provides users with more confounds.
Besides, DeepPrep now offers more flexible parameter settings.
For instance, multiple task labels and participant labels can be run by specifying ``--bold_task_type`` and ``--participant_label``.
To save processing time, users can choose to skip generating results in volume space or surface space by setting ``None``
to ``--bold_volume_space`` or ``--bold_surface_spaces``.


**Major updates:**

1. SPEED UP!!!

We further reduced the processing time for a single subject from 32.5 minutes to 30.8 minutes (statstics from 10 test
subjects in the UKBB dataset).

2. Updated SynthMorph to the latest version ``freesurfer/synthmorph:2`` for volumetric registration.

SynthMorph proposed a novel end-to-end deep learning approach for volumetric registration,
which performs fast and robust joint affine-deformable registration.
Check this amazing work for more details at `Hoffmann et al.,`_.

DeepPrep leverages SynthMorph for aligning T1w-space volume with template (i.e. ``MNI152NLin6Asym``).
We adjusted the default resolution from 256 to 192, offering a more computationally efficient solution with reduced RAM and VRAM demands while maintaining accuracy.

3. Reduced spatial smoothness of BOLD-to-template volume by single-step interpolation.

The previous version of DeepPrep showed larger spatial smoothness due to multiple-step interpolations during preprocessing.
These multiple-step interpolations primarily resulted from difficulties in fusing different transformation matrices from various tools,
particularly between SynthMorph and other tools. By reconciling the transformation process and eliminating unnecessary interpolations,
DeepPrep aligns the original BOLD to the template with a single interpolation, thereby reducing spatial smoothness. For the test subjects,
the smoothness was reduced from *6.42 ± 0.24 mm* to FWHM *4.27 ± 0.15 mm*, which is comparable to fMRIPrep's smoothness of *4.06 ± 0.18 mm*.


4. Add more confounds, including:

    + 24HMP: 3 translations and 3 rotation (6HMP), their temporal derivatives (12HMP), and their quadratic terms (24HMP).
    + 12 Global signals: csf, white_matter, global_signal their temporal derivatives, and their quadratic terms.
    + Outlier detection:  framewise_displacement, rmsd, dvars, std_dvars, non_steadv_state_outlier, motion_outlier.
    + Discrete cosine-basis regressors: cosine
    + CompCor confounds:  PCA regressors, saves top 10 components for e_comp_cor and saves 50% of explained variance of the rest, i.e. anatomical CompCor (a_comp_cor), temporal CompCor (t_comp_cor), outside brainmask CompCor (e_comp_cor). CompCor estimates from WM, CSF, and their union region is a_comp_cor, CompCor estimates from each of WM, csf are w_comp_cor, and c_comp_cor. eCompCor is a complementary extension of aCompCor and tCompCor. Its noise mask is assigned as the background of the field of view. Regressing out these components improves test-retest reliability. We will publish a paper to elucidate the method and its performance.



*Of Note:*

There's a significant difference in CSF masks generated from DeepPrep and fMRIPrep pipelines,
DeepPrep uses a CSF mask derived from the ventricle mask segmented by the FastSurferCNN, consistent with FreeSurfer,
whereas fMRIPrep uses a probability map that covers the whole brain and mostly adjacent gray matter.
The main rationale behind our choice is that the segmented ventricle mask is more accurate and sufficiently representative of noise.
More importantly, it is less likely to introduce "real" brain activity from the gray matter, given its relatively large spatial distance from gray matter.

DeepPrep CSF mask sample

.. image:: https://download.anning.info/ninganme-public/DeepPrep/docs/source/24.1.1/_static/changes/dp_csf.png
   :align: center

|

fMRIPrep CSF mask sample

.. image:: https://download.anning.info/ninganme-public/DeepPrep/docs/source/24.1.1/_static/changes/fp_csf.png
   :align: center

|



**Other updates:**

1. You can now run multiple tasks with ``--bold_task_type '[task1 task2 task3 ...]'``. (#107)

2. Select multiple participants to process with ``--participant_label '[001 002 003 ...]'``.

3. Opt to skip generating outputs in either volume space with ``--bold_volume_space None`` or in surface space with ``--bold_surface_spaces None``. (#115, from @xingyu-liu)
