.. include:: links.rst

----------
What's new
----------

24.1.0 (May 27, 2024)
---------------------

**Major updates:**
1. Update SynthMorph to the latest version `freesurfer/synthmorph:2` for volume registration.
2. Apply a chain of transformations to register the original BOLD to template directly, which preserves spatial smoothing in outputs.
3. Add more confounds, including:
    + 24HMP: 3 translations and 3 rotation, and their temporal derivatives & quadratic terms
    + 12 Global signals: csf, white_matter, global_signal, and their temporal derivatives & quadratic terms
    + Outlier detection:  framewise_displacement, rmsd, dvars, std_dvars, non_steadv_state_outlier, motion_outlier
    + Discrete cosine-basis regressors: cosine
    + CompCor confounds:  PCA regressors, i.e. anatomical compcor (a_comp_cor), temporal compcor (t_comp_cor), outside brainmask compcor (e_comp_cor).

**Other updates:**
1. You can now run multiple tasks with `--bold_task_type '[task1 task2 task3 ...]'`
2. Select multiple participants to process with `--participant_label '[001 002 003 ...]'`
3. Opt to skip generating outputs in either volume space with `--bold_volume_space None` or in surface space with `--bold_surface_spaces None`