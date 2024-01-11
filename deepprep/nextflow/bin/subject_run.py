import os
from pathlib import Path

print(__file__)

print(os.path.abspath(os.curdir))

from fmriprep.workflows.base import init_single_subject_wf
from fmriprep import config

config.execution.bids_dir = '/mnt/ngshare/temp/ds004498'
config.execution.log_dir = '/mnt/ngshare/temp/ds004498dp/log'
config.execution.fmriprep_dir = '/mnt/ngshare/temp/ds004498dp1'
config.execution.fs_license_file = '/mnt/ngshare/temp/ds004498/license.txt'
config.execution.fs_subjects_dir = '/mnt/ngshare/temp/ds004498/Recon720'
config.execution.output_dir = '/mnt/ngshare/temp/ds004498dp1'
config.execution.output_spaces = 'sbref run individual T1w fsnative'
config.execution.participant_label = [ "CIMT001",]
config.execution.task_id = 'rest'
config.execution.templateflow_home = '/home/fmriprep/.cache/templateflow'
config.execution.work_dir = '/mnt/ngshare/temp/ds004498dp1/work'

config.workflow.anat_only = False
config.workflow.bold2t1w_dof = 6
config.workflow.bold2t1w_init = 'register'
config.workflow.cifti_output = False
config.workflow.dummy_scans = 0
config.workflow.fmap_bspline = False
config.workflow.fmap_bspline = False
config.workflow.hires = True
config.workflow.ignore =[]
config.workflow.level = "full"
config.workflow.longitudinal = False
config.workflow.run_msmsulc = False
config.workflow.medial_surface_nan = False
config.workflow.project_goodvoxels = False
config.workflow.regressors_all_comps = False
config.workflow.regressors_dvars_th = 1.5
config.workflow.regressors_fd_th = 0.5
config.workflow.run_reconall = True
config.workflow.spaces = "sbref run individual"
config.workflow.use_aroma = False
config.workflow.use_bbr = True
config.workflow.use_syn_sdc = False
config.workflow.me_t2s_fit_method = "curvefit"

work_dir = Path(config.execution.work_dir)
work_dir.mkdir(parents=True, exist_ok=True)
config_file = work_dir / 'fmriprep.toml'
config.to_filename(config_file)
config.load(config_file)

# t1w_preproc = '/mnt/ngshare/temp/ds004498fp/sub-CIMT001/ses-38659/anat/sub-CIMT001_ses-38659_run-01_desc-preproc_T1w.nii.gz'
# t1w_mask = '/mnt/ngshare/temp/ds004498fp/sub-CIMT001/ses-38659/anat/sub-CIMT001_ses-38659_run-01_desc-brain_mask.nii.gz'
# t1w_dseg = '/mnt/ngshare/temp/ds004498fp/sub-CIMT001/ses-38659/anat/sub-CIMT001_ses-38659_run-01_dseg.nii.gz'
# t1w_tpms = []
# fsnative2t1w_xfm = '/mnt/ngshare/temp/ds004498fp/sub-CIMT001/ses-38659/anat/sub-CIMT001_ses-38659_run-01_from-fsnative_to-T1w_mode-image_xfm.txt'

t1w_preproc = '/mnt/ngshare/temp/ds004498/Recon720/T1.nii.gz'
t1w_mask = '/mnt/ngshare/temp/ds004498/Recon720/mask.nii.gz'
t1w_dseg = '/mnt/ngshare/temp/ds004498/Recon720/wm_dseg.nii.gz'
t1w_tpms = []
fsnative2t1w_xfm = '/mnt/ngshare/temp/ds004498/Recon720/xfm.txt'

subjects_dir = '/mnt/ngshare/temp/ds004498fp/Recon'
subject_id = 'sub-CIMT001'

single_subject_wf = init_single_subject_wf(subject_id.split('-')[1])
single_subject_wf.inputs.inputnode.t1w_preproc = t1w_preproc
single_subject_wf.inputs.inputnode.t1w_mask = t1w_mask
single_subject_wf.inputs.inputnode.t1w_dseg = t1w_dseg
single_subject_wf.inputs.inputnode.t1w_tpms = t1w_tpms
single_subject_wf.inputs.inputnode.subjects_dir = subjects_dir
single_subject_wf.inputs.inputnode.subject_id = subject_id
single_subject_wf.inputs.inputnode.fsnative2t1w_xfm = fsnative2t1w_xfm

single_subject_wf.base_dir = config.execution.work_dir
single_subject_wf.run()
