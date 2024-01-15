import argparse
import os
from pathlib import Path

from fmriprep.workflows.fieldmap import init_single_subject_fieldmap_wf
from fmriprep.workflows.bold.base import init_bold_wf
from fmriprep import config


def update_config(bids_dir, bold_preprocess_dir, fs_license_file, fs_subjects_dir,
                  subject_id, task_id, spaces, templateflow_home):
    config.execution.bids_dir = bids_dir
    config.execution.log_dir = f'{bold_preprocess_dir}/tmp/log'
    config.execution.fs_license_file = fs_license_file
    config.execution.fs_subjects_dir = fs_subjects_dir
    config.execution.output_dir = bold_preprocess_dir
    config.execution.output_spaces = spaces
    config.execution.participant_label = [ subject_id,]
    config.execution.task_id = task_id
    config.execution.templateflow_home = templateflow_home
    config.execution.work_dir = f'{bold_preprocess_dir}/tmp'
    config.execution.fmriprep_dir = config.execution.output_dir

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--subjects_dir", required=False)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--bold_series", type=str, nargs='+', default=[], required=True)  # BOLD Series，not one bold file
    parser.add_argument("--spaces", type=str, nargs='+', default=['individual', 'T1w', 'fsnative'], required=False)  # BOLD Series，not one bold file
    parser.add_argument("--t1w_preproc", required=False)
    parser.add_argument("--t1w_mask", required=False)
    parser.add_argument("--t1w_dseg", required=False)
    parser.add_argument("--fsnative2t1w_xfm", required=False)
    parser.add_argument("--fs_license_file", required=False)
    parser.add_argument("--fieldmap", required=False, default='False')
    parser.add_argument("--templateflow_home", required=False, default='/home/root/.cache/templateflow')
    args = parser.parse_args()
    """
    if filedmap:
    --bids_dir /mnt/ngshare/temp/ds004498
    --bold_preprocess_dir /mnt/ngshare/temp/ds004498deepprep
    --subject_id sub-CIMT001
    --bold_id sub-CIMT001_ses-38659_task-rest_run-01
    --task_id rest
    --bold_series /mnt/ngshare/temp/ds004498/sub-CIMT001/ses-38659/func/sub-CIMT001_ses-38659_task-rest_run-01_bold.nii.gz
    --spaces individual T1w fsnative
    --fieldmap True
    else:
    --bids_dir /mnt/ngshare/temp/ds004498
    --subjects_dir /mnt/ngshare/temp/ds004498/Recon720
    --bold_preprocess_dir /mnt/ngshare/temp/ds004498deepprep
    --subject_id sub-CIMT001
    --bold_id sub-CIMT001_ses-38659_task-rest_run-01
    --task_id rest
    --bold_series /mnt/ngshare/temp/ds004498/sub-CIMT001/ses-38659/func/sub-CIMT001_ses-38659_task-rest_run-01_bold.nii.gz
    --spaces individual T1w fsnative
    --t1w_preproc /mnt/ngshare/temp/ds004498/Recon720/T1.nii.gz
    --t1w_mask /mnt/ngshare/temp/ds004498/Recon720/mask.nii.gz
    --t1w_dseg /mnt/ngshare/temp/ds004498/Recon720/wm_dseg.nii.gz
    --fsnative2t1w_xfm /mnt/ngshare/temp/ds004498/Recon720/xfm.txt
    --fs_license_file /mnt/ngshare/temp/ds004498/license.txt
    """
    subject_id = args.subject_id
    subject_id_split = subject_id.split('-')[1]
    bold_id = args.bold_id

    t1w_preproc = args.t1w_preproc
    t1w_mask = args.t1w_mask
    t1w_dseg = args.t1w_dseg
    t1w_tpms = []
    fsnative2t1w_xfm = args.fsnative2t1w_xfm

    spaces = ' '.join(args.spaces)
    update_config(args.bids_dir, args.bold_preprocess_dir, args.fs_license_file,
                  args.subjects_dir, args.subject_id, args.task_id, spaces,
                  args.templateflow_home)
    work_dir = Path(config.execution.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_file = work_dir / 'config.toml'
    config.to_filename(config_file)
    config.load(config_file)

    bold_runs = [args.bold_series]
    single_subject_fieldmap_wf, estimator_map = init_single_subject_fieldmap_wf(subject_id_split, bold_runs)

    if args.fieldmap.upper() == 'TRUE':  # run fieldmap
        if single_subject_fieldmap_wf:
            single_subject_fieldmap_wf.base_dir = os.path.join(config.execution.work_dir, f'{subject_id}_wf', f'{bold_id}_wf')
            single_subject_fieldmap_wf.run()
    else:  # run preproc
        fieldmap_id = estimator_map.get(bold_runs[0][0])

        from nipype.pipeline import engine as pe
        from nipype.interfaces import utility as niu
        inputnode = pe.Node(
            niu.IdentityInterface(
                fields=[
                    "subjects_dir",
                    "subject_id",
                    "t1w_preproc",
                    "t1w_mask",
                    "t1w_tpms",
                    "fmap_mask",
                    "fsnative2t1w_xfm",
                ]
            ),
            name="inputnode",
        )
        inputnode.t1w_preproc = t1w_preproc
        inputnode.t1w_mask = t1w_mask
        inputnode.t1w_dseg = t1w_dseg
        inputnode.t1w_tpms = t1w_tpms
        inputnode.subjects_dir = config.execution.fs_subjects_dir
        inputnode.subject_id = subject_id
        inputnode.fsnative2t1w_xfm = fsnative2t1w_xfm

        bold_wf = init_bold_wf(
            bold_series=bold_runs[0],
            precomputed={},
            fieldmap_id=fieldmap_id,
        )
        bold_wf.name = 'bold_wf'

        from niworkflows.engine.workflows import LiterateWorkflow as Workflow
        workflow = Workflow(name=f'{bold_id}_wf')
        workflow.base_dir = os.path.join(config.execution.work_dir, f'{subject_id}_wf')

        workflow.connect([
            (inputnode, bold_wf, [
                ("t1w_preproc", "inputnode.t1w_preproc"),
                ("t1w_mask", "inputnode.t1w_mask"),
                ("t1w_dseg", "inputnode.t1w_dseg"),
                ("t1w_tpms", "inputnode.t1w_tpms"),
                ("subjects_dir", "inputnode.subjects_dir"),
                ("subject_id", "inputnode.subject_id"),
                ("fsnative2t1w_xfm", "inputnode.fsnative2t1w_xfm"),
            ]),
        ])  # fmt:skip

        if fieldmap_id:
            workflow.connect([
                (single_subject_fieldmap_wf, bold_wf, [
                    ("outputnode.fmap", "inputnode.fmap"),
                    ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                    ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                    ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                    ("outputnode.fmap_id", "inputnode.fmap_id"),
                    ("outputnode.method", "inputnode.sdc_method"),
                ]),
            ])  # fmt:skip

        workflow.run()
