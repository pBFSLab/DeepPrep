#! /usr/bin/env python3
import os
from pathlib import Path
import argparse

print(__file__)

print(os.path.abspath(os.curdir))

from fmriprep.workflows.fieldmap import init_single_subject_fieldmap_wf
from fmriprep import config
from niworkflows.utils.bids import collect_data
from niworkflows.utils.connections import listify


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="DeepPrep: Prepare input data for BOLD process"
    )

    parser.add_argument("--subject_id_unique", required=True)
    parser.add_argument("--boldfile_id_unique", required=True)
    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_path", required=True)
    parser.add_argument("--bold_task_type", required=True)
    args = parser.parse_args()

    subject_id = args.subject_id_unique.split('-')[1]
    bold_id = args.boldfile_id_unique

    config.execution.bids_dir = args.bids_dir
    # config.execution.log_dir = '/mnt/ngshare/DeepPrep_Docker/DeepPrep_workdir/ds004498deepprep/log'
    # config.execution.fmriprep_dir = '/mnt/ngshare/DeepPrep_Docker/DeepPrep_workdir/ds004498deepprep'
    # config.execution.fs_license_file = '/mnt/ngshare/DeepPrep_Docker/DeepPrep_workdir/ds004498/license.txt'
    # config.execution.fs_subjects_dir = '/mnt/ngshare/DeepPrep_Docker/DeepPrep_workdir/ds004498'
    config.execution.output_dir = args.bold_preprocess_path
    # config.execution.output_spaces = 'sbref run individual T1w fsnative'
    config.execution.participant_label = [ subject_id,]
    config.execution.task_id = args.bold_task_type
    # config.execution.templateflow_home = '/home/fmriprep/.cache/templateflow'
    # config.execution.work_dir = '/mnt/ngshare/DeepPrep_Docker/DeepPrep_workdir/ds004498deepprep/work'

    # config.workflow.anat_only = False
    # config.workflow.bold2t1w_dof = 6
    # config.workflow.bold2t1w_init = 'register'
    # config.workflow.cifti_output = False
    # config.workflow.dummy_scans = 0
    # config.workflow.fmap_bspline = False
    # config.workflow.fmap_bspline = False
    # config.workflow.hires = True
    config.workflow.ignore =[]
    # config.workflow.level = "full"
    # config.workflow.longitudinal = False
    # config.workflow.run_msmsulc = False
    # config.workflow.medial_surface_nan = False
    # config.workflow.project_goodvoxels = False
    # config.workflow.regressors_all_comps = False
    # config.workflow.regressors_dvars_th = 1.5
    # config.workflow.regressors_fd_th = 0.5
    # config.workflow.run_reconall = True
    # config.workflow.spaces = "sbref run individual"
    # config.workflow.use_aroma = False
    # config.workflow.use_bbr = True
    # config.workflow.use_syn_sdc = False
    # config.workflow.me_t2s_fit_method = "curvefit"

    work_dir = Path(config.execution.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_file = work_dir / 'fmriprep.toml'
    config.to_filename(config_file)
    config.load(config_file)

    subject_data = collect_data(
        config.execution.layout,
        subject_id,
        task=config.execution.task_id,
        echo=config.execution.echo_idx,
        bids_filters=config.execution.bids_filters,
    )[0]

    bold_runs = [
        sorted(
            listify(run),
            key=lambda file: config.execution.layout.get_metadata(file).get('EchoTime', 0),
        )
        for run in subject_data['bold']
    ]

    single_subject_fieldmap_wf, estimator_map = init_single_subject_fieldmap_wf(subject_id, bold_runs)
    if single_subject_fieldmap_wf:
        single_subject_fieldmap_wf.base_dir = os.path.join(config.execution.work_dir, f'sub_{subject_id}_wf', f'{bold_id}')
        single_subject_fieldmap_wf.run()

        fieldmap_id = estimator_map.get(bold_runs[0][0])
        print(fieldmap_id)
    else:
        fieldmap_id = None

    with open(f'{fieldmap_id}', 'w') as f:
        f.write("")

