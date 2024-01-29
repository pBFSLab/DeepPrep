#! /usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

from fmriprep.workflows.fieldmap import init_single_subject_fieldmap_wf
from fmriprep.workflows.bold.base import init_bold_wf
from fmriprep import config


def get_output_space(output_spaces):
    """
    # TODO: only support T1w fsnative fsaverage
    """
    _output_spaces = ['T1w']
    for output_space in output_spaces:
        if 'fsnative' in output_space:
            _output_spaces.append(output_space)
        elif 'fsaverage' in output_space:
            _output_spaces.append(output_space)
    return _output_spaces


def update_config(bids_dir, bold_preprocess_dir, work_dir, fs_license_file, fs_subjects_dir,
                  subject_id, task_id, spaces, templateflow_home):
    config.execution.bids_dir = bids_dir
    config.execution.log_dir = f'{work_dir}/log'
    config.execution.fs_license_file = fs_license_file
    config.execution.fs_subjects_dir = fs_subjects_dir
    config.execution.output_dir = bold_preprocess_dir
    config.execution.output_spaces = spaces
    config.execution.participant_label = [ subject_id,]
    config.execution.task_id = task_id
    config.execution.templateflow_home = templateflow_home
    config.execution.work_dir = work_dir
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


def get_bold_func_path(bids_orig, bids_preproc, bold_orig_file):
    from bids import BIDSLayout
    layout_orig = BIDSLayout(bids_orig, validate=False)
    layout_preproc = BIDSLayout(bids_preproc, validate=False)
    info = layout_orig.parse_file_entities(bold_orig_file)

    boldref_t1w_info = info.copy()
    boldref_t1w_info['space'] = 'T1w'
    boldref_t1w_info['suffix'] = 'boldref'
    boldref_t1w_file = layout_preproc.get(**boldref_t1w_info)[0]

    return Path(boldref_t1w_file).parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--subjects_dir", required=False)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--bold_series", type=str, nargs='+', default=[], required=False)  # BOLD Series，not one bold file
    parser.add_argument("--bold_spaces", type=str, nargs='+', default=['individual', 'T1w', 'fsnative'], required=False)  # BOLD Series，not one bold file
    parser.add_argument("--t1w_preproc", required=False)
    parser.add_argument("--t1w_mask", required=False)
    parser.add_argument("--t1w_dseg", required=False)
    parser.add_argument("--fsnative2t1w_xfm", required=False)
    parser.add_argument("--fs_license_file", required=False)
    parser.add_argument("--bold_sdc", required=False, default='False')
    parser.add_argument("--templateflow_home", required=False, default='/home/root/.cache/templateflow')
    parser.add_argument("--qc_result_path", required=True)
    args = parser.parse_args()
    """
    if filedmap:
    --bids_dir /mnt/ngshare/temp/ds004498
    --bold_preprocess_dir /mnt/ngshare/temp/ds004498deepprep
    --subject_id sub-CIMT001
    --task_id rest
    --spaces individual T1w fsnative
    --fieldmap True
    else:
    --bids_dir /mnt/ngshare/temp/ds004498
    --subjects_dir /mnt/ngshare/temp/ds004498/Recon720
    --bold_preprocess_dir /mnt/ngshare/temp/ds004498deepprep
    --subject_id sub-CIMT001
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

    t1w_preproc = args.t1w_preproc
    t1w_mask = args.t1w_mask
    t1w_dseg = args.t1w_dseg
    t1w_tpms = []
    fsnative2t1w_xfm = args.fsnative2t1w_xfm

    print("t1w_preproc :", t1w_preproc)
    print("t1w_mask :", t1w_mask)
    print("t1w_dseg :", t1w_dseg)
    print("fsnative2t1w_xfm :", fsnative2t1w_xfm)

    bold_spaces = get_output_space(args.bold_spaces)
    spaces = ' '.join(bold_spaces)
    update_config(args.bids_dir, args.bold_preprocess_dir, args.work_dir, args.fs_license_file,
                  args.subjects_dir, args.subject_id, args.task_id, spaces,
                  args.templateflow_home)
    work_dir = Path(config.execution.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_file = work_dir / config.execution.run_uuid / 'config.toml'
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config.to_filename(config_file)
    config.load(config_file)

    output_dir = Path(config.execution.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path(args.bold_preprocess_dir) / subject_id / 'figures'
    qc_dir = Path(args.qc_result_path) / subject_id / 'figures'

    from niworkflows.utils.bids import collect_data
    from niworkflows.utils.connections import listify

    subject_data = collect_data(
        config.execution.layout,
        subject_id_split,
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

    single_subject_fieldmap_wf, estimator_map = init_single_subject_fieldmap_wf(subject_id_split, bold_runs)

    if args.bold_sdc.upper() == 'TRUE':  # run fieldmap
        if single_subject_fieldmap_wf:
            base_dir = Path(config.execution.work_dir) / f'{subject_id}_wf' / f'{args.task_id}_wf'
            base_dir.mkdir(parents=True, exist_ok=True)
            single_subject_fieldmap_wf.base_dir = base_dir
            single_subject_fieldmap_wf.run()

    else:  # run preproc
        with open(args.bold_series[0], 'r') as f:
            data = f.readlines()
        data = [i.strip() for i in data]
        bold_file = data[1]
        bold_name = os.path.basename(bold_file).split('.')[0]

        fieldmap_id = estimator_map.get(bold_file)

        from nipype.pipeline import engine as pe
        from nipype.interfaces import utility as niu
        inputnode = pe.Node(
            niu.IdentityInterface(
                fields=[
                    "subjects_dir",
                    "subject_id",
                    "t1w_preproc",
                    "t1w_mask",
                    "t1w_dseg",
                    "t1w_tpms",
                    "fsnative2t1w_xfm",
                ]
            ),
            name="inputnode",
        )
        inputnode.inputs.t1w_preproc = t1w_preproc
        inputnode.inputs.t1w_mask = t1w_mask
        inputnode.inputs.t1w_dseg = t1w_dseg
        inputnode.inputs.t1w_tpms = t1w_tpms
        inputnode.inputs.subjects_dir = config.execution.fs_subjects_dir
        inputnode.inputs.subject_id = subject_id
        inputnode.inputs.fsnative2t1w_xfm = fsnative2t1w_xfm

        bold_wf = init_bold_wf(
            bold_series=[bold_file],
            precomputed={},
            fieldmap_id=fieldmap_id,
        )
        bold_wf.name = 'bold_wf'

        from niworkflows.engine.workflows import LiterateWorkflow as Workflow
        workflow = Workflow(name=f'{bold_name}_wf')
        base_dir = Path(config.execution.work_dir) / f'{subject_id}_wf' / f'{args.task_id}_wf'

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
            # reuse fieldmap result
            fmap_preproc_wf_dir = base_dir / workflow.name / single_subject_fieldmap_wf.name
            if fmap_preproc_wf_dir.exists():
                fmap_preproc_wf_dir.unlink()
            fmap_preproc_wf_dir.parent.mkdir(parents=True, exist_ok=True)
            fmap_preproc_wf_dir.symlink_to(base_dir / single_subject_fieldmap_wf.name)

            all_nodes = single_subject_fieldmap_wf.list_node_names()
            all_graph_nodes = single_subject_fieldmap_wf._graph.nodes
            remove_nodes = []
            for node in single_subject_fieldmap_wf._graph.nodes:
                if ('fmap_reports_wf' in node.fullname) or ('fmap_derivatives_wf' in node.fullname):
                    remove_nodes.append(node)
            single_subject_fieldmap_wf.remove_nodes(remove_nodes)
            # end

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

        outputnode = pe.Node(
            niu.IdentityInterface(
                fields=[
                    "subjects_dir",
                    "subject_id",
                    "t1w_preproc",
                    "t1w_mask",
                    "t1w_dseg",
                    "t1w_tpms",
                    "fsnative2t1w_xfm",
                ]
            ),
            name="outputnode",
        )

        workflow.base_dir = base_dir
        result = workflow.run()

        # get mcflirt result file
        mcflirt_node_name = ''
        for node_name in workflow.list_node_names():
            if 'mcflirt' in node_name:
                mcflirt_node_name = node_name
                break
        mcflirt_node_path = base_dir / workflow.name / mcflirt_node_name.replace('.', '/')
        list(mcflirt_node_path.glob('*'))
        func_path = get_bold_func_path(args.bids_dir, args.bold_preprocess_dir, bold_file)
        for mcflirt_file in mcflirt_node_path.glob('*'):
            if 'mcf.nii' in mcflirt_file.name and mcflirt_file.is_file():
                shutil.copyfile(mcflirt_file, func_path / mcflirt_file.name)
        # output
        # _bold_mcf.nii_rel.rms
        # _bold_mcf.nii_abs.rms
        # _bold_mcf.nii.par
        # end

        source_files = fig_dir.glob('*')
        qc_dir.mkdir(parents=True, exist_ok=True)
        for source_file in source_files:
            shutil.move(source_file, qc_dir / source_file.name)
