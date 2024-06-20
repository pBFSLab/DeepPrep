#! /usr/bin/env python3
import os
import argparse
import bids
import shutil
from pathlib import Path
from nipype import Node
from itertools import chain
from reports.reports_node import SubjectSummary, TemplateDimensions, AboutSummary


def get_t1w_and_bold(bids_dir, subject_ids, bold_task_type):
    layout = bids.BIDSLayout(bids_dir, derivatives=False, index_metadata=True)
    t1w_files = []
    bold_files = []

    for t1w_file in layout.get(return_type='filename', subject=subject_ids.split('-')[1], suffix="T1w", extension='.nii.gz'):
        t1w_files.append(t1w_file)

    if bold_task_type is not None:
        for bold_file in layout.get(return_type='filename', subject=subject_ids.split('-')[1], task=bold_task_type, suffix='bold', extension='.nii.gz'):
            bold_files.append(bold_file)
    return t1w_files, bold_files


def get_t1w(bids_dir, subject_id):
    layout = bids.BIDSLayout(bids_dir, derivatives=False)
    t1w_files = []

    for t1w_file in layout.get(return_type='filename', subject=subject_id.split('-')[1], suffix="T1w", extension='.nii.gz'):
        t1w_files.append(t1w_file)

    return t1w_files


def is_deepprep_recon(subjects_dir, subject_id):
    deepprep_version_file = Path(subjects_dir) / subject_id / 'scripts' / 'deepprep_version.txt'
    return deepprep_version_file.exists()


def SubjectSummary_run(subject_id, t1w_files, bold_files, subjects_dir, qc_report_path, std_spaces, nstd_spaces):

    if is_deepprep_recon(subjects_dir, subject_id):
        freesurfer_status = 'Run by DeepPrep'
    else:
        freesurfer_status = 'Pre-existing directory'

    node_name = 'SubjectSummary_run_node'
    Reports_node = Node(SubjectSummary(), node_name)

    Reports_node.interface.freesurfer_status = freesurfer_status
    Reports_node.inputs.t1w = t1w_files
    Reports_node.inputs.t2w = []
    Reports_node.inputs.subjects_dir = subjects_dir
    Reports_node.inputs.subject_id = subject_id
    Reports_node.inputs.bold = bold_files
    Reports_node.inputs.std_spaces = std_spaces
    Reports_node.inputs.nstd_spaces = nstd_spaces

    Reports_node.base_dir = Path().cwd() / subject_id
    Reports_node.run()

    subj_QC_figures_dir = Path(qc_report_path) / subject_id / 'figures'
    subj_QC_figures_dir.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(Path().cwd() / subject_id / node_name / 'report.html',
                    subj_QC_figures_dir / f'{subject_id}_desc-subjectsummary_report.html')


def TemplateDimensions_run(subject_id, t1w_files, qc_report_path, sub_workdir):
    # 这个步骤需要每个subject执行一次，将T1w作为输入

    node_name = f'{subject_id}_T1w_Reports_run_node'
    TemplateDimensions_node = Node(TemplateDimensions(), node_name)
    TemplateDimensions_node.inputs.t1w_list = t1w_files

    TemplateDimensions_node.base_dir = Path(sub_workdir)
    TemplateDimensions_node.run()
    report_path = Path(sub_workdir) / node_name / 'report.html'
    print(report_path)
    sub_qc_report_path = Path(qc_report_path) / subject_id / 'figures'
    sub_qc_report_path.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(Path(sub_workdir) / node_name / 'report.html',
                    Path(sub_qc_report_path) / f'{subject_id}_desc-templatedimensions_report.html')


def copy_config_and_get_command(qc_result_dir: Path, nextflow_log: Path):
    cmd_index = 'DEBUG nextflow.cli.Launcher - $> '
    config_index = 'User config file: '
    command = ''
    if nextflow_log.exists():
        with open(nextflow_log, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if cmd_index in line:
                command = line.strip().split(cmd_index)[1]
                command_file = qc_result_dir / 'nextflow.run.command'
                with open(command_file, 'w') as f:
                    f.write(command)
            elif config_index in line:
                config_file = line.strip().split(config_index)[1]
                shutil.copyfile(config_file, qc_result_dir / 'nextflow.run.config')
    return command


def AboutSummary_run(subject_id, qc_report_path, command, version):

    node_name = 'AboutSummary_Reports_run_node'
    AboutSummary_node = Node(AboutSummary(), node_name)

    AboutSummary_node.inputs.version = version
    AboutSummary_node.inputs.command = command

    AboutSummary_node.base_dir = Path().cwd()
    AboutSummary_node.run()
    shutil.copyfile(Path(node_name) / 'report.html',
                    Path(qc_report_path) / subject_id / 'figures' / f'{subject_id}_desc-aboutsummary_report.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- create summary reports"
    )

    parser.add_argument("--bids_dir", required=True, help="directory of BIDS type: /mnt/ngshare2/BIDS/MSC")
    parser.add_argument("--subjects_dir", required=True, help="directory of Recon results")
    parser.add_argument('--subject_id', type=str, required=True, help='subject_id')
    parser.add_argument("--task_type", type=str, nargs='+', default=[],  help="rest or etc..")
    parser.add_argument("--template_space", required=True, type=str, help="bold volume space")
    parser.add_argument("--qc_result_path", help="save qc report path", required=True)
    parser.add_argument("--deepprep_version", help="DeepPrep version", required=True)
    parser.add_argument("--nextflow_log", help="nextflow run log", required=True)
    parser.add_argument("--workdir", help="work dir", required=True)
    args = parser.parse_args()

    qc_result_path = Path(args.qc_result_path)
    if args.task_type != []:
        std_spaces = [str(args.template_space)]
        nstd_spaces = ["T1w", "fs_native"]
        t1w_files, bold_files = get_t1w_and_bold(args.bids_dir, args.subject_id, args.task_type)
        SubjectSummary_run(args.subject_id, t1w_files, bold_files, args.subjects_dir, qc_result_path, std_spaces,
                           nstd_spaces)
    else:
        t1w_files = get_t1w(args.bids_dir, args.subject_id)
    command = copy_config_and_get_command(qc_result_path, Path(args.nextflow_log))
    if len(t1w_files) > 0:
        sub_workdir = Path(args.workdir) / args.subject_id
        TemplateDimensions_run(args.subject_id, t1w_files, qc_result_path, sub_workdir)
    AboutSummary_run(args.subject_id, qc_result_path, command, args.deepprep_version)