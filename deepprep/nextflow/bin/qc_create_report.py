#! /usr/bin/env python3
import os
import argparse
from reports.core import run_reports
from uuid import uuid4
from pathlib import Path
from time import strftime
from reports.reports_node import SubjectSummary, TemplateDimensions, AboutSummary
from nipype import Node
import shutil
import bids
import json


def is_deepprep_recon(subjects_dir, subject_id):
    deepprep_version_file = Path(subjects_dir) / subject_id / 'scripts' / 'deepprep_version.txt'
    return deepprep_version_file.exists()


def get_t1w_and_bold(bids_dir, subject_id, bold_task_type):
    layout = bids.BIDSLayout(bids_dir, derivatives=False)
    t1w_files = []
    bold_files = []

    for t1w_file in layout.get(return_type='filename', subject=subject_id.split('-')[1], suffix="T1w", extension='.nii.gz'):
        t1w_files.append(t1w_file)

    if bold_task_type is not None:
        for bold_file in layout.get(return_type='filename', subject=subject_id.split('-')[1], task=bold_task_type, suffix='bold', extension='.nii.gz'):
            bold_files.append(bold_file)
    return t1w_files, bold_files


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

    Reports_node.base_dir = Path().cwd()
    Reports_node.run()
    shutil.copyfile(Path(node_name) / 'report.html',
                    Path(qc_report_path) / subject_id / 'figures' / f'{subject_id}_desc-subjectsummary_report.html')



def TemplateDimensions_run(subject_id, t1w_files, qc_report_path):
    # 这个步骤需要每个subject执行一次，将T1w作为输入

    node_name = 'T1w_Reports_run_node'
    TemplateDimensions_node = Node(TemplateDimensions(), node_name)
    TemplateDimensions_node.inputs.t1w_list = t1w_files

    TemplateDimensions_node.base_dir = Path().cwd()
    TemplateDimensions_node.run()
    shutil.copyfile(Path(node_name) / 'report.html',
                    Path(qc_report_path) / subject_id / 'figures' / f'{subject_id}_desc-templatedimensions_report.html')


def AboutSummary_run(subject_id, command, version):

    node_name = 'AboutSummary_Reports_run_node'
    AboutSummary_node = Node(AboutSummary(), node_name)

    AboutSummary_node.inputs.version = version
    AboutSummary_node.inputs.command = command

    AboutSummary_node.base_dir = Path().cwd()
    AboutSummary_node.run()
    shutil.copyfile(Path(node_name) / 'report.html',
                    Path(qc_report_path) / subject_id / 'figures' / f'{subject_id}_desc-aboutsummary_report.html')


def create_report(subj_qc_report_path, qc_report_path, subject_id, reports_utils_path):
    log_dir = subj_qc_report_path / 'logs'
    reports_spec = reports_utils_path / 'reports-spec-deepprep.yml'
    boilerplate_dir = reports_utils_path / 'logs'
    run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid4()}"
    if not log_dir.exists():
        cmd = f'cp -r {boilerplate_dir} {log_dir}'
        os.system(cmd)
    run_reports(subj_qc_report_path, subject_id, run_uuid,
                config=reports_spec, packagename='deepprep',
                reportlets_dir=qc_report_path)


def copy_config_and_get_command(qc_result_dir: Path, nextflow_log: Path):
    cmd_index = 'DEBUG nextflow.cli.Launcher - $> '
    config_index = 'User config file: '
    with open(nextflow_log, 'r') as f:
        lines = f.readlines()
    command = ''
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: create qc report"
    )
    parser.add_argument("--reports_utils_path", required=True)
    parser.add_argument("--bids_dir", help="BIDS dir path", required=True)
    parser.add_argument("--subjects_dir", help="SUBJECTS_DIR path", required=True)
    parser.add_argument("--qc_result_path", help="save qc report path", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_task_type", help="save qc report path", default=None)
    parser.add_argument("--deepprep_version", help="DeepPrep version", required=True)
    parser.add_argument("--nextflow_log", help="nextflow run log", required=True)
    args = parser.parse_args()

    # cur_path = os.getcwd()  # 必须添加这一行，否则生成的html会有图片重复
    # dataset_path = Path(cur_path) / os.path.basename(args.qc_result_path)
    qc_report_path = Path(args.qc_result_path)
    subj_qc_report_path = qc_report_path / args.subject_id
    subj_qc_report_path.mkdir(parents=True, exist_ok=True)
    reports_utils_path = Path(args.reports_utils_path)

    t1w_files, bold_files = get_t1w_and_bold(args.bids_dir, args.subject_id, args.bold_task_type)

    std_spaces = ["MNI152_T1_2mm"]
    nstd_spaces = ["reorient", "mc", "T1w_2mm"]
    command = copy_config_and_get_command(qc_report_path, Path(args.nextflow_log))

    SubjectSummary_run(args.subject_id, t1w_files, bold_files, args.subjects_dir, qc_report_path, std_spaces, nstd_spaces)
    if len(t1w_files) > 0:
        TemplateDimensions_run(args.subject_id, t1w_files, qc_report_path)
    AboutSummary_run(args.subject_id, command, args.deepprep_version)

    create_report(subj_qc_report_path, qc_report_path, args.subject_id, reports_utils_path)
