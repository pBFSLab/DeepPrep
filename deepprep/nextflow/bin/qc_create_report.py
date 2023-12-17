#! /usr/bin/env python3
import os
import argparse
from reports.core import run_reports
from uuid import uuid4
from pathlib import Path
from time import strftime
from reports.reports_node import SubjectSummary, TemplateDimensions, FunctionalSummary, AboutSummary
from nipype import Node
import shutil


def is_deepprep_recon(subjects_dir, subject_id):
    deepprep_version_file = Path(subjects_dir) / subject_id / 'scripts' / 'deepprep_version.txt'
    return deepprep_version_file.exists()


def SubjectSummary_run():
    # 这个步骤使用BIDS的输出作为输入就可以了

    t1w = ['/mnt/ssd/temp/UKB_1/sub-1000037ses02/ses-02/anat/sub-1000037ses02_ses-02_T1w.nii.gz']
    t2w = []
    bold = ['/mnt/ssd/temp/UKB_1/sub-1000037ses02/ses-02/func/sub-1000037ses02_ses-02_task-rest_run-01_bold.nii.gz']
    qc_report_dir = '/mnt/ssd/temp/UKB_1/UKB_1_QC'
    subjects_dir = '/mnt/ssd/temp/UKB_1/UKB_1_Recon'
    subject_id = 'sub-1000037ses02'
    std_spaces = ['MNI152NLin2009cAsym']
    nstd_spaces = ['T1w']

    if is_deepprep_recon(subjects_dir, subject_id):
        freesurfer_status = 'Run by DeepPrep'
    else:
        freesurfer_status = 'Pre-existing directory'

    node_name = 'SubjectSummary_run_node'
    Reports_node = Node(SubjectSummary(), node_name)

    Reports_node.interface.freesurfer_status = freesurfer_status
    Reports_node.inputs.t1w = t1w
    Reports_node.inputs.t2w = t2w
    Reports_node.inputs.subjects_dir = subjects_dir
    Reports_node.inputs.subject_id = subject_id
    Reports_node.inputs.bold = bold
    Reports_node.inputs.std_spaces = std_spaces
    Reports_node.inputs.nstd_spaces = nstd_spaces

    Reports_node.base_dir = Path().cwd()
    Reports_node.run()
    shutil.copyfile(Path(node_name) / 'report.html',
                    Path(qc_report_dir) / subject_id / 'figures' / f'{subject_id}_desc-subjectsummary_report.html')



def TemplateDimensions_run():
    # 这个步骤需要每个subject执行一次，将T1w作为输入

    subject_id = 'sub-1000037ses02'
    t1w = ['/mnt/ssd/temp/UKB_1/sub-1000037ses02/ses-02/anat/sub-1000037ses02_ses-02_T1w.nii.gz']

    node_name = 'T1w_Reports_run_node'
    TemplateDimensions_node = Node(TemplateDimensions(), node_name)
    TemplateDimensions_node.inputs.t1w_list = t1w

    TemplateDimensions_node.base_dir = Path().cwd()
    TemplateDimensions_node.run()
    shutil.copyfile(Path(node_name) / 'report.html',
                    Path(qc_report_dir) / subject_id / 'figures' / f'{subject_id}_desc-templatedimensions_report.html')


def FunctionalSummary_run():
    subject_id = 'sub-1000037ses02'
    bold_id = ''  # 这个步骤需要每个func使用一次

    node_name = 'functional_Reports_run_node'
    FunctionalSummary_node = Node(FunctionalSummary(), node_name)

    FunctionalSummary_node.inputs.orientation = 'LAS'
    FunctionalSummary_node.inputs.tr = 2
    FunctionalSummary_node.inputs.slice_timing = True
    FunctionalSummary_node.inputs.distortion_correction = 'None'

    # TODO these are not valid
    FunctionalSummary_node.inputs.pe_direction = 'i'
    FunctionalSummary_node.inputs.registration = 'FreeSurfer and SynthMorph'
    FunctionalSummary_node.inputs.registration_dof = 9
    FunctionalSummary_node.inputs.registration_init = 'register'
    FunctionalSummary_node.inputs.fallback = True

    FunctionalSummary_node.base_dir = Path().cwd()
    FunctionalSummary_node.run()
    shutil.copyfile(Path(node_name) / 'report.html',
                    Path(qc_report_dir) / subject_id / 'figures' / f'{bold_id}_desc-functionalsummary_report.html')


def AboutSummary_run():
    subject_id = 'sub-1000037ses02'

    node_name = 'AboutSummary_Reports_run_node'
    AboutSummary_node = Node(AboutSummary(), node_name)

    AboutSummary_node.inputs.version = 'v0.0.1'
    AboutSummary_node.inputs.command = 'nextflow run /root/workspace/DeepPrep/deepprep/nextflow/deepprep.nf -resume    -c /root/workspace/DeepPrep/deepprep/nextflow/nextflow.docker.local.config     -with-report /mnt/ssd/temp/UKB_1/UKB_1_QC/report.html     -with-timeline /mnt/ssd/temp/UKB_1/UKB_1_QC/timeline.html     --bids_dir /mnt/ssd/temp/UKB_1     --subjects_dir /mnt/ssd/temp/UKB_1/UKB_1_Recon --bold_preprocess_path /mnt/ssd/temp/UKB_1/UKB_1_BOLD     --qc_result_path /mnt/ssd/temp/UKB_1/UKB_1_QC     --bold_type rest  --bold_only true'

    AboutSummary_node.base_dir = Path().cwd()
    AboutSummary_node.run()
    shutil.copyfile(Path(node_name) / 'report.html',
                    Path(qc_report_dir) / subject_id / 'figures' / f'{subject_id}_desc-aboutsummary_report.html')


def create_report(subj_qc_report_dir, qc_report_dir, subject_id, nextflow_bin_path):
    log_dir = subj_qc_report_dir / 'logs'
    reports_spec = nextflow_bin_path / 'reports' / 'reports-spec-deepprep.yml'
    boilerplate_dir = nextflow_bin_path / 'reports' / 'logs'
    run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid4()}"
    if not log_dir.exists():
        cmd = f'cp -r {boilerplate_dir} {log_dir}'
        os.system(cmd)
    run_reports(subj_qc_report_dir, subject_id, run_uuid,
                config=reports_spec, packagename='deepprep',
                reportlets_dir=qc_report_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: create qc report"
    )
    parser.add_argument("--qc_result_path", help="save qc report path", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--nextflow_bin_path", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()
    qc_report_dir = Path(cur_path) / str(args.qc_result_path)
    subj_qc_report_dir = qc_report_dir / args.subject_id
    subj_qc_report_dir.mkdir(parents=True, exist_ok=True)
    nextflow_bin_path = Path(cur_path) / str(args.nextflow_bin_path)

    SubjectSummary_run()
    TemplateDimensions_run()
    FunctionalSummary_run()
    AboutSummary_run()

    create_report(subj_qc_report_dir, qc_report_dir, args.subject_id, nextflow_bin_path)
