#! /usr/bin/env python3
import os
import argparse
from reports.core import run_reports
from uuid import uuid4
import parser
from pathlib import Path
from time import strftime


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
    nextflow_bin_path = Path(cur_path) / str(args.nextflow_bin_path)

    create_report(subj_qc_report_dir, qc_report_dir, args.subject_id, nextflow_bin_path)