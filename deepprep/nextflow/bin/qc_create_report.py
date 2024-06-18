#! /usr/bin/env python3
import os
import argparse
from reports.core import run_reports
from uuid import uuid4
from pathlib import Path
from time import strftime


def create_report(subj_qc_report_path, qc_report_path, subject_id, reports_utils_path):
    log_dir = subj_qc_report_path / 'logs'
    reports_spec = reports_utils_path / 'reports-spec-deepprep.yml'
    boilerplate_dir = reports_utils_path / 'logs'
    run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid4()}"
    if not log_dir.exists():
        cmd = f'cp -r {boilerplate_dir} {log_dir}'
        os.system(cmd)

    subject_id = subject_id.split('sub-')[-1]
    run_reports(subj_qc_report_path, subject_id, run_uuid,
                config=reports_spec, packagename='deepprep',
                reportlets_dir=qc_report_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: create qc report"
    )
    parser.add_argument("--reports_utils_path", required=True)
    parser.add_argument("--bids_dir", help="BIDS dir path", required=True)
    parser.add_argument("--subjects_dir", help="SUBJECTS_DIR path", required=True)
    parser.add_argument("--qc_result_path", help="save qc report path", required=True)
    parser.add_argument("--subject_id", required=True)

    args = parser.parse_args()

    # cur_path = os.getcwd()  # 必须添加这一行，否则生成的html会有图片重复
    # dataset_path = Path(cur_path) / os.path.basename(args.qc_result_path)
    qc_report_path = Path(args.qc_result_path)
    subj_qc_report_path = qc_report_path / args.subject_id
    subj_qc_report_path.mkdir(parents=True, exist_ok=True)
    reports_utils_path = Path(args.reports_utils_path)

    create_report(subj_qc_report_path, qc_report_path, args.subject_id, reports_utils_path)
