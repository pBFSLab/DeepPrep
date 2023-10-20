#! /usr/bin/env python3
import sh
from pathlib import Path
import argparse
import os


def cmd(subject_id: str, subj_func_dir: Path, T1_fsnative_file):
    T1_fsnative2mm_file = subj_func_dir / f'{subject_id}_T1_2mm.nii.gz'
    if not T1_fsnative2mm_file.exists():
        sh.mri_convert('-ds', 2, 2, 2,
                       '-i', T1_fsnative_file,
                       '-o', T1_fsnative2mm_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- T1 to native2mm"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--t1_mgz", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = Path(cur_path) / str(args.subjects_dir) / args.subject_id
    t1_mgz = subject_dir / 'mri' / 'T1.mgz'

    cmd(args.subject_id, subj_func_dir, t1_mgz)