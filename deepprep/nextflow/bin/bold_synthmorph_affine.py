#! /usr/bin/env python3
import sh
from pathlib import Path
import argparse
import os


def run_rigid_registration(subject_id, script, subj_anat_dir, T1_file, template, mp):
    moved = Path(subj_anat_dir) / f'{subject_id}_space-MNI152_res-2mm_desc-affine_T1w.nii.gz'
    trans = Path(subj_anat_dir) / f'{subject_id}_from-T1w_to-MNI152_desc-affine_xfm.txt'

    cmd = f'python3 {script} -g -m affine -t {trans} -o {moved} {T1_file} {template} -mp {mp}'
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- synthmorph affine"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--synth_script", required=True)
    parser.add_argument("--t1_native2mm", required=True)
    parser.add_argument("--synth_model_path", required=True)
    parser.add_argument("--synth_template_path", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    # T1_2mm = subj_func_dir / f'{args.subject_id}_space-T1w_res-2mm_desc-skull_T1w.nii.gz'
    # template = Path(args.synth_template_path) / 'MNI152_T1_2mm.nii.gz'
    template = Path(args.synth_template_path) / 'MNI152_T1_2mm.nii.gz'
    run_rigid_registration(args.subject_id, args.synth_script, subj_func_dir, args.t1_native2mm, template, args.synth_model_path)
