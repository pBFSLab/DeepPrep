#! /usr/bin/env python3
import sh
from pathlib import Path
import argparse
import os


def run_norigid_registration(subject_id, script, subj_anat_dir, T1_file, template, affine_trans, mp):
    save_name = f'{subject_id}_T1_2mm_affine_synthmorph_space-MNI152_2mm'
    moved = Path(subj_anat_dir) / f'{save_name}.nii.gz'

    cmd = f'python3 {script} -g -i {affine_trans} -o {moved} {T1_file} {template} -mp {mp}'
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- synthmorph norigid"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--synth_script", required=True)
    parser.add_argument("--t1_native2mm", required=True)
    parser.add_argument("--affine_trans", required=True)
    parser.add_argument("--synth_model_path", required=True)
    parser.add_argument("--synth_template_path", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    T1_2mm = subj_func_dir / f'{args.subject_id}_T1_2mm.nii.gz'
    template = Path(cur_path) / str(args.synth_template_path) / 'MNI152_T1_2mm.nii.gz'
    affine_trans = subj_func_dir / f'{args.subject_id}_T1_2mm_affine_space-MNI152_2mm.xfm.txt'
    run_norigid_registration(args.subject_id, args.synth_script, subj_func_dir, T1_2mm, template, affine_trans, args.synth_model_path)
