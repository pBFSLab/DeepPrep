#! /usr/bin/env python3
import sh
from pathlib import Path
import argparse
import os


def run_norigid_registration(subject_id, script, subj_func_dir, T1_file, norm_2mm, template, affine_trans, mp):
    T1_save_name = f'{subject_id}_space-MNI152_res-2mm_desc-skull_T1w'
    moved = Path(subj_func_dir) / f'{T1_save_name}.nii.gz'

    norm_save_name = f'{subject_id}_space-MNI152_res-2mm_desc-noskull_T1w'
    apply_output = Path(subj_func_dir) / f'{norm_save_name}.nii.gz'
    cmd = f'python3 {script} -g -i {affine_trans} -o {moved} {T1_file} {template} -mp {mp} -a {norm_2mm} -ao {apply_output}'
    os.system(cmd)

    transvoxel = moved.parent / moved.name.replace('.nii.gz', '_transvoxel.npz')
    xfm = moved.parent / f'{subject_id}_from-T1w_to_MNI152_desc-nonlinear_xfm.npz'
    cmd = f'mv {transvoxel} {xfm}'
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- synthmorph norigid"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--synth_script", required=True)
    parser.add_argument("--t1_native2mm", required=True)
    parser.add_argument("--norm_native2mm", required=True)
    parser.add_argument("--affine_trans", required=True)
    parser.add_argument("--synth_model_path", required=True)
    parser.add_argument("--synth_template_path", required=True)
    args = parser.parse_args()

    preprocess_dir = Path(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    T1_2mm = args.t1_native2mm  # subj_func_dir / f'{args.subject_id}_space-T1w_res-2mm_desc-skull_T1w.nii.gz'
    norm_2mm = args.norm_native2mm  # subj_func_dir / f'{args.subject_id}_space-T1w_res-2mm_desc-noskull_T1w.nii.gz'
    template = Path(args.synth_template_path) / 'MNI152_T1_2mm.nii.gz'
    affine_trans = subj_func_dir / f'{args.subject_id}_from-T1w_to-MNI152_desc-affine_xfm.txt'
    run_norigid_registration(args.subject_id, args.synth_script, subj_func_dir, T1_2mm, norm_2mm, template, affine_trans, args.synth_model_path)
