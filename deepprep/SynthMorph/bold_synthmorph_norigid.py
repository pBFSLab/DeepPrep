#! /usr/bin/env python3
import sh
from pathlib import Path
import argparse
import os
import templateflow.api as tflow


def run_norigid_registration(subject_id, script, subj_anat_dir, T1_file, norm_2mm, template, affine_trans, mp, template_space):
    T1_save_name = f'{subject_id}_space-{template_space}_res-02_desc-skull_T1w'
    moved = Path(subj_anat_dir) / f'{T1_save_name}.nii.gz'

    norm_save_name = f'{subject_id}_space-{template_space}_res-02_desc-noskull_T1w'
    apply_output = Path(subj_anat_dir) / f'{norm_save_name}.nii.gz'
    cmd = f'python3 {script} -i {affine_trans} -o {moved} {T1_file} {template} -mp {mp} -a {norm_2mm} -ao {apply_output}'
    os.system(cmd)

    transvoxel = moved.parent / moved.name.replace('.nii.gz', '_transvoxel.npz')
    xfm = moved.parent / f'{subject_id}_from-T1w_to-{template_space}_desc-nonlinear_xfm.npz'
    cmd = f'mv {transvoxel} {xfm}'
    os.system(cmd)

    assert os.path.exists(moved), f"{moved}"
    assert os.path.exists(apply_output), f"{apply_output}"
    assert os.path.exists(xfm), f"{xfm}"


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
    parser.add_argument("--template_space", required=True)
    args = parser.parse_args()

    preprocess_dir = Path(args.bold_preprocess_dir) / args.subject_id
    subj_anat_dir = Path(preprocess_dir) / 'anat'
    subj_anat_dir.mkdir(parents=True, exist_ok=True)

    T1_2mm = args.t1_native2mm  # subj_func_dir / f'{args.subject_id}_space-T1w_res-2mm_desc-skull_T1w.nii.gz'
    norm_2mm = args.norm_native2mm  # subj_func_dir / f'{args.subject_id}_space-T1w_res-2mm_desc-noskull_T1w.nii.gz'
    template = tflow.get(args.template_space, desc=None, resolution=2, suffix='T1w', extension='nii.gz')
    affine_trans = subj_anat_dir / f'{args.subject_id}_from-T1w_to-{args.template_space}_desc-affine_xfm.txt'
    run_norigid_registration(args.subject_id, args.synth_script, subj_anat_dir, T1_2mm, norm_2mm, template, affine_trans, args.synth_model_path, args.template_space)
