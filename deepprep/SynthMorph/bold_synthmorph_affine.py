#! /usr/bin/env python3
import sh
from pathlib import Path
import argparse
import os
import templateflow.api as tflow



def run_rigid_registration(subject_id, script, subj_anat_dir, T1_file, template, mp, template_space):
    moved = Path(subj_anat_dir) / f'{subject_id}_space-{template_space}_res-02_desc-affine_T1w.nii.gz'
    trans = Path(subj_anat_dir) / f'{subject_id}_from-T1w_to-{template_space}_desc-affine_xfm.txt'

    cmd = f'python3 {script} -m affine -t {trans} -o {moved} {T1_file} {template} -mp {mp}'
    os.system(cmd)

    assert os.path.exists(moved), f"{moved}"
    assert os.path.exists(trans), f"{trans}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- synthmorph affine"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--synth_script", required=True)
    parser.add_argument("--t1_native2mm", required=True)
    parser.add_argument("--synth_model_path", required=True)
    parser.add_argument("--template_space", required=True)
    args = parser.parse_args()

    preprocess_dir = Path(args.bold_preprocess_dir) / args.subject_id
    subj_anat_dir = Path(preprocess_dir) / 'anat'
    subj_anat_dir.mkdir(parents=True, exist_ok=True)

    # T1_2mm = subj_func_dir / f'{args.subject_id}_space-T1w_res-2mm_desc-skull_T1w.nii.gz'
    # template = Path(args.synth_template_path) / 'MNI152_T1_2mm.nii.gz'
    template = tflow.get(args.template_space, desc=None, resolution=2, suffix='T1w', extension='nii.gz')
    run_rigid_registration(args.subject_id, args.synth_script, subj_anat_dir, args.t1_native2mm, template, args.synth_model_path, args.template_space)

