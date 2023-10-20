#! /usr/bin/env python3
import os
from pathlib import Path
import argparse


def run_norigid_registration_apply(script, bold, bold_output, T1_file, template, mc, transvoxel):

    cmd = f'python3 {script} -g -b {bold} -bo {bold_output} {T1_file} {template} -mc {mc} -tv {transvoxel}'
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- SynthmorphBoldApply"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    # parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    # parser.add_argument("--atlas_type", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--T1_file", required=True)
    parser.add_argument("--mc", required=True)
    parser.add_argument("--bold", required=True)
    parser.add_argument("--trans_vox", required=True)
    parser.add_argument("--synth_template_path", required=True)
    # parser.add_argument("--synth_model_path", required=True)
    parser.add_argument("--synth_script", required=True)
    # parser.add_argument("--gpuid", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()
    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    mc_file = subj_func_dir / f'{args.bold_id}_skip_reorient_stc_mc.nii.gz'
    bold = subj_func_dir / f'{args.bold_id}_skip_reorient_stc_mc_bbregister_space-native_2mm.nii.gz'
    bold_output = subj_func_dir / f'{args.bold_id}_skip_reorient_stc_mc_bbregister_space-native_2mm_synthmorph_space-MNI152_2mm.nii.gz'
    T1_file = subj_func_dir / f'{args.subject_id}_T1_2mm.nii.gz'
    transvoxel = subj_func_dir / f'{args.subject_id}_norm_2mm_affine_synthmorph_space-MNI152_2mm_transvoxel.npz'
    template = Path(cur_path) / str(args.synth_template_path) / 'MNI152_T1_2mm.nii.gz'

    run_norigid_registration_apply(args.synth_script, bold, bold_output, T1_file, template, mc_file, transvoxel)