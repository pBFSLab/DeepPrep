#! /usr/bin/env python3
import os
from pathlib import Path
import argparse
import templateflow.api as tflow


def run_norigid_registration_apply(script, bold, bold_output, fframe_bold_output, T1_file, template, mc, transvoxel):

    cmd = f'python3 {script} -g -b {bold} -bo {bold_output} -fbo {fframe_bold_output} {T1_file} {template} -mc {mc} -tv {transvoxel}'
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- SynthmorphBoldApply"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--T1_file", required=True)
    parser.add_argument("--mc", required=True)
    parser.add_argument("--bold", required=True)
    parser.add_argument("--trans_vox", required=True)
    parser.add_argument("--template_space", required=True)
    parser.add_argument("--template_resolution", required=True)
    parser.add_argument("--synth_script", required=True)
    args = parser.parse_args()

    preprocess_dir = Path(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    T1_2mm = args.T1_file
    mc_file = args.mc
    bold = args.bold
    transvoxel = args.trans_vox

    template_resolution = args.template_resolution
    template = tflow.get(args.template_space, desc=None, resolution=template_resolution, suffix='T1w', extension='nii.gz')
    bold_output = subj_func_dir / f'{args.bold_id}_space-{args.template_space}_res-{args.template_resolution}_bold.nii.gz'
    fframe_bold_output = subj_func_dir / f'{args.bold_id}_space-{args.template_space}_res-{args.template_resolution}_boldref.nii.gz'
    run_norigid_registration_apply(args.synth_script, bold, bold_output, fframe_bold_output, T1_2mm, template, mc_file, transvoxel)
    assert os.path.exists(bold_output), f'{bold_output}'
    assert os.path.exists(fframe_bold_output), f'{fframe_bold_output}'
