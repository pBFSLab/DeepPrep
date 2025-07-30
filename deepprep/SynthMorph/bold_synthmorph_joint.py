#! /usr/bin/env python3
from pathlib import Path
import argparse
import os
import templateflow.api as tflow


def parse_string_to_dict(input_template):
    result_dict = {'template': input_template.split(':', 1)[0] if ':' in input_template else input_template}

    if ':' in input_template:
        for pair in input_template.split(':', 1)[1].split(':'):
            if '-' in pair:
                key, value = pair.split('-', 1)
                result_dict[key] = value

    return result_dict

def run_joint_registration(subject_id, script, subj_anat_dir, norm_2mm, T1_file, template, mp, template_dict):
    template_space = template_dict['template']
    if 'cohort' in template_dict.keys():
        cohort = template_dict['cohort']
        moved = Path(subj_anat_dir) / f'{subject_id}_space-{template_space}_cohort-{cohort}_res-02_desc-skull_T1w.nii.gz'
        trans = Path(subj_anat_dir) / f'{subject_id}_from-T1w_to-{template_space}_cohort-{cohort}_desc-joint_trans.nii.gz'

        apply_output = Path(subj_anat_dir) / f'{subject_id}_space-{template_space}_cohort-{cohort}_res-02_desc-noskull_T1w.nii.gz'
    else:
        moved = Path(subj_anat_dir) / f'{subject_id}_space-{template_space}_res-02_desc-skull_T1w.nii.gz'
        trans = Path(subj_anat_dir) / f'{subject_id}_from-T1w_to-{template_space}_desc-joint_trans.nii.gz'

        apply_output = Path(subj_anat_dir) / f'{subject_id}_space-{template_space}_res-02_desc-noskull_T1w.nii.gz'
    cmd = f'python3 {script} -t {trans} -o {moved} {T1_file} {template} -mp {mp} -a {norm_2mm} -ao {apply_output}'
    os.system(cmd)

    assert os.path.exists(moved), f"{moved}"
    assert os.path.exists(trans), f"{trans}"
    assert os.path.exists(apply_output), f"{apply_output}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- synthmorph joint"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--synth_script", required=True)
    parser.add_argument("--t1_native2mm", required=True)
    parser.add_argument("--norm_native2mm", required=True)
    parser.add_argument("--synth_model_path", required=True)
    parser.add_argument("--template_space", required=True)
    args = parser.parse_args()

    preprocess_dir = Path(args.bold_preprocess_dir) / args.subject_id
    subj_anat_dir = Path(preprocess_dir) / 'anat'
    subj_anat_dir.mkdir(parents=True, exist_ok=True)

    T1_2mm = args.t1_native2mm  # subj_func_dir / f'{args.subject_id}_space-T1w_res-2mm_desc-skull_T1w.nii.gz'
    norm_2mm = args.norm_native2mm  # subj_func_dir / f'{args.subject_id}_space-T1w_res-2mm_desc-noskull_T1w.nii.gz'
    template_dict = parse_string_to_dict(args.template_space)
    if 'cohort' in template_dict.keys():
        template = tflow.get(template_dict['template'], desc=None, resolution=2, cohort=template_dict['cohort'], suffix='T1w', extension='nii.gz')
    else:
        template = tflow.get(args.template_space, desc=None, resolution=2, suffix='T1w', extension='nii.gz')
    assert template, f"template_space: {template}"  # 名称错误导致template为[]的问题
    run_joint_registration(args.subject_id, args.synth_script, subj_anat_dir, norm_2mm, T1_2mm, template, args.synth_model_path, template_dict)
