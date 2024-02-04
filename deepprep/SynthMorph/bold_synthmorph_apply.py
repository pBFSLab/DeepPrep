#! /usr/bin/env python3
import os
from pathlib import Path
import argparse
import templateflow.api as tflow


def run_norigid_registration_apply(script, upsampled_dir, fframe_bold_output, T1_file, template, transvoxel, bold_file, batch_size):

    cmd = f'python3 {script}  -fbo {fframe_bold_output} {T1_file} {template} -tv {transvoxel} -ob {bold_file} -up {upsampled_dir} -bs {batch_size}'
    os.system(cmd)

def get_space_t1w_bold(bids_orig, bids_preproc, bold_orig_file):
    from bids import BIDSLayout
    layout_orig = BIDSLayout(bids_orig, validate=False)
    layout_preproc = BIDSLayout(bids_preproc, validate=False)
    info = layout_orig.parse_file_entities(bold_orig_file)
    bold_t1w_info = info.copy()
    bold_t1w_info['space'] = 'T1w'
    bold_t1w_info['suffix'] = 'bold'
    bold_t1w_info['extension'] = '.nii.gz'
    bold_t1w_file = layout_preproc.get(**bold_t1w_info)[0]

    return bold_t1w_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- SynthmorphBoldApply"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--upsampled_dir", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--T1_file", required=True)
    parser.add_argument("--subject_boldfile_txt_bold", required=True)
    parser.add_argument("--trans_vox", required=True)
    parser.add_argument("--template_space", required=True)
    parser.add_argument("--template_resolution", required=True)
    parser.add_argument("--batch_size", required=True)
    parser.add_argument("--synth_script", required=True)
    args = parser.parse_args()

    T1_2mm = args.T1_file
    transvoxel = args.trans_vox

    with open(args.subject_boldfile_txt_bold, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_file = data[1]
    bold_t1w_file = get_space_t1w_bold(args.bids_dir, args.bold_preprocess_dir, bold_file)

    template_resolution = args.template_resolution
    template = tflow.get(args.template_space, desc=None, resolution=template_resolution, suffix='T1w', extension='nii.gz')
    fframe_bold_output = Path(bold_t1w_file.dirname) / f'{args.bold_id}_space-{args.template_space}_res-{template_resolution}_boldref.nii.gz'
    run_norigid_registration_apply(args.synth_script, args.upsampled_dir, fframe_bold_output, T1_2mm, template, transvoxel, bold_file, int(args.batch_size))

    assert os.path.exists(fframe_bold_output), f'{fframe_bold_output}'
