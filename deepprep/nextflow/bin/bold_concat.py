#! /usr/bin/env python3
import os
from pathlib import Path
import argparse
import shutil


def concat_bold(transformed_dir, concat_bold_file):
    cmd = f'mri_concat --i {transformed_dir}/* --o {concat_bold_file}'
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
        description="DeepPrep: Bold PreProcessing workflows -- BoldConcat"
    )
    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--subject_boldfile_txt_bold", required=True)
    parser.add_argument("--template_space", required=True)
    parser.add_argument("--template_resolution", required=True)
    parser.add_argument("--transform_dir", required=True)
    args = parser.parse_args()

    with open(args.subject_boldfile_txt_bold, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_file = data[1]
    bold_t1w_file = get_space_t1w_bold(args.bids_dir, args.bold_preprocess_dir, bold_file)

    bold_output = Path(bold_t1w_file.dirname) / f'{args.bold_id}_space-{args.template_space}_res-{args.template_resolution}_desc-preproc_bold.nii.gz'
    concat_bold(args.transform_dir, bold_output)
    assert os.path.exists(bold_output), f'{bold_output}'
    rm_dir = Path(args.transform_dir).parent
    shutil.rmtree(rm_dir)
