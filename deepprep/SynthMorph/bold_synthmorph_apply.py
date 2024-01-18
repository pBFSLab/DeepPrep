#! /usr/bin/env python3
import os
from pathlib import Path
import argparse
import templateflow.api as tflow
import shutil
from multiprocessing import Pool


def upsampling_bold(tar_t1w_file, split_bold_file, unsampling_dir):
    tar_split_bold_file = Path(unsampling_dir) / Path(split_bold_file).name
    cmd = f'mri_convert -rl {tar_t1w_file} {split_bold_file} {tar_split_bold_file} --out_data_type float'
    os.system(cmd)

def split_bold_convert_concat(ori_bold_file, work_dir, tar_t1w_file, bold_id, process_num):
    split_bold_dir = Path(work_dir) / bold_id / 'split'
    split_bold_dir.mkdir(exist_ok=True, parents=True)
    split_bold_files = split_bold_dir / 's.nii.gz'
    unsampling_dir = Path(work_dir) / bold_id / 'unsampling'
    unsampling_dir.mkdir(exist_ok=True, parents=True)

    # 1.split bold
    cmd = f'mri_convert --split {ori_bold_file} {split_bold_files}'
    os.system(cmd)

    #2. unsampled bold
    multiprocess = []

    split_bold_files = sorted(os.listdir(split_bold_dir))
    for n in range(len(split_bold_files)):
        split_bold_file = split_bold_dir / split_bold_files[n]
        multiprocess.append((tar_t1w_file, split_bold_file, unsampling_dir))
    with Pool(process_num) as pool:
        pool.starmap(upsampling_bold, multiprocess)

    #3. concat bold
    concat_bold_file = Path(work_dir) / bold_id / Path(ori_bold_file).name.replace('.nii.gz', '_unsampled.nii.gz')
    cmd = f'mri_concat --i {unsampling_dir}/* --o {concat_bold_file}'
    os.system(cmd)

    rm_dir = split_bold_dir.parent
    return concat_bold_file, rm_dir


def run_norigid_registration_apply(script, bold, bold_output, fframe_bold_output, T1_file, template, transvoxel):

    cmd = f'python3 {script} -g -b {bold} -bo {bold_output} -fbo {fframe_bold_output} {T1_file} {template} -tv {transvoxel}'
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
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--T1_file", required=True)
    parser.add_argument("--subject_boldfile_txt_bold", required=True)
    parser.add_argument("--trans_vox", required=True)
    parser.add_argument("--template_space", required=True)
    parser.add_argument("--template_resolution", required=True)
    parser.add_argument("--process_num", required=True)
    parser.add_argument("--synth_script", required=True)
    args = parser.parse_args()



    T1_2mm = args.T1_file
    transvoxel = args.trans_vox

    with open(args.subject_boldfile_txt_bold, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_file = data[1]
    bold_t1w_file = get_space_t1w_bold(args.bids_dir, args.bold_preprocess_dir, bold_file)
    unsampled_bold, rm_dir = split_bold_convert_concat(bold_t1w_file.path, args.work_dir, T1_2mm, args.bold_id, process_num=int(args.process_num))

    template_resolution = args.template_resolution
    template = tflow.get(args.template_space, desc=None, resolution=template_resolution, suffix='T1w', extension='nii.gz')
    bold_output = Path(bold_t1w_file.dirname) / f'{args.bold_id}_space-{args.template_space}_res-{args.template_resolution}_desc-preproc_bold.nii.gz'
    fframe_bold_output = Path(bold_t1w_file.dirname) / f'{args.bold_id}_space-{args.template_space}_res-{args.template_resolution}_boldref.nii.gz'
    run_norigid_registration_apply(args.synth_script, unsampled_bold, bold_output, fframe_bold_output, T1_2mm, template, transvoxel)
    assert os.path.exists(bold_output), f'{bold_output}'
    assert os.path.exists(fframe_bold_output), f'{fframe_bold_output}'
    shutil.rmtree(rm_dir)
