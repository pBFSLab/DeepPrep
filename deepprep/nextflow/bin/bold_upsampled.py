#! /usr/bin/env python3
import os
from pathlib import Path
import argparse
from multiprocessing import Pool


def upsampling_bold(tar_t1w_file, split_bold_file, upsampled_dir):
    tar_split_bold_file = Path(upsampled_dir) / Path(split_bold_file).name
    cmd = f'mri_convert -rl {tar_t1w_file} {split_bold_file} {tar_split_bold_file} --out_data_type float'
    os.system(cmd)

def split_bold_convert(ori_bold_file, work_dir, tar_t1w_file, bold_id, process_num):
    split_bold_dir = Path(work_dir) / bold_id / 'split'
    split_bold_dir.mkdir(exist_ok=True, parents=True)
    split_bold_files = split_bold_dir / 's.nii.gz'
    upsampled_dir = Path(work_dir) / bold_id / 'upsampling'
    upsampled_dir.mkdir(exist_ok=True, parents=True)

    # 1.split bold
    cmd = f'mri_convert --split {ori_bold_file} {split_bold_files}'
    os.system(cmd)

    #2. upsampled bold
    multiprocess = []

    split_bold_files = sorted(os.listdir(split_bold_dir))
    for n in range(len(split_bold_files)):
        split_bold_file = split_bold_dir / split_bold_files[n]
        multiprocess.append((tar_t1w_file, split_bold_file, upsampled_dir))
    with Pool(process_num) as pool:
        pool.starmap(upsampling_bold, multiprocess)

    rm_dir = split_bold_dir.parent
    return upsampled_dir, rm_dir


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
        description="DeepPrep: Bold PreProcessing workflows -- BoldUpsampled"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--T1_file", required=True)
    parser.add_argument("--subject_boldfile_txt_bold", required=True)
    parser.add_argument("--process_num", required=True)
    args = parser.parse_args()

    T1_2mm = args.T1_file

    with open(args.subject_boldfile_txt_bold, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_file = data[1]
    bold_t1w_file = get_space_t1w_bold(args.bids_dir, args.bold_preprocess_dir, bold_file)
    upsampled_dir, rm_dir = split_bold_convert(bold_t1w_file.path, args.work_dir, T1_2mm, args.bold_id, process_num=int(args.process_num))
    assert os.path.exists(upsampled_dir), f'{upsampled_dir}'