#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import argparse
import os.path
import shutil

from bids import BIDSLayout
import json
import uuid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- get_bold_files"
    )

    # input path required
    parser.add_argument("--bids_dir", required=True, help="preprocessed BIDS directory")
    # input flags required
    parser.add_argument("--task_id", required=True, help="")
    parser.add_argument("--space", required=False, help="")
    # input flags optional
    parser.add_argument("--subject_id", required=False, help="")
    # output path
    parser.add_argument("--output_dir", required=True, help="")
    parser.add_argument("--work_dir", required=True, help="")

    args = parser.parse_args()

    """test
--bids_dir /mnt/ngshare/Data_User/xuna/denoise/ADXW026_MSYU2/Noise_DeepPrep_2410/BOLD
--task_id rest
--space fsaverage6
--output_dir /mnt/ngshare/Data_User/xuna/denoise/ADXW026_MSYU2/Noise_DeepPrep_2410_postprocess/BOLD
--work_dir /mnt/ngshare/Data_User/xuna/denoise/ADXW026_MSYU2/Noise_DeepPrep_2410_postprocess/WorkDir
    """

    assert os.path.isdir(args.bids_dir)
    assert os.path.isfile(os.path.join(args.bids_dir, 'dataset_description.json'))

    bids_dir = args.bids_dir
    subject_id = args.subject_id
    task_id = args.task_id

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    shutil.copyfile(os.path.join(args.bids_dir, 'dataset_description.json'), os.path.join(args.output_dir, 'dataset_description.json'))

    database_path = os.path.join(args.work_dir, str(uuid.uuid4()))
    layout_bids = BIDSLayout(bids_dir, validate=False, database_path=database_path, reset_database=True)

    orig_entities = {
        'suffix': 'bold',
        'space': args.space,
    }
    if task_id:
        orig_entities['task'] = task_id
    if subject_id:
        orig_entities['subject'] = subject_id
    bold_orig_files = layout_bids.get(**orig_entities)

    for bold_orig_file in bold_orig_files:
        bold_orig_file_path = bold_orig_file.path
        extension = layout_bids.parse_file_entities(bold_orig_file_path)['extension']
        if not extension in ('.nii.gz', '.func.gii'):
            continue

        bold_id = bold_orig_file.filename.split(extension)[0]
        with open(bold_id + '.json', 'w') as f:
            data_json = {
                'bold_file': bold_orig_file_path,
                'bids_database_path': database_path
            }
            json.dump(data_json, f, indent=4)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
