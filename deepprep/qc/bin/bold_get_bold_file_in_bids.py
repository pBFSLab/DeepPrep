#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import argparse
import os
import shutil
import json
import uuid
from bids import BIDSLayout


def main():
    """Main function to extract BOLD fMRI files from a BIDS directory based on specified criteria."""
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- get_bold_files"
    )

    # Input path required
    parser.add_argument("--bids_dir", required=True, help="Preprocessed BIDS directory")
    # Input flags required
    parser.add_argument("--task_id", required=False, nargs='+', help="Task ID(s)")
    # Input flags optional
    parser.add_argument("--subject_id", required=False, nargs='+', help="Subject ID(s)")
    # Output path
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--work_dir", required=True, help="Work directory")

    args = parser.parse_args()

    # Validate input paths
    assert os.path.isdir(args.bids_dir), "BIDS directory does not exist"
    assert os.path.isfile(os.path.join(args.bids_dir, 'dataset_description.json')), "dataset_description.json not found in BIDS directory"

    bids_dir = args.bids_dir
    subject_id = args.subject_id
    task_id = args.task_id

    # Create output and work directories if they do not exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    shutil.copyfile(os.path.join(args.bids_dir, 'dataset_description.json'), os.path.join(args.output_dir, 'dataset_description.json'))

    database_path = os.path.join(args.work_dir, str(uuid.uuid4()))
    layout_bids = BIDSLayout(bids_dir, validate=False, database_path=database_path, reset_database=True)

    orig_entities = {
        'suffix': 'bold',
        'extension': ['.nii.gz', '.nii']
    }
    if task_id:
        orig_entities['task'] = task_id
    if subject_id:
        orig_entities['subject'] = subject_id

    bold_orig_files = layout_bids.get(**orig_entities)

    for bold_orig_file in bold_orig_files:
        bold_orig_file_path = bold_orig_file.path
        extension = layout_bids.parse_file_entities(bold_orig_file_path)['extension']

        bold_id = bold_orig_file.filename.split(extension)[0]
        with open(bold_id + '.json', 'w') as f:
            data_json = {
                'bids_dir': bids_dir,
                'bold_file': bold_orig_file_path,
                'bids_database_path': database_path
            }
            json.dump(data_json, f, indent=4)


if __name__ == '__main__':
    main()