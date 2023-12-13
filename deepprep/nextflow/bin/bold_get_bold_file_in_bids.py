#! /usr/bin/env python3
import os
import argparse
import bids
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- get_bold_files"
    )

    parser.add_argument("--bids_dir", required=True, help="directory of BIDS type: /mnt/ngshare2/BIDS/MSC")
    parser.add_argument("--task_type", required=True, type=str, help="rest or etc..")
    args = parser.parse_args()

    layout = bids.BIDSLayout(args.bids_dir, derivatives=False)
    subject_dict = {}
    subject_ids = []
    bold_filess = []
    if args.task_type is not None:
        bids_bolds = layout.get(return_type='filename', task=args.task_type, suffix='bold', extension='.nii.gz')
    else:
        bids_bolds = layout.get(return_type='filename', suffix='bold', extension='.nii.gz')

    for bold_file in bids_bolds:
        sub_info = layout.parse_file_entities(bold_file)
        subject_id = f"sub-{sub_info['subject']}"
        subject_dict.setdefault(subject_id, []).append(bold_file)
        subject_ids = list(subject_dict.keys())
        bold_filess = list(subject_dict.values())
    for subject_id, bold_files in zip(subject_ids, bold_filess):
        for bold_file in bold_files:
            filename = os.path.basename(bold_file).split('.')[0]
            with open(f'{filename}', 'w') as f:
                f.write(subject_id + '\n')
                f.writelines(bold_file)
