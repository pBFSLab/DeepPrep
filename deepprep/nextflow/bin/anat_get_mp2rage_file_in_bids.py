#! /usr/bin/env python3
import argparse
import bids
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: sMRI and fMRI PreProcessing workflows: get mp2rage files"
    )

    parser.add_argument("--bids-dir", help="directory of BIDS type: /mnt/ngshare2/BIDS/MSC", required=True)
    parser.add_argument('--subject-ids', type=str, nargs='+', default=[], help='specified subject_id')
    args = parser.parse_args()

    if len(args.subject_ids) != 0:
        subject_ids = [subject_id[4:] if subject_id.startswith('sub-') else subject_id for subject_id in args.subject_ids]
    else:
        subject_ids = args.subject_ids
    layout = bids.BIDSLayout(args.bids_dir, derivatives=False)
    subject_dict = {}
    for unit1_file in layout.get(return_type='filename', subject=subject_ids, suffix="UNIT1", extension=['.nii.gz', '.nii']):
        sub_info = layout.parse_file_entities(unit1_file)
        subject_id = f"sub-{sub_info['subject']}"
        subject_dict.setdefault(subject_id, []).append(unit1_file)
    for mp2rage_file in layout.get(return_type='filename', subject=subject_ids, suffix="MP2RAGE", extension=['.nii.gz', '.nii']):
        sub_info = layout.parse_file_entities(mp2rage_file)
        subject_id = f"sub-{sub_info['subject']}"
        subject_dict.setdefault(subject_id, []).append(mp2rage_file)
    subject_ids = list(subject_dict.keys())
    subject_data_files = list(subject_dict.values())
    for subject_id, subject_data_file in zip(subject_ids, subject_data_files):
        with open(os.path.join(os.getcwd(), f'{subject_id}-mp2rage'), 'w') as f:
            f.write(subject_id + '\n')
            f.write('\n'.join(subject_data_file))
