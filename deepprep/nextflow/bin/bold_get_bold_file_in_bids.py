#! /usr/bin/env python3
import os
import argparse
import bids
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- get_bold_files"
    )

    parser.add_argument("--bids_dir", required=True, help="directory of BIDS type: /mnt/ngshare2/BIDS/MSC")
    parser.add_argument("--subjects_dir", required=True, help="directory of Recon results")
    parser.add_argument('--subject_ids', type=str, nargs='+', default=[], help='specified subject_id')
    parser.add_argument("--task_type", required=True, type=str, help="rest or etc..")
    parser.add_argument("--bold_only", required=True, type=str, help="TRUE or FALSE")
    args = parser.parse_args()

    if len(args.subject_ids) != 0:
        bold_subject_ids = [subject_id.split('-')[1] for subject_id in args.subject_ids]
        anat_subject_ids = bold_subject_ids
    else:
        bold_subject_ids = args.subject_ids
        anat_subject_ids = bold_subject_ids
    layout = bids.BIDSLayout(args.bids_dir, derivatives=False)
    bold_subject_dict = {}
    anat_subject_dict = {}
    bold_filess = []
    if args.task_type is not None:  # TODO delete extension=.nii.gz
        bids_bolds = layout.get(return_type='filename', subject=bold_subject_ids, task=args.task_type, suffix='bold', extension='.nii.gz')
    else:
        bids_bolds = layout.get(return_type='filename', subject=bold_subject_ids, suffix='bold', extension='.nii.gz')
    for bold_file in bids_bolds:
        sub_info = layout.parse_file_entities(bold_file)
        bold_subject_id = f"sub-{sub_info['subject']}"
        bold_subject_dict.setdefault(bold_subject_id, []).append(bold_file)
        bold_subject_ids = list(bold_subject_dict.keys())
        bold_filess = list(bold_subject_dict.values())
    if args.bold_only == 'FALSE':
        for t1w_file in layout.get(return_type='filename', subject=anat_subject_ids, suffix="T1w", extension='.nii.gz'):
            sub_info = layout.parse_file_entities(t1w_file)
            anat_subject_id = f"sub-{sub_info['subject']}"
            anat_subject_dict.setdefault(anat_subject_id, []).append(t1w_file)
            anat_subject_ids = list(anat_subject_dict.keys())
        bold_subject_ids = list(set(bold_subject_ids) & set(anat_subject_ids))

    else:
        recon_subject_ids = os.listdir(args.subjects_dir)
        recon_subject_ids = [item for item in recon_subject_ids if 'fsaverage' not in item]
        bold_subject_ids = list(set(bold_subject_ids) & set(recon_subject_ids))
    filter_bold_filess = []
    for bold_files in bold_filess:
        if os.path.basename(bold_files[0]).split('_')[0] in bold_subject_ids:
            filter_bold_filess.append(bold_files)
        else:
            continue
    for subject_id, bold_files in zip(sorted(bold_subject_ids), sorted(filter_bold_filess)):
        for bold_file in bold_files:
            if os.path.basename(bold_file).split('_')[0] in bold_subject_ids:
                bold_id = os.path.basename(bold_file).split('_bold')[0]
                with open(f'{bold_id}', 'w') as f:
                    f.write(subject_id + '\n')
                    f.writelines(bold_file)
