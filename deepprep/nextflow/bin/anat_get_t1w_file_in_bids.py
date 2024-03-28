#! /usr/bin/env python3
import os
import argparse
import bids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: sMRI and fMRI PreProcessing workflows"
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
    t1w_filess = []
    for t1w_file in layout.get(return_type='filename', subject=subject_ids, suffix="T1w", extension='.nii.gz'):
        sub_info = layout.parse_file_entities(t1w_file)
        subject_id = f"sub-{sub_info['subject']}"
        subject_dict.setdefault(subject_id, []).append(t1w_file)
        subject_ids = list(subject_dict.keys())
        t1w_filess = list(subject_dict.values())
    for subject_id, t1w_files in zip(subject_ids, t1w_filess):
        with open(f'{subject_id}', 'w') as f:
            f.write(subject_id + '\n')
            f.write('\n'.join(t1w_files))
