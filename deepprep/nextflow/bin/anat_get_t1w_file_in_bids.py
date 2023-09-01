#! /usr/bin/env python3
import os
import argparse
import bids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: sMRI and fMRI PreProcessing workflows"
    )

    parser.add_argument("--bids-dir", help="directory of BIDS type: /mnt/ngshare2/BIDS/MSC", required=True)
    # parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    layout = bids.BIDSLayout(args.bids_dir, derivatives=False)
    subject_dict = {}
    subject_ids = []
    t1w_filess = []
    for t1w_file in layout.get(return_type='filename', suffix="T1w", extension='.nii.gz'):
        print(t1w_file)
        sub_info = layout.parse_file_entities(t1w_file)
        subject_id = f"sub-{sub_info['subject']}"
        subject_dict.setdefault(subject_id, []).append(t1w_file)
        subject_ids = list(subject_dict.keys())
        t1w_filess = list(subject_dict.values())
    print(subject_ids)
    print(t1w_filess)
    # print(args.out_dir)
    for subject_id, t1w_files in zip(subject_ids, t1w_filess):
        with open(f'{subject_id}', 'w') as f:
            f.write(subject_id + '\n')
            f.writelines(t1w_files)
