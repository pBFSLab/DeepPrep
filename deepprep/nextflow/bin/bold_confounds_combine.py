#! /usr/bin/env python3
import os
import argparse
from pathlib import Path
import pandas as pd


def get_confounds_file(subject_id, bids_preproc, bold_orig_file, update_entities):
    from bids import BIDSLayout
    assert subject_id.startswith('sub-')
    layout_preproc = BIDSLayout(str(os.path.join(bids_preproc, subject_id)),
                                config=['bids', 'derivatives'], validate=False)
    info = layout_preproc.parse_file_entities(bold_orig_file)

    confounds_info = info.copy()
    if update_entities:
        for k,v in update_entities.items():
            confounds_info[k] = v
        confounds_file = layout_preproc.get(**confounds_info)[0]
    else:
        confounds_file = layout_preproc.get(**confounds_info)[0]

    return Path(confounds_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- BoldSkipReorient"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--bold_file", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--confounds_part1", required=True)
    parser.add_argument("--confounds_part2", required=True)
    args = parser.parse_args()
    """
    input:
    --bids_dir /mnt/ngshare/DeepPrep_Docker/DeepPrep_workdir/UKB20
    --bold_preprocess_dir /mnt/ngshare/DeepPrep_Docker/DeepPrep_workdir/UKB20_pbfslab2410/BOLD
    --work_dir /mnt/ngshare/DeepPrep_Docker/DeepPrep_workdir/UKB20_pbfslab2410/WorkDir
    --bold_id sub-1001513_ses-01_task-rest
    --subject_id sub-1001513
    --bold_file /mnt/ngshare/DeepPrep_Docker/DeepPrep_workdir/UKB20_pbfslab2410/BOLD/sub-1001513/ses-01/func/sub-1001513_ses-01_task-rest_space-T1w_desc-preproc_bold.nii.gz
    --confounds_part1 True
    --confounds_part2 True
    output:
    confounds_file
    """

    confounds_dir = Path(args.work_dir) / 'confounds' / args.subject_id / args.bold_id

    confounds_part1 = confounds_dir / 'confounds_part1.tsv'

    with open(args.bold_file, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_orig_file = data[1]
    update_entities = {'desc': 'confounds', 'suffix': 'timeseries', 'extension': '.tsv'}
    confounds_file = get_confounds_file(args.subject_id, args.bold_preprocess_dir, bold_orig_file, update_entities)

    df1 = pd.read_csv(confounds_part1, sep='\t')
    df2 = pd.read_csv(confounds_file, sep='\t')
    cols_replace = ['csf', 'csf_derivative1', 'csf_wm']
    cols_concat = [f'e_comp_cor_{str(i).zfill(2)}' for i in range(10)]
    for c in cols_replace:
        if c in df2.columns:
            df2[c] = df1[c]
        else:
            cols_concat = [c] + cols_concat
    df = pd.concat((df2, df1[cols_concat]), axis=1)
    df.to_csv(confounds_file, sep='\t', index=False)
