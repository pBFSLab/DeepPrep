#! /usr/bin/env python3
import os
import argparse
from pathlib import Path
import ants
import numpy as np
import sh


def get_wm_dseg_file(aparc_aseg_mgz, wm_dseg_nii):
    data = ants.image_read(aparc_aseg_mgz)
    data_origin = data.origin[:3]
    data_spacing = data.spacing[:3]
    data_direction = data.direction[:3, :3].copy()
    data_np = data.numpy()
    data_wm = np.where((data_np == 41) | (data_np == 2), 2, 0)
    data_wm = data_wm.astype(np.float32)
    data_wm = ants.from_numpy(data_wm, data_origin, data_spacing, data_direction.copy())
    ants.image_write(data_wm, wm_dseg_nii)

def get_fsnative2T1w_xfm(fsnative2T1w_xfm):
    lines = ['#Transform {}',
             'Transform: MatrixOffsetTransformBase_double_3_3',
             'Parameters: 1 0 0 0 1 0 0 0 1 0 0 0',
             'FixedParameters: 0 0 0']
    with open(fsnative2T1w_xfm, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

def cmd(t1_mgz, mask_mgz, aparc_aseg_mgz, t1_nii, mask_nii, wm_dseg_nii, fsnative2T1w_xfm):
    os.system(f"mri_convert {t1_mgz} {t1_nii}")
    os.system(f"mri_convert {mask_mgz} {mask_nii}")

    get_wm_dseg_file(aparc_aseg_mgz, wm_dseg_nii)
    get_fsnative2T1w_xfm(fsnative2T1w_xfm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="DeepPrep: Prepare input data for BOLD process"
    )

    parser.add_argument("--bold_preprocess_path", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--t1_mgz", required=True)
    parser.add_argument("--mask_mgz", required=True)
    parser.add_argument("--aparc_aseg_mgz", required=True)
    parser.add_argument("--t1_nii", required=True)
    parser.add_argument("--mask_nii", required=True)
    parser.add_argument("--wm_dseg_nii", required=True)
    parser.add_argument("--fsnative2T1w_xfm", required=True)
    args = parser.parse_args()

    out_dir = Path(args.bold_preprocess_path) / args.subject_id / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd(args.t1_mgz, args.mask_mgz, args.aparc_aseg_mgz, args.t1_nii, args.mask_nii, args.wm_dseg_nii, args.fsnative2T1w_xfm)