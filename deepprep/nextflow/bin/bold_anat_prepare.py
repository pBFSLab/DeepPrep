#! /usr/bin/env python3
import os
import argparse
from pathlib import Path
import ants
import numpy as np
import sh


def get_seg_file(aseg_mgz, wm_dseg_nii, wm_probseg_nii, gm_probseg_nii, csf_probseg_nii):
    data = ants.image_read(aseg_mgz)
    data_origin = data.origin[:3]
    data_spacing = data.spacing[:3]
    data_direction = data.direction[:3, :3].copy()
    data_np = data.numpy()
    # get WM dseg & probseg
    data_wm = np.where((data_np == 2) | (data_np == 7) | (data_np == 41) | (data_np == 46) | (data_np == 77), 2, 0)
    data_probseg = np.where((data_np == 2) | (data_np == 7) | (data_np == 41) | (data_np == 46) | (data_np == 77), 1, 0)
    data_wm = data_wm.astype(np.float32)
    data_probseg = data_probseg.astype(np.float32)
    data_wm = ants.from_numpy(data_wm, data_origin, data_spacing, data_direction.copy())
    data_probseg = ants.from_numpy(data_probseg, data_origin, data_spacing, data_direction.copy())
    ants.image_write(data_wm, wm_dseg_nii)
    ants.image_write(data_probseg, wm_probseg_nii)
    # get GM, CSF probseg
    data_gm_probseg = np.where((data_np == 3) | (data_np == 8) | (data_np == 42) | (data_np == 47), 1, 0)
    data_csf_probseg = np.where((data_np == 24), 1, 0)
    data_gm_probseg = data_gm_probseg.astype(np.float32)
    data_csf_probseg = data_csf_probseg.astype(np.float32)
    data_gm_probseg = ants.from_numpy(data_gm_probseg, data_origin, data_spacing, data_direction.copy())
    data_csf_probseg = ants.from_numpy(data_csf_probseg, data_origin, data_spacing, data_direction.copy())
    ants.image_write(data_gm_probseg, gm_probseg_nii)
    ants.image_write(data_csf_probseg, csf_probseg_nii)

def get_fsnative2T1w_xfm(fsnative2T1w_xfm):
    lines = ['#Insight Transform File V1.0',
             '#Transform 0',
             'Transform: MatrixOffsetTransformBase_double_3_3',
             'Parameters: 1 0 0 0 1 0 0 0 1 0 0 0',
             'FixedParameters: 0 0 0']
    with open(fsnative2T1w_xfm, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

def cmd(t1_mgz, mask_mgz, aseg_mgz, t1_nii, mask_nii, wm_dseg_nii, fsnative2T1w_xfm, wm_probseg_nii, gm_probseg_nii, csf_probseg_nii):
    os.system(f"mri_convert {t1_mgz} {t1_nii}")
    os.system(f"mri_binarize --i {mask_mgz} --o {mask_nii} --min 0.0001")

    get_seg_file(aseg_mgz, wm_dseg_nii, wm_probseg_nii, gm_probseg_nii, csf_probseg_nii)
    get_fsnative2T1w_xfm(fsnative2T1w_xfm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="DeepPrep: Prepare input data for BOLD process"
    )

    parser.add_argument("--bold_preprocess_path", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--t1_mgz", required=True)
    parser.add_argument("--mask_mgz", required=True)
    parser.add_argument("--aseg_mgz", required=True)
    parser.add_argument("--t1_nii", required=True)
    parser.add_argument("--mask_nii", required=True)
    parser.add_argument("--wm_dseg_nii", required=True)
    parser.add_argument("--fsnative2T1w_xfm", required=True)
    parser.add_argument("--wm_probseg_nii", required=True)
    parser.add_argument("--gm_probseg_nii", required=True)
    parser.add_argument("--csf_probseg_nii", required=True)
    args = parser.parse_args()

    out_dir = Path(args.bold_preprocess_path) / args.subject_id / 'anat'
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd(args.t1_mgz, args.mask_mgz, args.aseg_mgz, args.t1_nii, args.mask_nii, args.wm_dseg_nii,
        args.fsnative2T1w_xfm, args.wm_probseg_nii, args.gm_probseg_nii, args.csf_probseg_nii)