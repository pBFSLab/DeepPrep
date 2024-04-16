#! /usr/bin/env python3
import os
import argparse
from pathlib import Path
from nipype.interfaces import fsl
import ants
import numpy as np


def get_pve_file(subjects_dir, work_dir, subject_id, wm_probseg_nii, gm_probseg_nii, csf_probseg_nii):
    split_bold_dir = Path(work_dir) / subject_id / 'fsl_fast'
    split_bold_dir.mkdir(exist_ok=True, parents=True)

    brain_mgz = Path(subjects_dir) / subject_id / 'mri' / 'brain.mgz'
    brain_nii = split_bold_dir / 'brain.nii.gz'
    os.system(f"mri_convert {brain_mgz} {brain_nii}")

    fast = fsl.FAST(segments=True, no_bias=True, probability_maps=False)
    fast.inputs.in_files = brain_nii
    fast.inputs.out_basename = 'fast_'
    out = fast.run()
    for pve in out.outputs.partial_volume_files:
        if str(pve).endswith('pve_0.nii.gz'):
            cmd = f'rsync -arv {pve} {csf_probseg_nii}'
            os.system(cmd)
        elif str(pve).endswith('pve_1.nii.gz'):
            cmd = f'rsync -arv {pve} {gm_probseg_nii}'
            os.system(cmd)
        elif str(pve).endswith('pve_2.nii.gz'):
            cmd = f'rsync -arv {pve} {wm_probseg_nii}'
            os.system(cmd)
    assert (Path(csf_probseg_nii).exists(), Path(gm_probseg_nii).exists(), Path(wm_probseg_nii).exists())

def get_seg_file(wm_dseg_nii, wm_probseg_nii):
    data = ants.image_read(wm_probseg_nii)
    data_origin = data.origin[:3]
    data_spacing = data.spacing[:3]
    data_direction = data.direction[:3, :3].copy()
    data_np = data.numpy()
    # get WM dseg & probseg
    data_wm = np.where((data_np != 0), 2, 0)
    data_wm = data_wm.astype(np.float32)
    data_wm = ants.from_numpy(data_wm, data_origin, data_spacing, data_direction.copy())
    ants.image_write(data_wm, wm_dseg_nii)
    assert Path(wm_dseg_nii).exists()

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

def cmd(subjects_dir, work_dir, subject_id, t1_mgz, mask_mgz, t1_nii, mask_nii, wm_dseg_nii, fsnative2T1w_xfm, wm_probseg_nii, gm_probseg_nii, csf_probseg_nii):
    os.system(f"mri_convert {t1_mgz} {t1_nii}")
    os.system(f"mri_binarize --i {mask_mgz} --o {mask_nii} --min 0.0001")

    get_pve_file(subjects_dir, work_dir, subject_id, wm_probseg_nii, gm_probseg_nii, csf_probseg_nii)
    get_seg_file(wm_dseg_nii, wm_probseg_nii)
    get_fsnative2T1w_xfm(fsnative2T1w_xfm)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="DeepPrep: Prepare input data for BOLD process"
    )

    parser.add_argument("--bold_preprocess_path", required=True)
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--t1_mgz", required=True)
    parser.add_argument("--mask_mgz", required=True)
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

    cmd(args.subjects_dir, args.work_dir, args.subject_id, args.t1_mgz, args.mask_mgz, args.t1_nii, args.mask_nii,
        args.wm_dseg_nii, args.fsnative2T1w_xfm, args.wm_probseg_nii, args.gm_probseg_nii, args.csf_probseg_nii)