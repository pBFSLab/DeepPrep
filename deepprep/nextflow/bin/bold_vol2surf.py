#! /usr/bin/env python3
import os
import sh
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import surface


def vol2surf(vol, hemi_pial, hemi_white, hemi_w_g_pct, hemi_fsnative_surf_output):
    img = nib.load(vol)
    surf_data = surface.vol_to_surf(img,
                                    surf_mesh=hemi_pial,
                                    inner_mesh=hemi_white)
    surf_data[np.isnan(surf_data)] = 0
    hemi_template = nib.load(hemi_w_g_pct)
    tarimg = nib.Nifti1Image(surf_data.reshape([surf_data.shape[0], 1, 1, surf_data.shape[1]]), hemi_template.affine,
                             hemi_template.header)
    nib.save(tarimg, hemi_fsnative_surf_output)


def surf2surf(hemi_fsnative_surf_output, subjects_dir, freesurfer_home, subject_id, hemi, trgsubject,
              hemi_fsaverage_surf_output):
    os.environ['FREESURFER_HOME'] = freesurfer_home
    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['PATH'] = f'{args.freesurfer_home}/bin:' + os.environ['PATH']

    cmd = f'mri_surf2surf --hemi {hemi} --srcsurfval {hemi_fsnative_surf_output} --srcsubject {subject_id} --trgsubject {trgsubject} --trgsurfval {hemi_fsaverage_surf_output}'
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- vol2surf"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument('--subjects_dir', type=str, help='subjects_dir')
    parser.add_argument('--freesurfer_home', type=str, help='freesurfer_home')
    parser.add_argument("--hemi_white", required=True)
    parser.add_argument("--hemi_pial", required=True)
    parser.add_argument("--hemi_w_g_pct_mgh", required=True)
    parser.add_argument("--bbregister_native_2mm", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--trgsubject", default='fsaverage6')
    parser.add_argument("--hemi_fsnative_surf_output", required=True)
    parser.add_argument("--hemi_fsaverage_surf_output", required=True)
    args = parser.parse_args()

    preprocess_dir = Path(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'

    hemi_white = Path(args.subjects_dir) / args.subject_id / 'surf' / os.path.basename(args.hemi_white)
    hemi_pial = Path(args.subjects_dir) / args.subject_id / 'surf' / os.path.basename(args.hemi_pial)
    hemi_w_g_pct = Path(args.subjects_dir) / args.subject_id / 'surf' / os.path.basename(args.hemi_w_g_pct_mgh)
    bbregister_native_2mm = subj_func_dir / os.path.basename(args.bbregister_native_2mm)

    hemi = str(os.path.basename(args.hemi_white)).split('.')[0]
    hemi_fsnative_surf_output = subj_func_dir / os.path.basename(args.hemi_fsnative_surf_output)
    hemi_fsaverage_surf_output = subj_func_dir / os.path.basename(args.hemi_fsaverage_surf_output)

    trgsubject_dir = Path(args.subjects_dir) / args.trgsubject
    if not trgsubject_dir.exists():
        os.system(f"ln -sf {Path(args.freesurfer_home) / f'subjects/{args.trgsubject}'} {trgsubject_dir}")
    vol2surf(bbregister_native_2mm, hemi_pial, hemi_white, hemi_w_g_pct, hemi_fsnative_surf_output)
    surf2surf(hemi_fsnative_surf_output, args.subjects_dir, args.freesurfer_home, args.subject_id, hemi,
              args.trgsubject, hemi_fsaverage_surf_output)
