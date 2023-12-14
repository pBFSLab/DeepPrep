#! /usr/bin/env python3
import sys
import sh
from pathlib import Path
import argparse
import os


def cmd(subj_func_dir: Path, mc: Path, bbregister_dat: Path, subjects_dir: Path, subject_id: str):
    mov = mc
    reg = bbregister_dat

    # project aparc+aseg to mc
    seg = Path(subjects_dir) / subject_id / 'mri' / 'aparc+aseg.mgz'  # Recon
    func = subj_func_dir / mc.name.replace('bold.nii.gz', 'aseg.nii.gz')
    wm = subj_func_dir / mc.name.replace('bold.nii.gz', 'wm.nii.gz')
    vent = subj_func_dir / mc.name.replace('bold.nii.gz', 'ventricles.nii.gz')
    csf = subj_func_dir / mc.name.replace('bold.nii.gz', 'csf.nii.gz')
    # project brainmask.mgz to mc
    targ = Path(subjects_dir) / subject_id / 'mri' / 'brainmask.mgz'  # Recon
    mask = subj_func_dir / mc.name.replace('bold.nii.gz', 'brainmask.nii.gz')
    binmask = subj_func_dir / mc.name.replace('bold.nii.gz', 'brainmask.bin.nii.gz')

    shargs = [
        '--seg', seg,
        '--temp', mov,
        '--reg', reg,
        '--o', func]
    sh.mri_label2vol(*shargs, _out=sys.stdout)

    shargs = [
        '--i', func,
        '--wm',
        '--erode', 1,
        '--o', wm]
    sh.mri_binarize(*shargs, _out=sys.stdout)

    shargs = [
        '--i', func,
        '--min', 24,
        '--max', 24,
        '--o', csf]
    sh.mri_binarize(*shargs, _out=sys.stdout)

    shargs = [
        '--i', func,
        '--ventricles',
        '--o', vent]
    sh.mri_binarize(*shargs, _out=sys.stdout)

    shargs = [
        '--reg', reg,
        '--targ', targ,
        '--mov', mov,
        '--inv',
        '--o', mask]
    sh.mri_vol2vol(*shargs, _out=sys.stdout)

    shargs = [
        '--i', mask,
        '--o', binmask,
        '--min', 0.0001]
    sh.mri_binarize(*shargs, _out=sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- MKbrainmask"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--mc", required=True)
    parser.add_argument("--bbregister_dat", required=True)
    # parser.add_argument("--aparaseg", required=True)
    parser.add_argument("--bold_id", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    mc_file = subj_func_dir / os.path.basename(args.mc)
    bbregister_dat = subj_func_dir / os.path.basename(args.bbregister_dat)
    cmd(subj_func_dir, mc_file, bbregister_dat, args.subjects_dir, args.subject_id)
