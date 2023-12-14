#! /usr/bin/env python3
import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import os
import shutil


def reorient_to_ras(input_path, output_path):
    img = nib.load(input_path)
    orig_ornt = nib.orientations.io_orientation(img.header.get_sform())
    RAS_ornt = nib.orientations.axcodes2ornt('RAS')
    if np.array_equal(orig_ornt, RAS_ornt) is True:
        print(f"{input_path} is already in RAS orientation. Copying to {output_path}.")
        shutil.copy(input_path, output_path)
    else:
        newimg = img.as_reoriented(orig_ornt)
        nib.save(newimg, output_path)
        print('orig_ornt: \n', orig_ornt)
        print(f"Successfully reorient {input_path} to RAS orientation and saved to {output_path}.")


def cmd(bids_dir: Path, subj_func_dir: Path, subj_tmp_dir: Path, bold_id: str, run: str, subject_id):
    bold = Path(bids_dir) / f'{bold_id}_bold.nii.gz'
    # tmp dir
    tmp_run = subj_tmp_dir / 'get_bold_ref' / bold_id
    if tmp_run.exists():
        shutil.rmtree(tmp_run)
    link_dir = tmp_run / subject_id / 'bold' / run
    if not link_dir.exists():
        link_dir.mkdir(parents=True, exist_ok=True)

    bold_first_frame = Path(link_dir) / f'{bold_id}_desc-fframe_bold.nii.gz'
    reorient_bold = Path(link_dir) / bold_first_frame.name.replace('.nii.gz', '_reorient.nii.gz')

    # boldref
    cmd = f'mri_convert {bold} {bold_first_frame} --frame 0'
    os.system(cmd)

    # reorient
    reorient_to_ras(bold_first_frame, reorient_bold)

    # STC
    faln_fname = reorient_bold.name.replace('_reorient.nii.gz', '_reorient')
    stc_fname = reorient_bold.name.replace('_reorient.nii.gz', '_reorient_stc')
    shargs = [
        '-s', subject_id,
        '-d', tmp_run,
        '-fsd', 'bold',
        '-so', 'odd',
        '-ngroups', 1,
        '-i', faln_fname,
        '-o', stc_fname,
        '-nolog']
    sh.stc_sess(*shargs, _out=sys.stdout)

    shutil.copyfile(link_dir / f'{stc_fname}.nii.gz', subj_func_dir / f'{subject_id}_boldref.nii.gz')
    open(subj_func_dir / f'{subject_id}_boldref.log', 'w').write(f'{bold} \n-> mc \n-> stc')

    DEBUG = True
    if not DEBUG:
        shutil.rmtree(tmp_run, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- boldref"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    subj_tmp_dir = Path(preprocess_dir) / 'tmp'
    subj_tmp_dir.mkdir(parents=True, exist_ok=True)

    if 'ses-' in args.bold_id:
        session = args.bold_id.split('_')[1]
        subj_bids_dir = Path(cur_path) / str(args.bids_dir) / args.subject_id / session / 'func'
    else:
        subj_bids_dir = Path(cur_path) / str(args.bids_dir) / args.subject_id / 'func'
    run = '001'
    cmd(subj_bids_dir, subj_func_dir, subj_tmp_dir, args.bold_id, run, args.subject_id)
