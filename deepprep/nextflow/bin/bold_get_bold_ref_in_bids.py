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
        print(f"Successfully reorient {input_path} to RAS orientation and saved to {output_path}.")


def cmd(subj_func_dir: Path, bold: Path, run: str, subject_id, bold_name: str):
    boldref = Path(subj_func_dir) / Path(bold).name.replace('.nii.gz', '_boldref.nii.gz')
    reorient_bold = Path(subj_func_dir) / Path(boldref).name.replace('.nii.gz', '_reorient.nii.gz')
    open(f'{subj_func_dir}/{subject_id}_boldref.log', 'w').write(str(bold))

    # boldref
    cmd = f'mri_convert {bold} {boldref} --frame 0'
    os.system(cmd)

    # reorient
    reorient_to_ras(boldref, reorient_bold)

    ori_path = subj_func_dir
    tmp_run = subj_func_dir / bold_name / run
    if tmp_run.exists():
        shutil.rmtree(tmp_run)
    link_dir = tmp_run / subject_id / 'bold' / run
    if not link_dir.exists():
        link_dir.mkdir(parents=True, exist_ok=True)
    link_files = os.listdir(subj_func_dir)
    nii_files = [file for file in link_files if file.endswith('_boldref_reorient.nii.gz')]
    for link_file in nii_files:
        try:
            src_file = subj_func_dir / link_file
            dst_file = link_dir / link_file
            dst_file.symlink_to(src_file)
        except:
            continue

    # STC
    faln_fname = reorient_bold.name.replace('_reorient.nii.gz', '_reorient')
    stc_fname = reorient_bold.name.replace('_reorient', '_reorient_stc')
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
    shutil.move(link_dir / f'{stc_fname}.nii.gz',
                ori_path / f'{subject_id}_boldref.nii.gz')
    cmd = f'rm -rf {boldref}'
    os.system(cmd)
    cmd = f'rm -rf {reorient_bold}'
    os.system(cmd)

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

    bids_dir = Path(cur_path) / str(args.bids_dir) / args.subject_id
    session = args.bold_id.split('_')[1]
    bold_file = bids_dir / session / 'func' / f'{args.bold_id}.nii.gz'
    run = '001'
    cmd(subj_func_dir, bold_file, run, args.subject_id, args.bold_id)


