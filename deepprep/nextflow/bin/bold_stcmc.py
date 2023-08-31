#! /usr/bin/env python3
import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import os
import shutil


def cmd(subj_func_dir: Path, skip_reorient: Path, run: str, subject_id, bold_name: str):
    tmp_run = subj_func_dir / bold_name / run
    if tmp_run.exists():
        shutil.rmtree(tmp_run)
    link_dir = tmp_run / subject_id / 'bold' / run
    if not link_dir.exists():
        link_dir.mkdir(parents=True, exist_ok=True)
    link_files = os.listdir(subj_func_dir)
    link_files.remove(bold_name)
    for link_file in link_files:
        try:
            src_file = subj_func_dir / link_file
            dst_file = link_dir / link_file
            dst_file.symlink_to(src_file)
        except:
            continue

    # STC
    input_fname = skip_reorient.name.replace('_skip_reorient.nii.gz', '_skip_reorient')
    faln_fname = skip_reorient.name.replace('_skip_reorient.nii.gz', '_skip_reorient_stc')
    mc_fname = skip_reorient.name.replace('_skip_reorient.nii.gz', '_skip_reorient_stc_mc')
    shargs = [
        '-s', subject_id,
        '-d', tmp_run,
        '-fsd', 'bold',
        '-so', 'odd',
        '-ngroups', 1,
        '-i', input_fname,
        '-o', faln_fname,
        '-nolog']
    sh.stc_sess(*shargs, _out=sys.stdout)

    """
    mktemplate-sess 会生成两个template
    1. 一个放到 bold/template.nii.gz，使用的是run 001的first frame，供mc-sess --per-session使用
    2. 一个放到 bold/run/template.nii.gz 使用的是每个run的mid frame，供mc-sess --per-run参数使用(default)
    """
    shargs = [
        '-s', subject_id,
        '-d', tmp_run,
        '-fsd', 'bold',
        '-funcstem', faln_fname,
        '-nolog']
    sh.mktemplate_sess(*shargs, _out=sys.stdout)

    # Mc
    shargs = [
        '-s', subject_id,
        '-d', tmp_run,
        '-per-run',
        '-fsd', 'bold',
        '-fstem', faln_fname,
        '-fmcstem', mc_fname,
        '-nolog']
    sh.mc_sess(*shargs, _out=sys.stdout)

    ori_path = subj_func_dir
    try:
        # Stc
        DEBUG = False
        if DEBUG:
            shutil.move(link_dir / f'{faln_fname}.nii.gz',
                        ori_path / f'{faln_fname}.nii.gz')
            shutil.move(link_dir / f'{faln_fname}.nii.gz.log',
                        ori_path / f'{faln_fname}.nii.gz.log')
        else:
            (ori_path / f'{input_fname}.nii.gz').unlink(missing_ok=True)

        # Template reference for mc
        shutil.copyfile(link_dir / 'template.nii.gz',
                        ori_path / f'{faln_fname}_boldref.nii.gz')
        shutil.copyfile(link_dir / 'template.log',
                        ori_path / f'{faln_fname}_boldref.log')

        # Mc
        shutil.move(link_dir / f'{mc_fname}.nii.gz',
                    ori_path / f'{mc_fname}.nii.gz')

        shutil.move(link_dir / f'{mc_fname}.mcdat',
                    ori_path / f'{mc_fname}.mcdat')

        if DEBUG:
            shutil.move(link_dir / f'{mc_fname}.mat.aff12.1D',
                        ori_path / f'{mc_fname}.mat.aff12.1D')
            shutil.move(link_dir / f'{mc_fname}.nii.gz.mclog',
                        ori_path / f'{mc_fname}.nii.gz.mclog')
            shutil.move(link_dir / 'mcextreg',
                        ori_path / f'{mc_fname}.mcextreg')
            shutil.move(link_dir / 'mcdat2extreg.log',
                        ori_path / f'{mc_fname}.mcdat2extreg.log')
    except:
        pass
    shutil.rmtree(ori_path / bold_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- STC"
    )

    # parser.add_argument("--cur_path", required=True)
    parser.add_argument("--bold_preproces_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--skip_reorient", required=True)
    parser.add_argument("--bold_id", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir  = Path(cur_path) / str(args.bold_preproces_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    skip_reorient_file = subj_func_dir / f'{args.bold_id}_skip_reorient.nii.gz'

    run = '001'
    cmd(subj_func_dir, skip_reorient_file, run, args.subject_id, args.bold_id)

