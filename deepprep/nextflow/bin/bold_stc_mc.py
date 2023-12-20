#! /usr/bin/env python3
import sys
import sh
from pathlib import Path
import argparse
import os
import shutil


def cmd(subj_func_dir: Path, subj_tmp_dir: Path, skip_reorient: Path, run: str, subject_id, bold_id: str):

    # create tmp dir
    tmp_run = subj_tmp_dir / 'stc_mc' / bold_id
    if tmp_run.exists():
        shutil.rmtree(tmp_run)
    link_dir = tmp_run / subject_id / 'bold' / run
    if not link_dir.exists():
        link_dir.mkdir(parents=True, exist_ok=True)
    src_file = skip_reorient
    dst_file = link_dir / skip_reorient.name
    dst_file.symlink_to(src_file)

    # STC
    input_fname = skip_reorient.name.replace('.nii.gz', '')
    stc_fname = f'{bold_id}_space-stc_bold'
    mc_fname = f'{bold_id}_space-mc_bold'
    shargs = [
        '-s', subject_id,
        '-d', tmp_run,
        '-fsd', 'bold',
        '-so', 'odd',
        '-ngroups', 1,
        '-i', input_fname,
        '-o', stc_fname,
        '-nolog']
    sh.stc_sess(*shargs, _out=sys.stdout)


    # """
    # mktemplate-sess 会生成两个template
    # 1. 一个放到 bold/template.nii.gz，使用的是run 001的first frame，供mc-sess --per-session使用
    # 2. 一个放到 bold/run/template.nii.gz 使用的是每个run的mid frame，供mc-sess --per-run参数使用(default)
    # """
    # shargs = [
    #     '-s', subject_id,
    #     '-d', tmp_run,
    #     '-fsd', 'bold',
    #     '-funcstem', faln_fname,
    #     '-nolog']
    # sh.mktemplate_sess(*shargs, _out=sys.stdout)
    boldref = Path(subj_func_dir) / f'{subject_id}_boldref.nii.gz'
    shutil.copyfile(boldref, link_dir / 'template.nii.gz')

    # Mc
    shargs = [
        '-s', subject_id,
        '-d', tmp_run,
        '-per-run',
        '-fsd', 'bold',
        '-fstem', stc_fname,
        '-fmcstem', mc_fname,
        '-nolog']
    sh.mc_sess(*shargs, _out=sys.stdout)

    # Stc
    shutil.copyfile(link_dir / f'{stc_fname}.nii.gz.log',
                    subj_func_dir / f'{bold_id}_from-scanner_to-stc_xfm.log')

    # Template reference for mc
    # shutil.copyfile(link_dir / 'template.nii.gz',
    #                 ori_path / f'{mc_fname}ref.nii.gz')  # _space-mc_boldref.nii.gz
    # shutil.copyfile(link_dir / 'template.log',
    #                 ori_path / f'{mc_fname}ref.log')

    # Mc
    mc_bold_file = subj_func_dir / f'{mc_fname}.nii.gz'
    shutil.move(link_dir / f'{mc_fname}.nii.gz', mc_bold_file)

    mc_boldref_file = subj_func_dir / mc_bold_file.name.replace('bold.nii.gz', 'boldref.nii.gz')
    cmd = f'mri_convert {mc_bold_file} {mc_boldref_file} --frame 0'
    os.system(cmd)

    shutil.move(link_dir / f'{mc_fname}.mat.aff12.1D',
                subj_func_dir / f'{bold_id}_from-stc_to-mc_xfm.mat.aff12.1D')
    shutil.move(link_dir / f'{mc_fname}.mcdat',
                subj_func_dir / f'{bold_id}_from-stc_to-mc_xfm.mcdat')


    DEBUG = False
    if not DEBUG:
        shutil.rmtree(tmp_run, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- STC"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--reorient", required=True)  # _space-reorient_bold.nii.gz
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)
    subj_tmp_dir = Path(preprocess_dir) / 'tmp'
    subj_tmp_dir.mkdir(parents=True, exist_ok=True)

    skip_reorient_file = subj_func_dir / os.path.basename(args.reorient)

    run = '001'
    cmd(subj_func_dir, subj_tmp_dir, skip_reorient_file, run, args.subject_id, args.bold_id)
