#! /usr/bin/env python3
import sys
import sh
from pathlib import Path
import argparse
import os

def cmd(subj_func_dir: Path, bold: Path, subject_id: str, subjects_dir):
    # mov = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_stc_mc.nii.gz')
    mov = bold
    reg = subj_func_dir / bold.name.replace('.nii.gz',
                                            '_from_mc_to_fsnative_bbregister_rigid.dat')
    os.environ["SUBJECTS_DIR"] = subjects_dir
    print(os.environ["SUBJECTS_DIR"])
    # exit()
    shargs = [
        '--bold',
        '--s', subject_id,
        '--mov', mov,
        '--reg', reg]
    sh.bbregister(*shargs, _out=sys.stdout)

    DEBUG = False
    if not DEBUG:
        (subj_func_dir / reg.name.replace('.dat', '.dat.mincost')).unlink(missing_ok=True)
        (subj_func_dir / reg.name.replace('.dat', '.dat.param')).unlink(missing_ok=True)
        (subj_func_dir / reg.name.replace('.dat', '.dat.sum')).unlink(missing_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- bbregister"
    )

    parser.add_argument("--bold_preproces_dir", required=True)
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--mc", required=True)
    parser.add_argument("--bold_id", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir  = Path(cur_path) / str(args.bold_preproces_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    mc_file = subj_func_dir / f'{args.bold_id}_skip_reorient_stc_mc.nii.gz'
    cmd(subj_func_dir, mc_file, args.subject_id, args.subjects_dir)