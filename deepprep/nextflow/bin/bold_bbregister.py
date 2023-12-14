#! /usr/bin/env python3
import sys
import sh
from pathlib import Path
import argparse
import os


def cmd(subj_func_dir: Path, mov: Path, reg: Path, subject_id: str, subjects_dir):

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

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--mc", required=True)
    parser.add_argument("--bold_id", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    mov = subj_func_dir / os.path.basename(args.mc)

    reg = subj_func_dir / f'{args.bold_id}_from-mc_to-T1w_desc-rigid_xfm.dat'  # output
    cmd(subj_func_dir, mov, reg, args.subject_id, args.subjects_dir)
