#! /usr/bin/env python3
import os
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: sMRI and fMRI PreProcessing workflows"
    )

    parser.add_argument("--subjects-dir", required=True)
    parser.add_argument("--t1wfile-path", required=True)
    args = parser.parse_args()

    with open(args.t1wfile_path, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    subject_id = data[0]
    output_dir = Path(args.subjects_dir) / subject_id / "mri" / "orig"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, t1w_file in enumerate(data[1:]):
        output_file = output_dir / f'{i+1:03d}.mgz'
        if not output_file.exists():
            os.system(f"mri_convert {t1w_file} {output_file}")
            print(f"mri_convert {t1w_file} {output_file}")

    with open(subject_id, 'w') as f:
        f.write(subject_id)
