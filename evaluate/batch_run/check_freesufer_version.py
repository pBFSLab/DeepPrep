import os
from pathlib import Path
import shutil

if __name__ == '__main__':
    data_path = Path('/mnt/nfs/output')
    save_path = Path('/mnt/nfs/FreeSurfer720')
    subjs = os.listdir(data_path)
    stamp_list = list()
    rerun_sbujs = list()
    for subj in subjs:
        build_stamp_file = data_path / subj / 'scripts' / 'build-stamp.txt'
        with open(build_stamp_file) as f:
            line = f.read()
        freesurfer_version = line.strip()
        stamp_list.append(freesurfer_version)
        if freesurfer_version != 'freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0-2beb96c':
            rerun_sbujs.append(subj)

    for idx, subj in enumerate(rerun_sbujs):
        print(f'{idx}/{len(rerun_sbujs)}: {subj}')
        shutil.move(data_path / subj, save_path / subj)
