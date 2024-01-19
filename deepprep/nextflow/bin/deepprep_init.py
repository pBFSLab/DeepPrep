#! /usr/bin/env python3
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: init"
    )
    parser.add_argument('--freesurfer_home', required=True, help="directory of freesurfer fsaverage")
    parser.add_argument("--subjects_dir", required=True, help="directory of Recon results")
    parser.add_argument("--bold_spaces", type=str, nargs='+', required=True, help="type of bold space outputs")
    args = parser.parse_args()

    spaces = ' '.join(args.bold_spaces)
    subjects_dir = args.subjects_dir
    freesurfer_fsaverage6_dir = os.path.join(args.freesurfer_home, 'subjects', 'fsaverage6')
    freesurfer_fsaverage_dir = os.path.join(args.freesurfer_home, 'subjects', 'fsaverage')

    if os.path.exists(f'{subjects_dir}/fsaverage'):
        os.system(f'rm -r {subjects_dir}/fsaverage')
    os.system(f'cp -r {freesurfer_fsaverage_dir} {subjects_dir}')
    if 'fsaverage6' in spaces:
        if os.path.exists(f'{subjects_dir}/fsaverage6'):
            os.system(f'rm -r {subjects_dir}/fsaverage6')
        os.system(f'cp -r {freesurfer_fsaverage6_dir} {subjects_dir}')
