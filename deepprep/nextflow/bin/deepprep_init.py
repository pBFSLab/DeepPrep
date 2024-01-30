#! /usr/bin/env python3
import os
import argparse
from pathlib import Path
import json


def create_dataset_description(dataset_path: Path):
    descriptions_info_qc = {
        "Name": "DeepPrep - MRI PREProcessing workflow",
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "DeepPrep",
                "Version": "",
                "CodeURL": ""
            }
        ],
        "HowToAcknowledge": "Please cite our paper , and include the generated citation boilerplate within the Methods section of the text."
    }
    dataset_description_file = dataset_path / 'dataset_description.json'
    if not dataset_description_file.exists():
        with open(dataset_description_file, 'w') as jf_config:
            json.dump(descriptions_info_qc, jf_config, indent=4)
        print(f'create DeepPrep results dataset_description.json: {dataset_description_file}')


def init_output_dir(output_dir, subjects_dir):
    output_path = Path(output_dir)
    work_dir = output_path / 'WorkDir'
    bold_preprocess_dir = output_path / 'BOLD'
    qc_result_dir = output_path / 'QC'
    subjects_dir = Path(subjects_dir)

    for dir_path in [work_dir, bold_preprocess_dir, qc_result_dir, subjects_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    create_dataset_description(bold_preprocess_dir)
    create_dataset_description(qc_result_dir)
    return subjects_dir


def init_subject_dir(subjects_dir, freesurfer_home, bold_spaces):
    freesurfer_subjects_dir = os.path.join(freesurfer_home, 'subjects')
    if 'fsaverage' not in bold_spaces:
        bold_spaces.append('fsaverage')
    for space in bold_spaces:
        if 'fsaverage' in space:
            source_path = os.path.join(freesurfer_subjects_dir, space)
            target_path = os.path.join(subjects_dir, space)
            if not os.path.exists(target_path):
                os.system(f'cp -r {source_path} {subjects_dir}')
            else:
                os.system(f'rsync -arv {source_path}/ {target_path}/')

    # spaces = ' '.join(bold_spaces)
    # freesurfer_fsaverage6_dir = os.path.join(freesurfer_home, 'subjects', 'fsaverage6')
    # freesurfer_fsaverage_dir = os.path.join(freesurfer_home, 'subjects', 'fsaverage')
    # if os.path.exists(f'{subjects_dir}/fsaverage'):
    #     os.system(f'rm -r {subjects_dir}/fsaverage')
    # os.system(f'cp -r {freesurfer_fsaverage_dir} {subjects_dir}')
    # if 'fsaverage6' in spaces:
    #     if os.path.exists(f'{subjects_dir}/fsaverage6'):
    #         os.system(f'rm -r {subjects_dir}/fsaverage6')
    #     os.system(f'cp -r {freesurfer_fsaverage6_dir} {subjects_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: init"
    )
    parser.add_argument('--freesurfer_home', required=True, help="directory of freesurfer fsaverage")
    parser.add_argument("--bids_dir", required=True, help="directory of bids")
    parser.add_argument("--output_dir", required=True, help="directory of results")
    parser.add_argument("--subjects_dir", required=True, help="directory of Recon results")
    parser.add_argument("--bold_spaces", type=str, nargs='+', required=True, help="type of bold space outputs")
    parser.add_argument("--bold_only", type=str, required=True, help="only run bold preprocess")
    args = parser.parse_args()

    # check path exist
    assert os.path.isdir(args.bids_dir) and len(os.listdir(args.bids_dir)) > 0, f'Please check bids_dir path: {args.bids_dir}'
    if args.bold_only.upper() == 'TRUE':
        assert os.path.isdir(args.subjects_dir) and len(os.listdir(args.subjects_dir)) > 0, f'Please check subjects_dir path: {args.subjects_dir}'

    # init output dir
    _subjects_dir = init_output_dir(args.output_dir, args.subjects_dir)
    init_subject_dir(_subjects_dir, args.freesurfer_home, args.bold_spaces)
