import os
from pathlib import Path
import json
import zipfile
import bids

if __name__ == '__main__':
    data_path = Path('/home/weiwei/workspace/DeepPrep/app_pipeline/data/NC_15/upload')
    save_path = Path('/home/weiwei/workdata/DeepPrep/BoldPipeline/TestData')

    dataset_description = dict()
    dataset_description['Name'] = 'DeepPrep/test/V001'
    dataset_description['BIDSVersion'] = '1.4.0'

    dataset_description_file = save_path / 'dataset_description.json'
    with open(dataset_description_file, 'w') as jf:
        json.dump(dataset_description, jf, indent=4)

    layout = bids.BIDSLayout(save_path)
    subjs = ['NC_15']

    # freesurfer
    derivative_freesurfer_path = save_path / 'derivatives' / 'freesurfer'
    derivative_freesurfer_path.mkdir(parents=True, exist_ok=True)
    dataset_description = dict()
    dataset_description['Name'] = 'FreeSurfer Outputs'
    dataset_description['BIDSVersion'] = '1.4.0'
    dataset_description['DatasetType'] = 'derivative'
    dataset_description['GeneratedBy'] = [{'Name': 'freesurfer', 'Version': '6.0.0'}]
    dataset_description_file = derivative_freesurfer_path / 'dataset_description.json'
    with open(dataset_description_file, 'w') as jf:
        json.dump(dataset_description, jf, indent=4)

    for idx, subj in enumerate(subjs):
        print(f'{idx}/{len(subjs)} subj: {subj}')
        runs = os.listdir(data_path / subj / 'anat')
        for run in runs:
            t1_file_path = data_path / subj / 'anat' / run / f'{subj}_mpr{run}.nii.gz'
            entities = dict()
            entities['subject'] = f'{idx + 1:03}'
            entities['session'] = '01'
            entities['datatype'] = 'anat'
            entities['suffix'] = 'T1w'
            layout.write_to_file(entities, copy_from=t1_file_path)
        runs = os.listdir(data_path / subj / 'bold')
        for run in runs:
            bold_file_path = data_path / subj / 'bold' / run / f'{subj}_bld{run}_rest.nii.gz'
            entities = dict()
            entities['subject'] = f'{idx + 1:03}'
            entities['session'] = '01'
            entities['run'] = run
            entities['datatype'] = 'func'
            entities['task'] = 'rest'
            entities['suffix'] = 'bold'
            layout.write_to_file(entities, copy_from=bold_file_path)
        # freesurfer
        recon_all_file = data_path / subj / f'{subj}_reconall.zip'
        with zipfile.ZipFile(recon_all_file) as zf:
            zf.extractall(derivative_freesurfer_path / f'sub-{idx + 1:03}')
