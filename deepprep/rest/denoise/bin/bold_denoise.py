#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import os
import argparse
from bids import BIDSLayout

from filters.filters import bandpass_nifti
from regressors.regressors import glm_nifti
import json


def set_environ(freesurfer_home, subjects_dir):
    # FreeSurfer recon-all env
    os.environ['FREESURFER_HOME'] = freesurfer_home
    if subjects_dir:
        os.environ['SUBJECTS_DIR'] = subjects_dir
    else:
        os.environ['SUBJECTS_DIR'] = f'{freesurfer_home}/subjects'
    os.environ['PATH'] = f'{freesurfer_home}/bin:{freesurfer_home}/mni/bin:{freesurfer_home}/tktools:' + \
                         f'{freesurfer_home}/fsfast/bin:' + os.environ['PATH']
    os.environ['MINC_BIN_DIR'] = f'{freesurfer_home}/mni/bin'
    os.environ['MINC_LIB_DIR'] = f'{freesurfer_home}/mni/lib'
    os.environ['PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
    os.environ['MNI_PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
    # FreeSurfer fsfast env
    os.environ['FSF_OUTPUT_FORMAT'] = 'nii.gz'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'


def fsaverageN_smN(input_path, output_path, hemi, space, fwhm=6):
    cmd = ' '.join([
                    'mri_surf2surf',
                    '--hemi', hemi,
                    '--s', space,
                    '--sval', input_path,
                    '--label-src', hemi + '.cortex.label',
                    '--fwhm', str(fwhm),
                    '--tval', output_path,
                    '--reshape'
                    ])
    print(f'run: {cmd}')
    os.system(cmd)
    assert os.path.exists(output_path)


def volume_smooth(input_path, output_path, fwhm=6):
    cmd = ' '.join([
                    'mri_fwhm',
                    '--i', input_path,
                    f"--to-fwhm {fwhm} ",
                    f"--o {output_path} ",
                    f"--to-fwhm-tol 0.5"
                    ])
    print(f'run: {cmd}')
    os.system(cmd)
    assert os.path.exists(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: denise"
    )

    # input
    parser.add_argument("--bold_preprocess_dir", required=True, help="DeepPrep preprocessed BOLD dir")
    parser.add_argument("--bold_preproc_file", required=True, help="preprocessed BOLD file json")
    parser.add_argument("--confounds_index_file", required=True, help="confounds_index_file")
    parser.add_argument("--repetition_time", required=False, help="RepetitionTime of BOLD file (optional)")
    parser.add_argument("--freesurfer_home", required=False, help="freesurfer home path (optional)")
    parser.add_argument("--subjects_dir", required=False, help="DeepPrep Recon dir is required if the space of BOLD is fsnative (optional)")
    parser.add_argument("--skip_frame", required=False, help="how many frames to skip before postprocessing? (optional)")
    parser.add_argument("--bandpass", required=True, help="BOLD bandpass range, default '0.01-0.08'")
    parser.add_argument("--surface_fwhm", required=False, help="(INT) using to smoothing file with fwhm (optional)", default=0)
    parser.add_argument("--volume_fwhm", required=False, help="(INT) using to smoothing file with fwhm (optional)", default=0)
    # output
    parser.add_argument("--output_dir", required=True, help="denoised dir path")
    parser.add_argument("--work_dir", required=True, help="work dir path")
    args = parser.parse_args()

    if args.freesurfer_home:
        set_environ(os.path.join(args.freesurfer_home), args.subjects_dir)

    assert os.environ.get('FREESURFER_HOME', None), "FREESURFER_HOME environment variable not set, please add --freesurfer_home /path/to/freesurfer"

    assert os.path.isdir(args.bold_preprocess_dir)
    assert os.path.isfile(os.path.join(args.bold_preprocess_dir, 'dataset_description.json'))
    assert os.path.isfile(args.bold_preproc_file)
    assert os.path.isfile(args.confounds_index_file)

    bandpass_low, bandpass_high = args.bandpass.split('-')
    bandpass_low = float(bandpass_low)
    bandpass_high = float(bandpass_high)
    assert bandpass_low > 0
    assert bandpass_high > 0
    band = [bandpass_low, bandpass_high]

    confounds_index_file = args.confounds_index_file

    with open(args.bold_preproc_file, 'r') as f:
        bold_preproc_json = json.load(f)
    bold_preproc_file = bold_preproc_json.get('bold_file', None)
    bids_database_path = bold_preproc_json.get('bids_database_path', None)
    space = bold_preproc_json.get('space', None)
    subject_id = bold_preproc_json.get('subject_id', None)
    assert os.path.isfile(bold_preproc_file)

    layout_pre = BIDSLayout(args.bold_preprocess_dir, validate=False)
    bold_preproc_file = layout_pre.get_file(bold_preproc_file)

    if layout_pre.get_metadata(bold_preproc_file.path).get('RepetitionTime', None):
        TR = layout_pre.get_metadata(bold_preproc_file.path)['RepetitionTime']
    elif args.repetition_time:
        TR = float(args.repetition_time)
    else:
        raise ValueError("Cant found TR info in metadata file, please set the TR parameter: --repetition_time <TR>")

    entities = layout_pre.parse_file_entities(bold_preproc_file.path)

    confounds_entities = {'desc': 'confounds', 'extension': '.tsv'}
    for entity in ['subject', 'session', 'run', 'task']:
        if entities.get(entity, None):
            confounds_entities[entity] = entities.get(entity)
    confounds_file = layout_pre.get(**confounds_entities)[0]

    # output
    output_dir = os.path.join(args.output_dir, os.path.dirname(bold_preproc_file.relpath))
    os.makedirs(output_dir, exist_ok=True)

    if entities['extension'] == '.func.gii':
        # output
        nii_file = os.path.join(output_dir, bold_preproc_file.filename.replace('func.gii', 'nii.gz'))
        nii_file_name = os.path.basename(nii_file)
        bandpass_file = os.path.join(output_dir, nii_file_name.replace(f'_bold', f'_desc-bandpass_bold'))
        bandpass_json_file = bandpass_file.replace('.nii.gz', '.json')
        regression_file = os.path.join(output_dir, nii_file_name.replace(f'_bold', f'_desc-regression_bold'))
        regression_json_file = regression_file.replace('.nii.gz', '.json')
        smooth_file = os.path.join(output_dir, nii_file_name.replace(f'_bold', f'_desc-fwhm_bold'))
        smooth_json_file = smooth_file.replace('.nii.gz', '.json')

        def to_nii(_in_file, _output_file):
            _cmd = f'mri_convert {_in_file} {_output_file}'
            os.system(_cmd)

        to_nii(bold_preproc_file.path, nii_file)
        assert os.path.exists(nii_file)
        print(f'>>> {nii_file}')

        bandpass_nifti(nii_file, bandpass_file, TR, int(args.skip_frame), band)
        assert os.path.exists(bandpass_file)
        print(f'>>> {bandpass_file}')
        with open(bandpass_json_file, 'w') as f:
            json.dump({'RepetitionTime': TR, 'bandpass': band}, f, indent=4)
        print(f'>>> {bandpass_json_file}')

        glm_nifti(confounds_index_file, bandpass_file, regression_file, confounds_file)
        assert os.path.exists(regression_file)
        print(f'>>> {regression_file}')
        with open(confounds_index_file, 'r') as f:
            confounds = f.readlines()
            confounds = [i.strip() for i in confounds]
        with open(regression_json_file, 'w') as f:
            json.dump({'RepetitionTime': TR, 'confounds': confounds}, f, indent=4)
        print(f'>>> {regression_json_file}')

        if 'hemi-L' in os.path.basename(smooth_file):
            hemi = 'lh'
        elif 'hemi-R' in os.path.basename(smooth_file):
            hemi = 'rh'
        else:
            raise ValueError(os.path.basename(smooth_file))

        fwhm = int(args.surface_fwhm)
        if fwhm > 0:
            if space == 'fsnative':
                os.environ['SUBJECTS_DIR'] = args.subjects_dir
                fsaverageN_smN(regression_file, smooth_file, hemi, subject_id, fwhm)
            else:
                fsaverageN_smN(regression_file, smooth_file, hemi, space, fwhm)
            assert os.path.exists(smooth_file)
            print(f'>>> {smooth_file}')
            with open(smooth_json_file, 'w') as f:
                json.dump({'RepetitionTime': TR, 'fwhm': fwhm}, f, indent=4)
            print(f'>>> {smooth_json_file}')

    elif entities['extension'] == '.nii.gz':
        bandpass_file = os.path.join(output_dir, bold_preproc_file.filename.replace('desc-preproc', 'desc-bandpass'))
        bandpass_json_file = bandpass_file.replace('.nii.gz', '.json')
        regression_file = os.path.join(output_dir, bold_preproc_file.filename.replace('desc-preproc', 'desc-regression'))
        regression_json_file = regression_file.replace('.nii.gz', '.json')
        smooth_file = os.path.join(output_dir, bold_preproc_file.filename.replace('desc-preproc', 'desc-fwhm'))
        smooth_json_file = smooth_file.replace('.nii.gz', '.json')

        bandpass_nifti(bold_preproc_file, bandpass_file, TR)
        assert os.path.exists(bandpass_file)
        print(f'>>> {bandpass_file}')
        with open(bandpass_json_file, 'w') as f:
            json.dump({'RepetitionTime': TR, 'bandpass': band}, f, indent=4)
        print(f'>>> {bandpass_json_file}')

        glm_nifti(confounds_index_file, bandpass_file, regression_file, confounds_file)
        assert os.path.exists(regression_file)
        print(f'>>> {regression_file}')
        with open(confounds_index_file, 'r') as f:
            confounds = f.readlines()
        with open(regression_json_file, 'w') as f:
            json.dump({'RepetitionTime': TR, 'confounds': confounds}, f, indent=4)
        print(f'>>> {regression_json_file}')

        fwhm = int(args.volume_fwhm)
        if fwhm > 0:
            volume_smooth(regression_file, smooth_file, fwhm)
            assert os.path.exists(smooth_file)
            print(f'>>> {smooth_file}')
            with open(smooth_json_file, 'w') as f:
                json.dump({'RepetitionTime': TR, 'fwhm': fwhm}, f, indent=4)
            print(f'>>> {smooth_json_file}')
    else:
        raise KeyError
