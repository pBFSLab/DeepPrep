#! /usr/bin/env python3
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA

from bold_mkbrainmask import anat2bold_t1w
from filters.filters import bandpass_nifti


def qnt_nifti(data, maskpaths):
    '''

    bold_path - path. Path of bold after bpass process.
    maskpaths - Path to file containing mask.
    '''

    if isinstance(maskpaths, str):
        maskpaths = [maskpaths]

    # Open mask.
    mask = None
    nvox_sum = 0
    for maskpath in maskpaths:
        mask_img = nib.load(maskpath)
        mask_one = mask_img.get_fdata().flatten() > 0
        nvox = float(mask_one.sum())
        assert nvox > 0, 'Null mask found in %s' % maskpath
        if mask is None:
            mask = mask_one
            nvox_sum = nvox
        else:
            mask = mask | mask_one
            nvox_sum += nvox

    p = 0
    result = []

    for iframe in range(data.shape[-1]):
        frame = data[:, :, :, iframe].flatten()
        total = frame[mask].sum()
        q = total / nvox_sum

        if iframe == 0:
            diff = 0.0
        else:
            diff = q - p
        result.append([q, diff])
        p = q
    return np.array(result)


def regressor_PCA_singlebold(pca_data, n):
    pca = PCA(n_components=n, random_state=False)
    pca_regressor = pca.fit_transform(pca_data.T)
    return pca_regressor


def regressors_PCA(data, maskpath):
    '''
    Generate PCA regressor from outer points of brain.
        bold_path - path. Path of bold.
        maskpath - Path to file containing mask.
        outpath  - Path to file to place the output.
    '''
    # PCA parameter.
    n = 10

    # Open mask.
    mask_img = nib.load(maskpath)
    mask = mask_img.get_fdata().swapaxes(0, 1)
    mask = mask.flatten(order='F') == 0
    nvox = float(mask.sum())
    assert nvox > 0, 'Null mask found in %s' % maskpath

    data = data.swapaxes(0, 1)
    vol_data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]), order='F')
    pca_data = vol_data[mask]
    pca_regressor = regressor_PCA_singlebold(pca_data, n)

    return pca_regressor


def compile_regressors(bold_path,
                       aseg_brainmask_bin, aseg_wm, aseg_ventricles, aseg_csf, tr,
                       output_file, nskip):

    img = nib.load(bold_path)
    data = img.get_fdata()
    data_bandpass = bandpass_nifti(data, nskip=nskip, order=2, band=[0.01, 0.08], tr=tr)

    whole_brain = qnt_nifti(data_bandpass, str(aseg_brainmask_bin))

    csf = qnt_nifti(data_bandpass, [str(aseg_ventricles), str(aseg_csf)])

    white_matter = qnt_nifti(data_bandpass, str(aseg_wm))

    csf_wm = qnt_nifti(data_bandpass, [str(aseg_ventricles), str(aseg_csf), str(aseg_wm)])

    # Generate PCA regressors of bpss nifti.
    e_comp_cor = regressors_PCA(data_bandpass, str(aseg_brainmask_bin))


    # Prepare regressors datas for download
    output_file = Path(output_file)
    label_header = ['global_signal', 'global_signal_derivative1', 'csf', 'csf_derivative1',
                    'white_matter', 'white_matter_derivative1', 'csf_wm', 'csf_wm_derivative1',
                    'e_comp_cor_00', 'e_comp_cor_01', 'e_comp_cor_02', 'e_comp_cor_03', 'e_comp_cor_04',
                    'e_comp_cor_05', 'e_comp_cor_06', 'e_comp_cor_07', 'e_comp_cor_08', 'e_comp_cor_09']

    confounds_np = np.concatenate([whole_brain, csf, white_matter, csf_wm, e_comp_cor], axis=1)
    confounds = pd.DataFrame(confounds_np, columns=label_header)
    # cal power2 of matter signal
    for label in label_header[:8]:
        confounds[f'{label}_power2'] = confounds[label] ** 2
    confounds.to_csv(output_file, index=False, sep='\t')

    return output_file


def get_space_t1w_bold(subject_id, bids_preproc, bold_orig_file):
    from bids import BIDSLayout
    assert subject_id.startswith('sub-')
    layout_preproc = BIDSLayout(str(os.path.join(bids_preproc, subject_id)),
                                config=['bids', 'derivatives'], validate=False)
    info = layout_preproc.parse_file_entities(bold_orig_file)

    boldref_t1w_info = info.copy()
    boldref_t1w_info['space'] = 'T1w'
    boldref_t1w_info['suffix'] = 'boldref'
    boldref_t1w_file = layout_preproc.get(**boldref_t1w_info)[0]

    bold_t1w_info = info.copy()
    bold_t1w_info['space'] = 'T1w'
    bold_t1w_info['desc'] = 'preproc'
    bold_t1w_info['suffix'] = 'bold'
    bold_t1w_file = layout_preproc.get(**bold_t1w_info)[0]

    TR = layout_preproc.get_metadata(bold_t1w_file.path)['RepetitionTime']

    return bold_t1w_file, boldref_t1w_file, TR


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- BoldSkipReorient"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--bold_file", required=True)
    parser.add_argument("--aseg_mgz", required=True)
    parser.add_argument("--brainmask_mgz", required=True)
    parser.add_argument("--skip_frame", required=True)
    args = parser.parse_args()
    """
    input:
    --bids_dir /mnt/ngshare2/DeepPrep_Test/test_BoldOnly/bids
    --bold_preprocess_dir /mnt/ngshare2/DeepPrep_Test/test_BoldOnly/test_BoldOnly_DP_2410_snone/BOLD
    --work_dir /mnt/ngshare2/DeepPrep_Test/test_BoldOnly/test_BoldOnly_DP_2410_snone/WorkDir
    --bold_id sub-1021440_ses-02_task-rest_run-01
    --bold_file /mnt/ngshare2/DeepPrep_Test/test_BoldOnly/test_BoldOnly_DP_2410_snone/BOLD/sub-1021440/ses-02/func/sub-1021440_ses-02_task-rest_run-01_space-T1w_desc-preproc_bold.nii.gz
    --aseg_mgz /mnt/ngshare2/DeepPrep_Test/test_BoldOnly/Recon/sub-1021440/mri/aseg.mgz
    --brainmask_mgz /mnt/ngshare2/DeepPrep_Test/test_BoldOnly/Recon/sub-1021440/mri/brainmask.mgz
    --skip_frame 0
    output:
    confounds_file
    """

    tmp_path = Path(args.work_dir)
    anat2bold_t1w_dir = tmp_path / 'anat2bold_t1w' / args.bold_id
    anat2bold_t1w_dir.mkdir(parents=True, exist_ok=True)

    aseg = anat2bold_t1w_dir / 'label-aseg_probseg.nii.gz'
    wm = anat2bold_t1w_dir / 'label-WM_probseg.nii.gz'
    vent = anat2bold_t1w_dir / 'label-ventricles_probseg.nii.gz'
    csf = anat2bold_t1w_dir / 'label-CSF_probseg.nii.gz'

    brainmask = anat2bold_t1w_dir / 'brainmask.nii.gz'
    brainmask_bin = anat2bold_t1w_dir / 'label-brain_probseg.nii.gz'

    with open(args.bold_file, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_orig_file = data[1]

    bold_space_t1w_file, boldref_space_t1w_file, TR = get_space_t1w_bold(subject_id=args.subject_id, bids_preproc=args.bold_preprocess_dir, bold_orig_file=bold_orig_file)

    anat2bold_t1w(args.aseg_mgz, args.brainmask_mgz, str(boldref_space_t1w_file.path),
                  str(aseg), str(wm), str(vent), str(csf), str(brainmask), str(brainmask_bin))

    output_dir = os.path.join(args.work_dir, 'confounds', args.subject_id, args.bold_id)
    os.makedirs(output_dir, exist_ok=True)
    confounds_file = Path(output_dir, 'confounds_part1.tsv')
    compile_regressors(str(bold_space_t1w_file.path),
                       brainmask_bin, wm, vent, csf, TR,
                       confounds_file, int(args.skip_frame))
    assert confounds_file.exists()
