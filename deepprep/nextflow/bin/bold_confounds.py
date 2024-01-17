#! /usr/bin/env python3
import argparse
import os
import sh
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
import shutil

from bold_mkbrainmask import anat2bold_t1w


def qnt_nifti(bpss_path, maskpath, outpath):
    '''

    bpss_path - path. Path of bold after bpass process.
    maskpath - Path to file containing mask.
    outpath  - Path to file to place the output.
    '''

    # Open mask.
    mask_img = nib.load(maskpath)
    mask = mask_img.get_fdata().flatten() > 0
    nvox = float(mask.sum())
    assert nvox > 0, 'Null mask found in %s' % maskpath

    p = 0
    with outpath.open('w') as f:
        img = nib.load(bpss_path)
        data = img.get_fdata()
        for iframe in range(data.shape[-1]):
            frame = data[:, :, :, iframe].flatten()
            total = frame[mask].sum()
            q = total / nvox

            if iframe == 0:
                diff = 0.0
            else:
                diff = q - p
            f.write('%10.4f\t%10.4f\n' % (q, diff))
            p = q


def regressor_PCA_singlebold(pca_data, n):
    pca = PCA(n_components=n, random_state=False)
    pca_regressor = pca.fit_transform(pca_data.T)
    return pca_regressor


def regressors_PCA(bpss_path, maskpath, outpath):
    '''
    Generate PCA regressor from outer points of brain.
        bpss_path - path. Path of bold after bpass process.
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

    with outpath.open('w') as f:
        img = nib.load(bpss_path)
        data = img.get_fdata().swapaxes(0, 1)
        vol_data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]), order='F')
        pca_data = vol_data[mask]
        pca_regressor = regressor_PCA_singlebold(pca_data, n)
        for iframe in range(data.shape[-1]):
            for idx in range(n):
                f.write('%10.4f\t' % (pca_regressor[iframe, idx]))
            f.write('\n')


def build_movement_regressors(movement_path: Path, mcpar_file: Path, output_file: Path):
    """
    MCFLIRT motion parameters, normalized to SPM format (X, Y, Z, Rx, Ry, Rz)
    """
    # *.par -> *.dat
    mcdat = pd.read_csv(mcpar_file, header=None, sep='  ', engine='python').to_numpy()
    dat_file = movement_path / mcpar_file.name.replace('par', 'dat')
    dat = mcdat
    dat_txt = list()
    for idx, row in enumerate(dat):
        dat_line = f'{idx + 1}{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}{1:10.6f}'
        dat_txt.append(dat_line)
    with open(dat_file, 'w') as f:
        f.write('\n'.join(dat_txt))

    # *.par -> *.ddat
    ddat_file = movement_path / mcpar_file.name.replace('par', 'ddat')
    ddat = mcdat
    ddat = ddat[1:, :] - ddat[:-1, ]
    ddat = np.vstack((np.zeros((1, 6)), ddat))
    ddat_txt = list()
    for idx, row in enumerate(ddat):
        if idx == 0:
            ddat_line = f'{idx + 1}{0:10.6f}{0:10.6f}{0:10.6f}{0:10.6f}{0:10.6f}{0:10.6f}{1:10.6f}'
        else:
            ddat_line = f'{idx + 1}{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}{0:10.6f}'
        ddat_txt.append(ddat_line)
    with open(ddat_file, 'w') as f:
        f.write('\n'.join(ddat_txt))

    # *.par -> *.rdat
    rdat_file = movement_path / mcpar_file.name.replace('par', 'rdat')
    rdat = mcdat
    # rdat_average = np.zeros(rdat.shape[1])
    # for idx, row in enumerate(rdat):
    #     rdat_average = (row + rdat_average * idx) / (idx + 1)
    rdat_average = rdat.mean(axis=0)
    rdat = rdat - rdat_average
    rdat_txt = list()
    for idx, row in enumerate(rdat):
        rdat_line = f'{idx + 1}{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}{1:10.6f}'
        rdat_txt.append(rdat_line)
    with open(rdat_file, 'w') as f:
        f.write('\n'.join(rdat_txt))

    # *.rdat, *.ddat -> *.rddat
    rddat_file = movement_path / mcpar_file.name.replace('par', 'rddat')
    rddat = np.hstack((rdat, ddat))
    rddat_txt = list()
    for idx, row in enumerate(rddat):
        rddat_line = f'{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}\t' + \
                     f'{row[6]:10.6f}{row[7]:10.6f}{row[8]:10.6f}{row[9]:10.6f}{row[10]:10.6f}{row[11]:10.6f}'
        rddat_txt.append(rddat_line)
    with open(rddat_file, 'w') as f:
        f.write('\n'.join(rddat_txt))

    rddat = np.around(rddat, 6)
    n = rddat.shape[0]
    ncol = rddat.shape[1]
    x = np.zeros(n)
    for i in range(n):
        x[i] = -1. + 2. * i / (n - 1)

    sxx = n * (n + 1) / (3. * (n - 1))

    sy = np.zeros(ncol)
    sxy = np.zeros(ncol)
    a0 = np.zeros(ncol)
    a1 = np.zeros(ncol)
    for j in range(ncol - 1):
        sy[j] = 0
        sxy[j] = 0
        for i in range(n):
            sy[j] += rddat[i, j]
            sxy[j] += rddat[i, j] * x[i]
        a0[j] = sy[j] / n
        a1[j] = sxy[j] / sxx
        for i in range(n):
            rddat[i, j] -= a1[j] * x[i]

    regressor_dat_txt = list()
    for idx, row in enumerate(rddat):
        regressor_dat_line = f'{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}' + \
                             f'{row[6]:10.6f}{row[7]:10.6f}{row[8]:10.6f}{row[9]:10.6f}{row[10]:10.6f}{row[11]:10.6f}'
        regressor_dat_txt.append(regressor_dat_line)
    with open(output_file, 'w') as f:
        f.write('\n'.join(regressor_dat_txt))


def compile_regressors(func_path: Path, bold_id: str, bpss_path: Path, confounds_dir_path,
                       mcdat_file, aseg_wm, aseg_brainmask, aseg_brainmask_bin, aseg_ventricles,
                       output_file):
    # Compile the regressors.

    # wipe mov regressors, if there
    mov_out_path = confounds_dir_path / 'mov_regressor.dat'
    build_movement_regressors(confounds_dir_path, mcdat_file, mov_out_path)

    wb_out_path = confounds_dir_path / 'WB_regressor_dt.dat'
    qnt_nifti(bpss_path, str(aseg_brainmask_bin), wb_out_path)

    vent_out_path = confounds_dir_path / 'ventricles_regressor_dt.dat'
    qnt_nifti(bpss_path, str(aseg_ventricles), vent_out_path)

    wm_out_path = confounds_dir_path / 'WM_regressor_dt.dat'
    qnt_nifti(bpss_path, str(aseg_wm), wm_out_path)

    pasted_out_path = confounds_dir_path / 'Vent_wm_dt.dat'
    with pasted_out_path.open('w') as f:
        sh.paste(vent_out_path, wm_out_path, _out=f)

    # Generate PCA regressors of bpss nifti.
    pca_out_path = confounds_dir_path / 'pca_regressor_dt.dat'
    regressors_PCA(bpss_path, str(aseg_brainmask), pca_out_path)

    fnames = [
        mov_out_path,
        wb_out_path,
        vent_out_path,
        pca_out_path]
    all_regressors_path = confounds_dir_path.parent / f'confounds.txt'
    regressors = []
    for fname in fnames:
        with fname.open('r') as f:
            regressors.append(
                np.array([
                    list(map(float, line.replace('-', ' -').strip().split()))
                    for line in f]))
    regressors = np.hstack(regressors)
    with all_regressors_path.open('w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(regressors)

    # Prepare regressors datas for download
    output_file = Path(output_file)
    num_row = len(regressors[:, 0])
    frame_no = np.arange(num_row).reshape((num_row, 1))
    download_regressors = np.concatenate((frame_no, regressors), axis=1)
    label_header = ['Frame', 'dL', 'dP', 'dS', 'pitch', 'yaw', 'roll',
                    'dL_d', 'dP_d', 'dS_d', 'pitch_d', 'yaw_d', 'roll_d',
                    'WB', 'WB_d', 'vent', 'vent_d', 'wm', 'wm_d',
                    'comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6', 'comp7', 'comp8', 'comp9', 'comp10']
    with output_file.open('w') as f:
        csv.writer(f, delimiter=' ').writerows([label_header])
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(download_regressors)

    return output_file


def get_space_t1w_bold(bids_orig, bids_preproc, bold_orig_file):
    from bids import BIDSLayout
    layout_orig = BIDSLayout(bids_orig, validate=False)
    layout_preproc = BIDSLayout(bids_preproc, validate=False)
    info = layout_orig.parse_file_entities(bold_orig_file)

    boldref_t1w_info = info.copy()
    boldref_t1w_info['space'] = 'T1w'
    boldref_t1w_info['suffix'] = 'boldref'
    boldref_t1w_file = layout_preproc.get(**boldref_t1w_info)[0]

    bold_t1w_info = info.copy()
    bold_t1w_info['space'] = 'T1w'
    bold_t1w_info['desc'] = 'preproc'
    bold_t1w_info['suffix'] = 'bold'
    bold_t1w_file = layout_preproc.get(**bold_t1w_info)[0]

    # sub-CIMT001_ses-38659_task-rest_run-01_bold_mcf.nii.gz.par
    bold_par_info = info.copy()
    bold_par_info.pop('datatype')
    bold_par_info['suffix'] = 'mcf'
    bold_par_info['extension'] = '.nii.par'
    bold_par = layout_preproc.get(**bold_par_info)[0]

    return bold_t1w_file.path, boldref_t1w_file.path, bold_par.path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- BoldSkipReorient"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--bold_file", required=True)
    parser.add_argument("--aseg_mgz", required=True)
    parser.add_argument("--brainmask_mgz", required=True)
    args = parser.parse_args()
    """
    input:
    --bids_dir /mnt/ngshare/temp/ds004498
    --bold_preprocess_dir /mnt/ngshare/temp/ds004498_DeepPrep/BOLD
    --bold_id sub-CIMT001_ses-38659_task-rest_run-01
    --bold_file /mnt/ngshare/temp/ds004498/sub-CIMT001/ses-38659/func/sub-CIMT001_ses-38659_task-rest_run-01_bold.nii.gz
    --aseg_mgz /mnt/ngshare/temp/ds004498/Recon720/sub-CIMT001/mri/aseg.mgz
    --brainmask_mgz /mnt/ngshare/temp/ds004498/Recon720/sub-CIMT001/mri/brainmask.mgz
    output:
    confounds_file
    """

    tmp_path = Path(args.bold_preprocess_dir) / 'tmp'

    confounds_dir_path = tmp_path / 'confounds' / args.bold_id
    anat2bold_t1w_dir = tmp_path / 'anat2bold_t1w' / args.bold_id
    confounds_dir_path.mkdir(parents=True, exist_ok=True)
    anat2bold_t1w_dir.mkdir(parents=True, exist_ok=True)

    aseg = anat2bold_t1w_dir / 'dseg.nii.gz'
    wm = anat2bold_t1w_dir / 'label-WM_probseg.nii.gz'
    vent = anat2bold_t1w_dir / 'label-ventricles_probseg.nii.gz'
    csf = anat2bold_t1w_dir / 'label-CSF_probseg.nii.gz'
    # project brainmask.mgz to mc
    mask = anat2bold_t1w_dir / 'desc-brain_mask.nii.gz'
    binmask = anat2bold_t1w_dir / 'desc-brain_maskbin.nii.gz'

    bold_space_t1w_file, boldref_space_t1w_file, bold_mcpar_file = get_space_t1w_bold(bids_orig=args.bids_dir, bids_preproc=args.bold_preprocess_dir, bold_orig_file=args.bold_file)

    anat2bold_t1w(args.aseg_mgz, args.brainmask_mgz, boldref_space_t1w_file,
                  str(aseg), str(wm), str(vent), str(csf), str(mask), str(binmask))

    confounds_file = Path(boldref_space_t1w_file).parent / f'{args.bold_id}_desc-confounds_timeseries.txt'
    compile_regressors(Path(args.bold_preprocess_dir), args.bold_id, bold_space_t1w_file, confounds_dir_path,
                       Path(bold_mcpar_file), wm, mask, binmask, vent,
                       confounds_file)
    assert confounds_file.exists()
