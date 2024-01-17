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


def build_movement_regressors(subject, movement_path: Path, fcmri_path: Path, mcdat_file: Path):
    # *.mcdata -> *.par
    par_file = movement_path / f'{subject}_rest_reorient_skip_faln_mc.par'
    mcdat = pd.read_fwf(mcdat_file, header=None).to_numpy()
    par = mcdat[:, 1:7]
    par_txt = list()
    for row in par:
        par_txt.append(f'{row[0]:.4f}  {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}  {row[4]:.4f}  {row[5]:.4f}')
    with open(par_file, 'w') as f:
        f.write('\n'.join(par_txt))

    # *.par -> *.dat
    dat_file = movement_path / f'{subject}_rest_reorient_skip_faln_mc.dat'
    dat = mcdat[:, [4, 5, 6, 1, 2, 3]]
    dat_txt = list()
    for idx, row in enumerate(dat):
        dat_line = f'{idx + 1}{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}{1:10.6f}'
        dat_txt.append(dat_line)
    with open(dat_file, 'w') as f:
        f.write('\n'.join(dat_txt))

    # *.par -> *.ddat
    ddat_file = movement_path / f'{subject}_rest_reorient_skip_faln_mc.ddat'
    ddat = mcdat[:, [4, 5, 6, 1, 2, 3]]
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
    rdat_file = movement_path / f'{subject}_rest_reorient_skip_faln_mc.rdat'
    rdat = mcdat[:, [4, 5, 6, 1, 2, 3]]
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
    rddat_file = movement_path / f'{subject}_rest_reorient_skip_faln_mc.rddat'
    rddat = np.hstack((rdat, ddat))
    rddat_txt = list()
    for idx, row in enumerate(rddat):
        rddat_line = f'{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}\t' + \
                     f'{row[6]:10.6f}{row[7]:10.6f}{row[8]:10.6f}{row[9]:10.6f}{row[10]:10.6f}{row[11]:10.6f}'
        rddat_txt.append(rddat_line)
    with open(rddat_file, 'w') as f:
        f.write('\n'.join(rddat_txt))

    regressor_dat_file = fcmri_path / f'{subject}_mov_regressor.dat'
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
    with open(regressor_dat_file, 'w') as f:
        f.write('\n'.join(regressor_dat_txt))


def compile_regressors(func_path: Path, subject, bold_id: str, bpss_path: Path, confounds_dir_path,
                       mcdat_file, aseg_wm, aseg_brainmask, aseg_brainmask_bin, aseg_ventricles
                       ):
    # Compile the regressors.

    # wipe mov regressors, if there
    mov_regressor_common_path = confounds_dir_path / ('%s_mov_regressor.dat' % subject)
    build_movement_regressors(subject, confounds_dir_path, confounds_dir_path, mcdat_file)
    mov_regressor_path = confounds_dir_path / ('%s_mov_regressor.dat' % subject)
    os.rename(mov_regressor_common_path, mov_regressor_path)

    out_path = confounds_dir_path / ('%s_WB_regressor_dt.dat' % subject)
    qnt_nifti(bpss_path, str(aseg_brainmask_bin), out_path)

    vent_out_path = confounds_dir_path / ('%s_ventricles_regressor_dt.dat' % subject)
    qnt_nifti(bpss_path, str(aseg_ventricles), vent_out_path)

    wm_out_path = confounds_dir_path / ('%s_wm_regressor_dt.dat' % subject)
    qnt_nifti(bpss_path, str(aseg_wm), wm_out_path)

    pasted_out_path = confounds_dir_path / ('%s_vent_wm_dt.dat' % subject)
    with pasted_out_path.open('w') as f:
        sh.paste(vent_out_path, wm_out_path, _out=f)

    # Generate PCA regressors of bpss nifti.
    pca_out_path = confounds_dir_path / ('%s_pca_regressor_dt.dat' % subject)
    regressors_PCA(bpss_path, str(aseg_brainmask), pca_out_path)

    fnames = [
        confounds_dir_path / ('%s_mov_regressor.dat' % subject),
        confounds_dir_path / ('%s_WB_regressor_dt.dat' % subject),
        confounds_dir_path / ('%s_vent_wm_dt.dat' % subject),
        confounds_dir_path / ('%s_pca_regressor_dt.dat' % subject)]
    all_regressors_path = confounds_dir_path.parent / f'{bold_id}_confounds.txt'
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
    download_all_regressors_path = Path(str(all_regressors_path).replace('.txt', '_view.txt'))
    num_row = len(regressors[:, 0])
    frame_no = np.arange(num_row).reshape((num_row, 1))
    download_regressors = np.concatenate((frame_no, regressors), axis=1)
    label_header = ['Frame', 'dL', 'dP', 'dS', 'pitch', 'yaw', 'roll',
                    'dL_d', 'dP_d', 'dS_d', 'pitch_d', 'yaw_d', 'roll_d',
                    'WB', 'WB_d', 'vent', 'vent_d', 'wm', 'wm_d',
                    'comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6', 'comp7', 'comp8', 'comp9', 'comp10']
    with download_all_regressors_path.open('w') as f:
        csv.writer(f, delimiter=' ').writerows([label_header])
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(download_regressors)
    shutil.copyfile(download_all_regressors_path, func_path / f'{bold_id}_desc-confounds_timeseries.txt')


    return all_regressors_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- BoldSkipReorient"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--bold_file", required=True)
    parser.add_argument("--mcdat", required=True)
    parser.add_argument("--aseg_wm", required=True)
    parser.add_argument("--aseg_brainmask", required=True)
    parser.add_argument("--aseg_brainmask_bin", required=True)
    parser.add_argument("--aseg_ventricles", required=True)
    args = parser.parse_args()

    func_path = Path(args.bold_preprocess_dir) / args.subject_id / 'func'
    tmp_path = Path(args.bold_preprocess_dir) / args.subject_id / 'tmp'
    confounds_dir_path = tmp_path / 'confounds' / args.bold_id
    try:
        confounds_dir_path.mkdir(parents=True, exist_ok=True)
    except:
        pass

    bold_file = Path(args.bold_file)
    assert bold_file.exists()

    mcdat_file = args.mcdat
    aseg_wm = args.aseg_wm
    aseg_brainmask = args.aseg_brainmask
    aseg_brainmask_bin = args.aseg_brainmask_bin
    aseg_ventricles = args.aseg_ventricles

    compile_regressors(func_path, args.subject_id, args.bold_id, bold_file, confounds_dir_path,
                       mcdat_file, aseg_wm, aseg_brainmask, aseg_brainmask_bin, aseg_ventricles
                       )
