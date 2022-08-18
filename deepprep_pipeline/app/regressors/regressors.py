import os
import sys
import csv
from pathlib import Path
import numpy as np
import sh
import nibabel as nib
from sklearn.decomposition import PCA
import pandas as pd


def qnt_nifti(bpss_path, maskpath, outpath):
    '''

    bpss_path - path. Path of bold after bpass process.
    maskpath - Path to file containing mask.
    outpath  - Path to file to place the output.
    '''

    # Open mask.
    mask_img = nib.load(maskpath)
    mask = mask_img.get_data().flatten() > 0
    nvox = float(mask.sum())
    assert nvox > 0, 'Null mask found in %s' % maskpath

    p = 0
    with outpath.open('w') as f:
        img = nib.load(bpss_path)
        data = img.get_data()
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
    mask = mask_img.get_data().swapaxes(0, 1)
    mask = mask.flatten(order='F') == 0
    nvox = float(mask.sum())
    assert nvox > 0, 'Null mask found in %s' % maskpath

    with outpath.open('w') as f:
        img = nib.load(bpss_path)
        data = img.get_data().swapaxes(0, 1)
        vol_data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]), order='F')
        pca_data = vol_data[mask]
        pca_regressor = regressor_PCA_singlebold(pca_data, n)
        for iframe in range(data.shape[-1]):
            for idx in range(n):
                f.write('%10.4f\t' % (pca_regressor[iframe, idx]))
            f.write('\n')


def build_movement_regressors(subject, bldrun, bold_path: Path, movement_path: Path, fcmri_path: Path):
    # *.mcdata -> *.par
    mcdat_file = bold_path / bldrun / f'{subject}_bld_rest_reorient_skip_faln_mc.mcdat'
    par_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.par'
    mcdat = pd.read_fwf(mcdat_file, header=None).to_numpy()
    par = mcdat[:, 1:7]
    par_txt = list()
    for row in par:
        par_txt.append(f'{row[0]:.4f}  {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}  {row[4]:.4f}  {row[5]:.4f}')
    with open(par_file, 'w') as f:
        f.write('\n'.join(par_txt))

    # *.par -> *.dat
    dat_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.dat'
    dat = mcdat[:, [4, 5, 6, 1, 2, 3]]
    dat_txt = list()
    for idx, row in enumerate(dat):
        dat_line = f'{idx + 1}{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}{1:10.6f}'
        dat_txt.append(dat_line)
    with open(dat_file, 'w') as f:
        f.write('\n'.join(dat_txt))

    # *.par -> *.ddat
    ddat_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.ddat'
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
    rdat_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.rdat'
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
    rddat_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.rddat'
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


def compile_regressors(preprocess_dir, bold_path, bldrun, subject, fcmri_path, bpss_path):
    # Compile the regressors.
    movement_path = preprocess_dir / subject / 'movement'
    movement_path.mkdir(exist_ok=True)

    # wipe mov regressors, if there
    mov_regressor_common_path = fcmri_path / ('%s_mov_regressor.dat' % subject)
    build_movement_regressors(subject, bldrun, bold_path, movement_path, fcmri_path)
    mov_regressor_path = fcmri_path / ('%s_bld%s_mov_regressor.dat' % (subject, bldrun))
    os.rename(mov_regressor_common_path, mov_regressor_path)

    mask_path = bold_path / bldrun / ('%s.brainmask.bin.nii.gz' % subject)
    out_path = fcmri_path / ('%s_bld%s_WB_regressor_dt.dat' % (subject, bldrun))
    qnt_nifti(bpss_path, str(mask_path), out_path)

    mask_path = bold_path / bldrun / ('%s.func.ventricles.nii.gz' % subject)
    vent_out_path = fcmri_path / ('%s_bld%s_ventricles_regressor_dt.dat' % (subject, bldrun))
    qnt_nifti(bpss_path, str(mask_path), vent_out_path)

    mask_path = bold_path / bldrun / ('%s.func.wm.nii.gz' % subject)
    wm_out_path = fcmri_path / ('%s_bld%s_wm_regressor_dt.dat' % (subject, bldrun))
    qnt_nifti(bpss_path, str(mask_path), wm_out_path)

    pasted_out_path = fcmri_path / ('%s_bld%s_vent_wm_dt.dat' % (subject, bldrun))
    with pasted_out_path.open('w') as f:
        sh.paste(vent_out_path, wm_out_path, _out=f)

    # Generate PCA regressors of bpss nifti.
    mask_path = bold_path / bldrun / ('%s.brainmask.nii.gz' % subject)
    pca_out_path = fcmri_path / ('%s_bld%s_pca_regressor_dt.dat' % (subject, bldrun))
    regressors_PCA(bpss_path, str(mask_path), pca_out_path)

    fnames = [
        fcmri_path / ('%s_bld%s_mov_regressor.dat' % (subject, bldrun)),
        fcmri_path / ('%s_bld%s_WB_regressor_dt.dat' % (subject, bldrun)),
        fcmri_path / ('%s_bld%s_vent_wm_dt.dat' % (subject, bldrun)),
        fcmri_path / ('%s_bld%s_pca_regressor_dt.dat' % (subject, bldrun))]
    all_regressors_path = fcmri_path / ('%s_bld%s_regressors.dat' % (subject, bldrun))
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
    download_all_regressors_path = Path(str(all_regressors_path).replace('.dat', '_download.txt'))
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

    return all_regressors_path


def glm_nifti(bpss_path, regressor_path):
    '''
    Compute the residuals of the voxel information in a NIFTI file.

    bpss_path - path. Path of bold after bpass process.
    regressor_path - Path object.

    Creates a file with the same name as :nifti_path:, but ending with
    '_resid.nii.gz'.
    return:
        resid_path - path. Path of bold after regression process.
    '''

    # Read the regressors.
    regressors = []
    with regressor_path.open('r') as f:
        for line in f:
            regressors.append(list(map(float, line.split())))
    regressors = np.array(regressors)

    # Read NIFTI values.
    nifti_img = nib.load(bpss_path)
    nifti_data = nifti_img.get_data()
    nx, ny, nz, nv = nifti_data.shape
    nifti_data = np.reshape(nifti_data, (nx * ny * nz, nv)).T

    # Assemble linear system and solve it.
    A = np.hstack((regressors, np.ones((regressors.shape[0], 1))))
    x = np.linalg.lstsq(A, nifti_data, rcond=-1)[0]

    # Compute residuals.
    residuals = nifti_data - np.matmul(A, x)
    residuals = np.reshape(residuals.T, (nx, ny, nz, nv))

    # Save result.
    resid_path = bpss_path.replace('.nii.gz', '_resid.nii.gz')
    hdr = nifti_img.header
    aff = nifti_img.affine
    new_img = nib.Nifti1Image(
        residuals.astype(np.float32), affine=aff, header=hdr)
    new_img.header['pixdim'] = nifti_img.header['pixdim']
    nib.save(new_img, resid_path)
    return resid_path


def var_nifti(resid_path, sd_path):
    '''
    resid_path - path. Path of bold after regression process.
    sd_path  - Path to destination file.
    '''
    img = nib.load(resid_path)
    frame_std = np.std(img.get_data(), axis=3)
    out_img = nib.Nifti1Image(frame_std, affine=img.affine, header=img.header)
    nib.save(out_img, sd_path)


def snr_main(fmri_fname, sd_fname, out_fname):
    '''
    Approximate the SNR of fMRI signal.

    fmri_fname - str. Path to fMRI signal file. Data is n x m x p x q
    sd_fname   - str. Path containing standard deviation volume. Data is
                 n x m x p.
    out_fname  - str. Path to save the result.

    '''

    # Extract fMRI time-series and take the average of each vertex over time.
    fmri_img = nib.load(fmri_fname)
    fmri_data = fmri_img.get_data()
    mean_fmri_data = fmri_data.mean(axis=3)

    # Approximate SNR.
    sd_img = nib.load(sd_fname)
    sd_data = sd_img.get_data()

    # Avoid being divided by 0 which causes redundant log
    sd_data[sd_data == 0] = np.nan
    snr_data = mean_fmri_data / sd_data
    snr_data[snr_data == np.nan] = 0

    # Save SNR.
    snr_img = nib.Nifti1Image(snr_data, np.eye(4))
    nib.save(snr_img, out_fname)


def regression(
        bpss_path,
        all_regressors_path):
    # Run regression.
    resid_path = glm_nifti(bpss_path, all_regressors_path)

    # Calculate standard deviation of residuals.
    sd_path = resid_path.replace('.nii.gz', '_sd1.nii.gz')

    var_nifti(resid_path, sd_path)
    snr_path = resid_path.replace('.nii.gz', '_snr.nii.gz')

    snr_main(resid_path, sd_path, snr_path)
