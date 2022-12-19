import os
import sys
import sh
from pathlib import Path
import bids
import ants
import csv
import numpy as np
from app_pipeline.filters.filters import _bandpass_nifti
from app_pipeline.regressors.regressors import regression
from app_pipeline.surface_projection.surface_projection import smooth_fs6, downsample_fs6_to_fs4
import pandas as pd
import nibabel as nib
from sklearn.decomposition import PCA
from multiprocessing import Pool


def project_surface_deepprep(input_path, hemi, outpath, reg_path):
    cmd = ['--mov', input_path,
           '--hemi', hemi,
           '--reg', reg_path,
           '--projfrac', 0.5,
           '--trgsubject', 'fsaverage6',
           '--o', outpath,
           '--reshape',
           '--interp', 'trilinear']
    sh.mri_vol2surf(cmd, _out=sys.stdout)


def project_surface_fmriprep(input_path, hemi, outpath):
    cmd = ['--mov', input_path,
           '--mni152reg',
           '--hemi', hemi,
           '--projfrac', 0.5,
           '--trgsubject', 'fsaverage6',
           '--o', outpath,
           '--reshape',
           '--interp', 'trilinear']
    sh.mri_vol2surf(cmd, _out=sys.stdout)


def bandpass_nifti(gauss_path, bpss_path, tr):
    '''
    Perform a bandpass filter on each voxel in each .nii.gz file listed,
    as lines, in :arg listfile:.

    gauss_path  - path. Path of bold after gauss filter process.
    tr        - float. TR period.

    Creates files with 'bpss' in their filenames.
    return:
        bpss_path - path. Path of bold after bpass process.
    '''

    # Get and set parameters.
    nskip = 0
    fhalf_lo = 0.0  # 1e-16  # make parameter the same to INDI_lab
    fhalf_hi = 0.08
    band = [fhalf_lo, fhalf_hi]
    order = 2
    # bpss_path = gauss_path.replace('.nii.gz', '_bpss.nii.gz')
    _bandpass_nifti(gauss_path, bpss_path, nskip, order, band, tr)
    return bpss_path


def glm_nifti(bpss_path, resid_path, regressor_path):
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
    hdr = nifti_img.header
    aff = nifti_img.affine
    new_img = nib.Nifti1Image(
        residuals.astype(np.float32), affine=aff, header=hdr)
    new_img.header['pixdim'] = nifti_img.header['pixdim']
    nib.save(new_img, resid_path)
    return resid_path


def bold_smooth_6_ants(t12mm, t12mm_sm6_file, verbose=False):
    # mask file
    MNI152_T1_2mm_brain_mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    brain_mask = ants.image_read(MNI152_T1_2mm_brain_mask)

    if isinstance(t12mm, str):
        bold_img = ants.image_read(t12mm)
    else:
        bold_img = t12mm

    bold_origin = bold_img.origin
    bold_spacing = bold_img.spacing
    bold_direction = bold_img.direction.copy()

    # smooth
    smoothed_img = ants.from_numpy(bold_img.numpy(), bold_origin[:3], bold_spacing[:3],
                                   bold_direction[:3, :3].copy(), has_components=True)
    # smoothed_img = ants.smooth_image(smoothed_img, sigma=6, FWHM=True)

    # mask
    smoothed_np = ants.smooth_image(smoothed_img, sigma=6, FWHM=True).numpy()
    del smoothed_img
    mask_np = brain_mask.numpy()
    masked_np = np.zeros(smoothed_np.shape, dtype=np.float32)
    idx = mask_np == 1
    masked_np[idx, :] = smoothed_np[idx, :]
    del smoothed_np
    masked_img = ants.from_numpy(masked_np, bold_origin, bold_spacing, bold_direction)
    del masked_np
    if verbose:
        # save
        ants.image_write(masked_img, str(t12mm_sm6_file))
    return masked_img


def app_regressors():
    bpss_path = '/home/weiwei/workdata/DeepPrep/workdir/ds000224/derivatives/fmriprep/sub-MSC01/ses-func01/func/sub-MSC01_ses-func01_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss.nii.gz'
    all_regressors_path = '/home/weiwei/workdata/App/sub-MSC01/preprocess/sub-MSC01/fcmri/sub-MSC01_bld001_regressors.dat'
    # df = pd.read_csv('/home/weiwei/workdata/App/sub-MSC01/preprocess/sub-MSC01/fcmri/sub-MSC01_bld001_regressors.dat',
    #                  sep=' ', header=None)
    regression(bpss_path, Path(all_regressors_path))
    resid_path = bpss_path.replace('.nii.gz', '_resid.nii.gz')
    sm6_file = resid_path.replace('.nii.gz', '_sm6.nii.gz')
    bold_smooth_6_ants(resid_path, sm6_file, verbose=True)


def regressor_PCA_singlebold(pca_data, n):
    pca = PCA(n_components=n, random_state=False)
    pca_regressor = pca.fit_transform(pca_data.T)
    return pca_regressor


def regressors_PCA(bpss_path, maskpath):
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

    img = nib.load(bpss_path)
    data = img.get_fdata().swapaxes(0, 1)
    vol_data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]), order='F')
    pca_data = vol_data[mask]
    pca_regressor = regressor_PCA_singlebold(pca_data, n)
    return pca_regressor
    # with outpath.open('w') as f:
    #     img = nib.load(bpss_path)
    #     data = img.get_data().swapaxes(0, 1)
    #     vol_data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]), order='F')
    #     pca_data = vol_data[mask]
    #     pca_regressor = regressor_PCA_singlebold(pca_data, n)
    #     for iframe in range(data.shape[-1]):
    #         for idx in range(n):
    #             f.write('%10.4f\t' % (pca_regressor[iframe, idx]))
    #         f.write('\n')


def regression_fsaverage6_by_DeepPrep_confounds(bids_dir, bold_preprocess_dir: Path, bold_smooth_dir: Path, subject_id, bold_preprocess_confounds):
    task = 'rest'

    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)

    sess = layout.get_session(subject=subj)
    for ses in sess:
        bids_bolds = layout.get(subject=subj, session=ses, task=task, suffix='bold', extension='.nii.gz')
        for bids_bold in bids_bolds:
            bold_path = Path(bids_bold.path)
            print(f'<<< {bold_path}')

            bold_name = Path(bids_bold).name.replace('.nii.gz', '')
            subj_fcmri_dir = Path(bold_preprocess_confounds) / subject_id / 'func' / 'fcmri'
            confounds_file = subj_fcmri_dir / bold_name / f'{subject_id}_regressors.dat'

            preprocess_func_path_dir = bold_preprocess_dir / f'sub-{subj}' / f'ses-{ses}' / 'func'
            smooth_preprocess_surf_path_dir = bold_smooth_dir / f'sub-{subj}' / f'ses-{ses}' / 'surf'

            for lr in ['L', 'R']:
                if lr == 'L':
                    hemi = 'lh'
                else:
                    hemi = 'rh'
                orig_path = preprocess_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.gii'
                nii_path = smooth_preprocess_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.nii.gz'
                if not nii_path.exists():
                    cmd = f'mri_surf2surf --srcsubject fsaverage6 --srcsurfval {orig_path} --trgsubject fsaverage6 --trgsurfval {nii_path} --hemi {hemi}'
                    os.system(cmd)
                    print(f'>>> {nii_path}')

                bpss_path = smooth_preprocess_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.bpss.nii.gz'
                if not bpss_path.exists():
                    entities = dict(bids_bold.entities)
                    TR = layout.get_tr(entities)
                    bandpass_nifti(nii_path, bpss_path, TR)
                    print(f'>>> {bpss_path}')

                resid_path = smooth_preprocess_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.bpss_resid.nii.gz'
                if not resid_path.exists():
                    glm_nifti(str(bpss_path), str(resid_path), confounds_file)
                    print(f'>>> {resid_path}')


def regression_MNI152_by_DeepPrep_confounds(bids_dir, bold_preprocess_dir: Path, bold_smooth_dir: Path, subject_id, bold_preprocess_confounds):
    task = 'rest'

    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)

    sess = layout.get_session(subject=subj)
    for ses in sess:
        bids_bolds = layout.get(subject=subj, session=ses, task=task, suffix='bold', extension='.nii.gz')
        for bids_bold in bids_bolds:
            bold_path = Path(bids_bold.path)
            print(f'<<< {bold_path}')

            bold_name = Path(bids_bold).name.replace('.nii.gz', '')
            subj_fcmri_dir = Path(bold_preprocess_confounds) / subject_id / 'func' / 'fcmri'
            confounds_file = subj_fcmri_dir / bold_name / f'{subject_id}_regressors.dat'

            preprocess_func_path_dir = bold_preprocess_dir / f'sub-{subj}' / f'ses-{ses}' / 'func'
            bold_preproc_file = preprocess_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz'

            smooth_preprocess_func_path_dir = bold_smooth_dir / f'sub-{subj}' / f'ses-{ses}' / 'func'
            # bandpass
            bpss_path = smooth_preprocess_func_path_dir / bold_preproc_file.name.replace('.nii.gz', '_bpss.nii.gz')
            if not bpss_path.exists():
                bold_img = nib.load(bold_preproc_file)
                TR = bold_img.header.get_zooms()[3]
                bandpass_nifti(str(bold_preproc_file), str(bpss_path), TR)
                print(f'>>> {bpss_path}')

            # regression
            resid_path = smooth_preprocess_func_path_dir / bpss_path.name.replace('.nii.gz', '_resid.nii.gz')
            if not resid_path.exists():
                glm_nifti(str(bpss_path), str(resid_path), confounds_file)
                print(f'>>> {resid_path}')


def cal_regression(bids_dir, bold_preprocess_dir: Path, bold_smooth_dir: Path, subject_id):
    task = 'rest'

    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)

    sess = layout.get_session(subject=subj)
    for ses in sess:
        bids_bolds = layout.get(subject=subj, session=ses, task=task, suffix='bold', extension='.nii.gz')
        for bids_bold in bids_bolds:
            bold_path = Path(bids_bold.path)
            print(f'<<< {bold_path}')

            preprocess_func_path_dir = bold_preprocess_dir / f'sub-{subj}' / f'ses-{ses}' / 'func'
            bold_preproc_file = preprocess_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz'

            mask_path = preprocess_func_path_dir / bold_preproc_file.name.replace('preproc_bold.nii.gz', 'brain_mask.nii.gz')
            fMRIPrep_confounds_path = preprocess_func_path_dir / bold_preproc_file.name.replace('space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz', 'desc-confounds_timeseries.tsv')

            smooth_func_path_dir = bold_smooth_dir / f'sub-{subj}' / f'ses-{ses}' / 'func'
            # bandpass
            bpss_path = smooth_func_path_dir / bold_preproc_file.name.replace('.nii.gz', '_bpss.nii.gz')
            if not bpss_path.exists():
                bold_img = nib.load(bold_preproc_file)
                TR = bold_img.header.get_zooms()[3]
                bandpass_nifti(str(bold_preproc_file), str(bpss_path), TR)
                print(f'>>> {bpss_path}')

            # cal confounds
            resid6_confounds_path = smooth_func_path_dir / bold_preproc_file.name.replace('space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz', 'desc-confounds_timeseries_resid6.dat')
            if not resid6_confounds_path.exists():
                df_fmri = pd.read_csv(fMRIPrep_confounds_path, sep='\t')
                fmri_confounds = list(df_fmri.columns)
                # reg_confounds = [confound for confound in fmri_confounds if 'trans' in confound or 'rot' in confound]
                # reg_confounds = reg_confounds + [confound for confound in fmri_confounds if
                #                                  'global_signal' in confound or 'csf' in confound or 'white_matter' in confound]
                # reg_confounds = [confound for confound in fmri_confounds if
                #                  'motion_outlier' not in confound and 'non_steady_state_outlier' not in confound and 'cosine' not in confound]
                # reg_confounds = fmri_confounds
                reg_confounds = [confound for confound in fmri_confounds if 'trans' in confound or 'rot' in confound]
                reg_confounds = reg_confounds + [confound for confound in fmri_confounds if
                                                 'global_signal' in confound or 'csf' in confound or 'white_matter' in confound]
                reg_confounds = [confound for confound in reg_confounds if 'power2' not in confound]
                regressors = df_fmri[reg_confounds].to_numpy()
                regressors[np.isnan(regressors)] = 0

                # PCA
                # Generate PCA regressors of bpss nifti.
                pca_regressor = regressors_PCA(bpss_path, str(mask_path))
                reg_confounds = reg_confounds + [f'comp{i + 1}' for i in range(10)]
                regressors = np.hstack((regressors, pca_regressor))

                with resid6_confounds_path.open('w') as f:
                    writer = csv.writer(f, delimiter=' ')
                    writer.writerows(regressors)
                print(f'>>> {resid6_confounds_path}')

                # Prepare regressors datas for download
                download_all_regressors_path = Path(str(resid6_confounds_path).replace('.dat', '_download.txt'))
                num_row = len(regressors[:, 0])
                frame_no = np.arange(num_row).reshape((num_row, 1))
                download_regressors = np.concatenate((frame_no, regressors), axis=1)
                label_header = ['Frame'] + reg_confounds
                with download_all_regressors_path.open('w') as f:
                    csv.writer(f, delimiter=' ').writerows([label_header])
                    writer = csv.writer(f, delimiter=' ')
                    writer.writerows(download_regressors)
                print(f'>>> {download_all_regressors_path}')


def regression_fsaverage6(bids_dir, bold_preprocess_dir: Path, bold_smooth_dir: Path, subject_id):
    task = 'rest'

    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)

    sess = layout.get_session(subject=subj)
    for ses in sess:
        bids_bolds = layout.get(subject=subj, session=ses, task=task, suffix='bold', extension='.nii.gz')
        for bids_bold in bids_bolds:
            bold_path = Path(bids_bold.path)
            print(f'<<< {bold_path}')

            preprocess_func_path_dir = bold_preprocess_dir / f'sub-{subj}' / f'ses-{ses}' / 'func'
            bold_preproc_file = preprocess_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz'

            smooth_func_path_dir = bold_smooth_dir / f'sub-{subj}' / f'ses-{ses}' / 'func'
            smooth_surf_path_dir = bold_smooth_dir / f'sub-{subj}' / f'ses-{ses}' / 'surf'

            confounds_file = smooth_func_path_dir / bold_preproc_file.name.replace('space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz', 'desc-confounds_timeseries_resid6.dat')
            for lr in ['L', 'R']:
                if lr == 'L':
                    hemi = 'lh'
                else:
                    hemi = 'rh'
                orig_path = preprocess_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.gii'
                nii_path = smooth_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.nii.gz'
                if not nii_path.exists():
                    cmd = f'mri_surf2surf --srcsubject fsaverage6 --srcsurfval {orig_path} --trgsubject fsaverage6 --trgsurfval {nii_path} --hemi {hemi}'
                    os.system(cmd)
                    print(f'>>> {nii_path}')

                bpss_path = smooth_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.bpss.nii.gz'
                if not bpss_path.exists():
                    entities = dict(bids_bold.entities)
                    TR = layout.get_tr(entities)
                    bandpass_nifti(nii_path, bpss_path, TR)
                    print(f'>>> {bpss_path}')

                resid_path = smooth_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.bpss_resid6.nii.gz'
                if not resid_path.exists():
                    glm_nifti(str(bpss_path), str(resid_path), confounds_file)
                    print(f'>>> {resid_path}')


def fsaverage6_smooth_and_project_fs5_fs4(bids_dir, bold_preprocess_dir: Path, bold_smooth_dir: Path, subject_id):
    task = 'rest'

    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)

    sess = layout.get_session(subject=subj)
    for ses in sess:
        bids_bolds = layout.get(subject=subj, session=ses, task=task, suffix='bold', extension='.nii.gz')
        for bids_bold in bids_bolds:
            bold_path = Path(bids_bold.path)
            print(f'<<< {bold_path}')

            # preprocess_func_path_dir = bold_preprocess_dir / f'sub-{subj}' / f'ses-{ses}' / 'func'
            # bold_preproc_file = preprocess_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz'

            # smooth_func_path_dir = bold_smooth_dir / f'sub-{subj}' / f'ses-{ses}' / 'func'
            smooth_surf_path_dir = bold_smooth_dir / f'sub-{subj}' / f'ses-{ses}' / 'surf'

            for lr in ['L', 'R']:
                if lr == 'L':
                    hemi = 'lh'
                else:
                    hemi = 'rh'

                resid_path = smooth_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.bpss_resid.nii.gz'
                sm6_path = smooth_surf_path_dir / resid_path.name.replace(".nii.gz", "_sm6.nii.gz")
                if not sm6_path.exists():
                    smooth_fs6(resid_path, hemi)
                    print(f'>>> {sm6_path}')

                # down_sample
                fs5_surf_path = smooth_surf_path_dir / sm6_path.name.replace(".nii.gz", "_fsaverage5.nii.gz")
                fs4_surf_path = smooth_surf_path_dir / sm6_path.name.replace(".nii.gz", "_fsaverage4.nii.gz")
                if not (fs5_surf_path.exists() or fs4_surf_path.exists()):
                    downsample_fs6_to_fs4(sm6_path, hemi)
                    print(f'>>> {fs5_surf_path}')
                    print(f'>>> {fs4_surf_path}')


def cal_fwhm(bids_dir, bold_preprocess_dir: Path, bold_smooth_dir: Path, subject_id):
    task = 'rest'

    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)

    sess = layout.get_session(subject=subj)
    datas = []
    for ses in sess:
        bids_bolds = layout.get(task=task, suffix='bold', extension='.nii.gz')
        for bids_bold in bids_bolds:
            bold_path = Path(bids_bold.path)
            print(f'<<< {bold_path}')

            subj = bold_path.name.split('_')[0].split('-')[1]
            ses = bold_path.name.split('_')[1].split('-')[1]

            smooth_surf_path_dir = bold_smooth_dir / f'sub-{subj}' / f'ses-{ses}' / 'surf'
            smooth_fwhm_path_dir = bold_smooth_dir / f'sub-{subj}' / f'ses-{ses}' / 'fwhm'

            smooth_fwhm_path_dir.mkdir(parents=True, exist_ok=True)

            for lr in ['L', 'R']:
                if lr == 'L':
                    hemi = 'lh'
                else:
                    hemi = 'rh'
                data_dict = {'subject': f'{bold_path.name.replace(".nii.gz", "")}_hemi-{hemi}'}
                orig_path = smooth_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.nii.gz'
                bpss_path = smooth_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.bpss.nii.gz'
                resid_path = smooth_surf_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.bpss_resid.nii.gz'
                sm6_path = smooth_surf_path_dir / resid_path.name.replace(".nii.gz", "_sm6.nii.gz")

                for file_path, filetype in zip([orig_path, bpss_path, resid_path, sm6_path], ['orig', 'bpss', 'resid', 'sm6']):
                    fwhm_path = smooth_fwhm_path_dir / file_path.name.replace('.nii.gz', '.txt')
                    if not fwhm_path.exists():
                        cmd = f'mris_fwhm --i {file_path} --subject fsaverage6 --hemi lh >> {fwhm_path}'
                        os.system(cmd)
                        print(f'>>> {fwhm_path}')
                    with open(fwhm_path, 'r') as f:
                        lines = f.readlines()
                        print(f'<<< {fwhm_path}')
                        data_dict[filetype] = float(lines[-1].split('=')[-1].strip())

                # down_sample
                fs4_surf_path = smooth_surf_path_dir / sm6_path.name.replace(".nii.gz", "_fsaverage4.nii.gz")
                fwhm_path = smooth_fwhm_path_dir / fs4_surf_path.name.replace('.nii.gz', '.txt')
                if not fwhm_path.exists():
                    cmd = f'mris_fwhm --i {fs4_surf_path} --subject fsaverage4 --hemi lh >> {fwhm_path}'
                    os.system(cmd)
                    print(f'>>> {fwhm_path}')
                with open(fwhm_path, 'r') as f:
                    lines = f.readlines()
                    print(f'<<< {fwhm_path}')
                    data_dict['fsaverage4'] = lines[-1].split('=')[-1].strip()
                datas.append(data_dict)
        break
    import pandas as pd
    df = pd.DataFrame.from_dict(datas)
    df.to_csv(bold_smooth_dir / 'fwhm.csv', index=False)
    print()


if __name__ == '__main__':
    from interface.run import set_envrion

    set_envrion()

    # data_path = Path('/mnt/ngshare/fMRIPrep_UKB_150/BIDS')
    # bold_preprocess_result_path = Path('/mnt/ngshare/fMRIPrep_UKB_150/UKB_150_BoldPreprocess')
    # bold_result_path = Path('/mnt/ngshare/fMRIPrep_UKB_150/UKB_150_BoldPreprocess_bpss_resid_smooth')

    # bids_path = Path('/mnt/ngshare/Data_Orig/MSC')
    # DeepPrep_bold_preprocess_result_path = Path('/mnt/ngshare2/MSC_all/MSC_output')
    # fMRIPrep_bold_fsaverage6_result_path = Path('/mnt/ngshare/testfMRIPrep_space/Bold_Preprocess')
    # fMRIPrep_bold_preprocess_result_path = Path('/mnt/ngshare2/weiweiMSC_all/MSC_output')
    # fMRIPrep_bold_preprocess_smooth_result_path = Path('/mnt/ngshare2/weiweiMSC_all/MSC_output_BoldPreprocess_bpss_resid_smooth')
    # DeepPrep_confounds_path = Path('/mnt/ngshare2/MSC_all/MSC_BoldPreprocess')

    bids_path = Path('/mnt/ngshare/Data_Orig/HNU_1')
    DeepPrep_bold_preprocess_result_path = Path('/mnt/ngshare2/MSC_all/MSC_output')
    fMRIPrep_bold_fsaverage6_result_path = Path('/mnt/ngshare/testfMRIPrep_space/Bold_Preprocess')
    fMRIPrep_bold_preprocess_result_path = Path('/mnt/ngshare2/weiweiMSC_all/MSC_output')
    fMRIPrep_bold_preprocess_smooth_result_path = Path(
        '/mnt/ngshare2/weiweiMSC_all/MSC_output_BoldPreprocess_bpss_resid_smooth')
    DeepPrep_confounds_path = Path('/mnt/ngshare2/MSC_all/MSC_BoldPreprocess')

    layout = bids.BIDSLayout(str(bids_path))
    subjects = sorted(layout.get_subjects())

    args = []
    for subj in subjects:
        subject_id = f'sub-{subj}'
        regression_MNI152_by_DeepPrep_confounds(bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id, DeepPrep_confounds_path)
        args.append([bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id, DeepPrep_confounds_path])

        regression_fsaverage6_by_DeepPrep_confounds(bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id, DeepPrep_confounds_path)
        args.append([bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id, DeepPrep_confounds_path])

        # cal_regression(bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id)
        # args.append([bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id])

        # regression_fsaverage6(bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id)
        # args.append([bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id])

        # fsaverage6_smooth_and_project_fs5_fs4(bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id)
        # args.append([bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id])

        # cal_fwhm(bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id)
        # args.append([bids_path, fMRIPrep_bold_preprocess_result_path, fMRIPrep_bold_preprocess_smooth_result_path, subject_id])

    # pool = Pool(10)
    # pool.starmap(cal_fwhm, args)
    # pool.close()
    # pool.join()

    # pool = Pool(5)
    # pool.starmap(regression_fsaverage6, args)
    # pool.close()
    # pool.join()
