import sys
import sh
from pathlib import Path
import bids
import ants
import numpy as np
from app_pipeline.filters.filters import gauss_nifti
from app_pipeline.filters.filters import _bandpass_nifti
from app_pipeline.regressors.regressors import regression, compile_regressors
# from deepprep_pipeline.bold_preprocess import bold_smooth_6_ants
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


def mimic_fmriprep_regressors(bold_preproc_path, mask_path, confounds_path, bpss_path, TR):
    # bold_preproc_file = derivatives/fmriprep/sub-MSC01/{sess}/func/sub-{subj}_ses-{ses}_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz
    # confounds_path = derivatives/fmriprep/sub-MSC01/{sess}/func/sub-MSC01_{sess}_task-rest_desc-confounds_timeseries.tsv
    # mask_path = derivatives/fmriprep/sub-MSC01/{sess}/func/sub-MSC01_{sess}_task-rest_space-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz
    if not bpss_path.exists():
        bpss_path = bandpass_nifti(str(bold_preproc_path), bpss_path, TR)
        print(f'>>> {bpss_path}')

    resid_stem = 'resid6'
    resid_path = str(bpss_path).replace('.nii.gz', f'_{resid_stem}.nii.gz')
    if not Path(resid_path).exists():
        df_fmri = pd.read_csv(confounds_path, sep='\t')
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
        mask_path = mask_path
        # pca_out_path = fcmri_path / ('%s_bld%s_pca_regressor_dt.dat' % (subject, bldrun))
        pca_regressor = regressors_PCA(bpss_path, str(mask_path))
        regressors = np.hstack((regressors, pca_regressor))
        reg_confounds = reg_confounds + [f'comp{i + 1}' for i in range(10)]
        with open(f'{resid_stem}.txt', 'w') as f:
            f.write('\n'.join(reg_confounds))

        # Read NIFTI values.
        nifti_img = nib.load(bpss_path)
        nifti_data = nifti_img.get_fdata()
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
        print(f'>>> {resid_path}')

    sm6_file = str(resid_path).replace('.nii.gz', '_sm6.nii.gz')
    if not Path(sm6_file).exists():
        bold_smooth_6_ants(resid_path, sm6_file, verbose=True)
        print(f'>>> {sm6_file}')


def batch_run():
    data_path = Path('/home/weiwei/workdata/DeepPrep/workdir/ds000224')
    layout = bids.BIDSLayout(str(data_path))
    subjs = sorted(layout.get_subjects())
    for subj in subjs:
        sess = layout.get_session(subject=subj)
        for ses in sess:
            bids_bolds = layout.get(subject=subj, session=ses, task='rest', suffix='bold', extension='.nii.gz')
            for bids_bold in bids_bolds:
                print(bids_bold)
                entities = dict(bids_bold.entities)
                TR = layout.get_tr(entities)
                bold_preproc_file = data_path / 'derivatives' / 'fmriprep' / f'sub-{subj}' / f'ses-{ses}' / 'func' / f'sub-{subj}_ses-{ses}_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz'

                # mc_path = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'
                # mc_path = bold_preproc_file
                # gauss_path = gauss_nifti(str(mc_path), 1000000000)

                # bandpass_filtering
                # bpss_path = bandpass_nifti(gauss_path, TR)
                # bpss_path = bandpass_nifti(str(bold_preproc_file), TR)
                bpss_path = str(bold_preproc_file).replace('.nii.gz', '_bpss.nii.gz')
                sm6_file = bpss_path.replace('.nii.gz', '_sm6.nii.gz')
                bold_smooth_6_ants(bpss_path, sm6_file, verbose=True)

    # compile_regressors, regression
    # fcmri_dir = Path('fcmri')
    # fcmri_dir.mkdir(exist_ok=True)
    # bold_dir = preprocess_dir / subj / 'bold'
    # all_regressors_path = compile_regressors(preprocess_dir, bold_dir, run, subj, fcmri_dir, bpss_path)
    # regression(bpss_path, all_regressors_path)


if __name__ == '__main__':
    from interface.run import set_envrion

    set_envrion()

    data_path = Path('/mnt/ngshare/fMRIPrep_UKB_150/BIDS')
    bold_preprocess_result_path = Path('/mnt/ngshare/fMRIPrep_UKB_150/UKB_150_BoldPreprocess')
    bold_result_path = Path('/mnt/ngshare/fMRIPrep_UKB_150/UKB_150_BoldPreprocess_bpss_resid_smooth')

    # data_path = Path('/mnt/ngshare2/MSC_all/MSC')
    # bold_preprocess_result_path = Path('/mnt/ngshare2/MSC_all/MSC_output')
    # bold_result_path = Path('/mnt/ngshare2/MSC_all/MSC_output_BoldPreprocess_bpss_resid_smooth')

    layout = bids.BIDSLayout(str(data_path))
    subjs = sorted(layout.get_subjects())
    args_mimic_fmriprep_regressors = list()
    args_project_surface_fmriprep = list()
    for subj in subjs:
        sess = layout.get_session(subject=subj)
        for ses in sess:
            bids_bolds = layout.get(subject=subj, session=ses, task='rest', suffix='bold', extension='.nii.gz')
            for bids_bold in bids_bolds:
                print(bids_bold)
                entities = dict(bids_bold.entities)
                TR = layout.get_tr(entities)

                preprocess_func_path_dir = bold_preprocess_result_path / f'sub-{subj}' / f'ses-{ses}' / 'func'
                preprocess_surf_path_dir = bold_preprocess_result_path / f'sub-{subj}' / f'ses-{ses}' / 'surf'

                result_func_path_dir = bold_result_path / f'sub-{subj}' / f'ses-{ses}' / 'func'
                result_surf_path_dir = bold_result_path / f'sub-{subj}' / f'ses-{ses}' / 'surf'

                result_func_path_dir.mkdir(parents=True, exist_ok=True)
                result_surf_path_dir.mkdir(parents=True, exist_ok=True)

                bold_preproc_file = preprocess_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz'
                mask_path = preprocess_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz'
                confounds_path = preprocess_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_run-01_desc-confounds_timeseries.tsv'

                bpss_path = result_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss.nii.gz'
                bold_resid_file = result_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss_resid6.nii.gz'
                bold_sm6_file = result_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss_resid6_sm6.nii.gz'
                # mimic_fmriprep_regressors(bold_preproc_file, mask_path, confounds_path, bpss_path, TR)

                if not bold_preproc_file.exists():
                    print(f'ERROR: {bold_preproc_file}')
                    continue

                lh_project_surface = result_surf_path_dir / f'lh.sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss_resid6_fsaverage6.nii.gz'
                rh_project_surface = result_surf_path_dir / f'rh.sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss_resid6_fsaverage6.nii.gz'
                if not lh_project_surface.exists():
                    # project_surface_fmriprep(bold_resid_file, 'lh', lh_project_surface)
                    args_project_surface_fmriprep.append([bold_resid_file, 'lh', lh_project_surface])
                if not rh_project_surface.exists():
                    # project_surface_fmriprep(bold_resid_file, 'rh', rh_project_surface)
                    args_project_surface_fmriprep.append([bold_resid_file, 'rh', lh_project_surface])

                lh_project_surface = result_surf_path_dir / f'lh.sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss_resid6_sm6_fsaverage6.nii.gz'
                rh_project_surface = result_surf_path_dir / f'rh.sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss_resid6_sm6_fsaverage6.nii.gz'
                if not lh_project_surface.exists():
                    # project_surface_fmriprep(bold_resid_file, 'lh', lh_project_surface)
                    args_project_surface_fmriprep.append([bold_sm6_file, 'lh', lh_project_surface])
                if not rh_project_surface.exists():
                    # project_surface_fmriprep(bold_resid_file, 'rh', rh_project_surface)
                    args_project_surface_fmriprep.append([bold_sm6_file, 'rh', lh_project_surface])

                args_mimic_fmriprep_regressors.append([bold_preproc_file, mask_path, confounds_path, bpss_path, TR])
                # try:
                #     mimic_fmriprep_regressors(bold_preproc_file, mask_path, confounds_path, bpss_path, TR)
                #     lh_project_surface = result_surf_path_dir / f'lh.sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss_resid6_fsaverage6.nii.gz'
                #     rh_project_surface = result_surf_path_dir / f'rh.sub-{subj}_ses-{ses}_task-rest_run-01_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss_resid6_fsaverage6.nii.gz'
                #     if not lh_project_surface.exists():
                #         project_surface(bold_resid_file, 'lh', lh_project_surface)
                #     if not rh_project_surface.exists():
                #         project_surface(bold_resid_file, 'rh', rh_project_surface)
                # except Exception as why:
                #     print(why)
                #     print(bold_preproc_file)
                #     print(mask_path)
                #     print(confounds_path)
                #     break
                print()
    pool = Pool(5)
    pool.starmap(mimic_fmriprep_regressors, args_mimic_fmriprep_regressors)
    pool.close()
    pool.join()
    pool = Pool(10)
    pool.starmap(project_surface_fmriprep, args_project_surface_fmriprep)
    pool.close()
    pool.join()
