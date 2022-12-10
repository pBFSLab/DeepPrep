import os

import bids
import ants

from app.filters.filters import bandpass_nifti
from app.regressors.regressors import *


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


def compile_regressors(func_path: Path, subject, bold_path: Path, bpss_path: Path):
    # Compile the regressors.
    movement_path = func_path / 'movement'
    movement_path.mkdir(exist_ok=True)

    fcmri_path = func_path / 'fcmri'
    fcmri_path.mkdir(exist_ok=True)

    # wipe mov regressors, if there
    mcdat_file = func_path / bold_path.name.replace('.nii.gz', '_skip_reorient_faln_mc.mcdat')
    mov_regressor_common_path = fcmri_path / ('%s_mov_regressor.dat' % subject)
    build_movement_regressors(subject, movement_path, fcmri_path, mcdat_file)
    mov_regressor_path = fcmri_path / ('%s_mov_regressor.dat' % subject)
    os.rename(mov_regressor_common_path, mov_regressor_path)

    mask_path = func_path / bold_path.name.replace('.nii.gz', '.brainmask.bin.nii.gz')
    out_path = fcmri_path / ('%s_WB_regressor_dt.dat' % subject)
    qnt_nifti(bpss_path, str(mask_path), out_path)

    mask_path = func_path / bold_path.name.replace('.nii.gz', '.func.ventricles.nii.gz')
    vent_out_path = fcmri_path / ('%s_ventricles_regressor_dt.dat' % subject)
    qnt_nifti(bpss_path, str(mask_path), vent_out_path)

    mask_path = func_path / bold_path.name.replace('.nii.gz', '.func.wm.nii.gz')
    wm_out_path = fcmri_path / ('%s_wm_regressor_dt.dat' % subject)
    qnt_nifti(bpss_path, str(mask_path), wm_out_path)

    pasted_out_path = fcmri_path / ('%s_vent_wm_dt.dat' % subject)
    with pasted_out_path.open('w') as f:
        sh.paste(vent_out_path, wm_out_path, _out=f)

    # Generate PCA regressors of bpss nifti.
    mask_path = func_path / bold_path.name.replace('.nii.gz', '.brainmask.nii.gz')
    pca_out_path = fcmri_path / ('%s_pca_regressor_dt.dat' % subject)
    regressors_PCA(bpss_path, str(mask_path), pca_out_path)

    fnames = [
        fcmri_path / ('%s_mov_regressor.dat' % subject),
        fcmri_path / ('%s_WB_regressor_dt.dat' % subject),
        fcmri_path / ('%s_vent_wm_dt.dat' % subject),
        fcmri_path / ('%s_pca_regressor_dt.dat' % subject)]
    all_regressors_path = fcmri_path / ('%s_regressors.dat' % subject)
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


def rest_cal_confounds(bids_dir, bold_preprocess_dir: Path, subject_id):
    task = 'rest'

    subject_dir = Path(bold_preprocess_dir) / subject_id
    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)
    subj_func_dir = Path(subject_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    if task is None:
        bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
    else:
        bids_bolds = layout.get(subject=subj, task=task, suffix='bold', extension='.nii.gz')
    for idx, bids_bold in enumerate(bids_bolds):
        bold_file = Path(bids_bold.path)
        print(f'<<< {bold_file}')
        mc_path = subj_func_dir / bold_file.name.replace('.nii.gz', '_skip_reorient_faln_mc.nii.gz')
        if not mc_path.exists():
            print(f'Exists Error: {mc_path}')
            continue

        bpss_path = subj_func_dir / mc_path.name.replace('.nii.gz', '_bpss.nii.gz')

        if not bpss_path.exists():
            bold_img = nib.load(bold_file)
            TR = bold_img.header.get_zooms()[3]
            bandpass_nifti(str(mc_path), TR)
            print(f'>>> {bpss_path}')

        all_regressors = compile_regressors(subj_func_dir, subject_id, bold_file, bpss_path)
        print(f'>>> {all_regressors}')


def regression_MNI152(bids_dir, bold_preprocess_dir: Path, subject_id):
    task = 'rest'

    subject_dir = Path(bold_preprocess_dir) / subject_id
    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)
    subj_func_dir = Path(subject_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    args = []
    bold_files = []
    if task is None:
        bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
    else:
        bids_bolds = layout.get(subject=subj, task=task, suffix='bold', extension='.nii.gz')
    for idx, bids_bold in enumerate(bids_bolds):
        bold_file = Path(bids_bold.path)
        print(f'<<< {bold_file}')
        mc_152_path = subj_func_dir / bold_file.name.replace('.nii.gz', '_skip_reorient_faln_mc_native_T1_2mm_MNI152_T1_2mm.nii.gz')
        if not mc_152_path.exists():
            print(f'Exists Error: {mc_152_path}')
            continue

        bpss_path = subj_func_dir / mc_152_path.name.replace('.nii.gz', '_bpss.nii.gz')

        if not bpss_path.exists():
            bold_img = nib.load(mc_152_path)
            TR = bold_img.header.get_zooms()[3]
            bandpass_nifti(str(mc_152_path), TR)
            print(f'>>> {bpss_path}')

        all_regressors = subj_func_dir / 'fcmri' / bold_file.name.replace('.nii.gz', '') / f'{subject_id}_regressors.dat'

        # regression orig_space bold
        resid_path = bpss_path.parent / bpss_path.name.replace('.nii.gz', '_resid.nii.gz')
        if not resid_path.exists():
            assert all_regressors.exists()
            glm_nifti(str(bpss_path), all_regressors)
            print(f'>>> {resid_path}')

        bold_files.append(bold_file)
        args.append([subj_func_dir, bold_file])


def regression_surface(bids_dir, bold_preprocess_dir: Path, subject_id):
    task = 'rest'

    subject_dir = Path(bold_preprocess_dir) / subject_id
    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)
    subj_func_dir = Path(subject_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    subj_surf_dir = Path(subject_dir) / 'surf'
    subj_surf_dir.mkdir(parents=True, exist_ok=True)

    if task is None:
        bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
    else:
        bids_bolds = layout.get(subject=subj, task=task, suffix='bold', extension='.nii.gz')
    for idx, bids_bold in enumerate(bids_bolds):
        bold_file = Path(bids_bold.path)
        print(f'<<< {bold_file}')

        for hemi in ['lh', 'rh']:
            surf_file_name = f'{hemi}.' + str(bold_file.name.replace('.nii.gz', '_skip_reorient_faln_mc_fsaverage6.nii.gz'))
            surf_path = subj_surf_dir / surf_file_name
            if not surf_path.exists():
                print(f'Exists Error: {surf_path}')
                continue

            bpss_path = subj_surf_dir / surf_file_name.replace('.nii.gz', '_bpss.nii.gz')
            if not bpss_path.exists():
                bold_img = nib.load(surf_path)
                TR = bold_img.header.get_zooms()[3]
                bandpass_nifti(str(surf_path), TR)
                print(f'>>> {bpss_path}')

            # regression orig_space bold
            all_regressors = subj_func_dir / 'fcmri' / bold_file.name.replace('.nii.gz', '') / f'{subject_id}_regressors.dat'
            resid_path = bpss_path.parent / bpss_path.name.replace('.nii.gz', '_resid.nii.gz')
            if not resid_path.exists():
                assert all_regressors.exists()
                glm_nifti(str(bpss_path), all_regressors)
                print(f'>>> {resid_path}')


def native_project_to_fs6(input_path, out_path, reg_path, hemi):
    """
    Project a volume (e.g., residuals) in native space onto the
    fsaverage6 surface.

    subject     - str. Subject ID.
    input_path  - Path. Volume to project.
    reg_path    - Path. Registaration .dat file.
    hemi        - str from {'lh', 'rh'}.

    Output file created at $DATA_DIR/surf/
     input_path.name.replace(input_path.ext, '_fsaverage6' + input_path.ext))
    Path object pointing to this file is returned.
    """
    sh.mri_vol2surf(
        '--mov', input_path,
        '--reg', reg_path,
        '--hemi', hemi,
        '--projfrac', 0.5,
        '--trgsubject', 'fsaverage6',
        '--o', out_path,
        '--reshape',
        '--interp', 'trilinear',
        _out=sys.stdout
    )
    return out_path


def native_project_to_native(input_path, out_path, reg_path, hemi):
    """
    Project a volume (e.g., residuals) in native space onto the
    native surface.

    input_path  - Path. Volume to project.
    reg_path    - Path. Registaration .dat file.
    hemi        - str from {'lh', 'rh'}.

    Output file created at $DATA_DIR/surf/
     input_path.name.replace(input_path.ext, '_fsaverage6' + input_path.ext))
    Path object pointing to this file is returned.
    """
    sh.mri_vol2surf(
        '--mov', input_path,
        '--reg', reg_path,
        '--hemi', hemi,
        '--projfrac', 0.5,
        '--o', out_path,
        '--reshape',
        '--interp', 'trilinear',
        _out=sys.stdout
    )
    return out_path


def mni152reg_project_to_fsaverage6(input_path, out_path, hemi):
    """
    Project a volume (e.g., residuals) in native space onto the
    native surface.

    input_path  - Path. Volume to project.
    reg_path    - Path. Registaration .dat file.
    hemi        - str from {'lh', 'rh'}.

    Output file created at $DATA_DIR/surf/
     input_path.name.replace(input_path.ext, '_fsaverage6' + input_path.ext))
    Path object pointing to this file is returned.
    """
    sh.mri_vol2surf(
        '--mov', input_path,
        '--mni152reg',
        '--hemi', hemi,
        '--projfrac', 0.5,
        '--trgsubject', 'fsaverage6',
        '--o', out_path,
        '--reshape',
        '--interp', 'trilinear',
        _out=sys.stdout
    )
    return out_path


def mni152reg_project_to_native(input_path, out_path, subject_id, hemi):
    """
    Project a volume (e.g., residuals) in native space onto the
    native surface.

    input_path  - Path. Volume to project.
    reg_path    - Path. Registaration .dat file.
    hemi        - str from {'lh', 'rh'}.

    Output file created at $DATA_DIR/surf/
     input_path.name.replace(input_path.ext, '_fsaverage6' + input_path.ext))
    Path object pointing to this file is returned.
    """
    sh.mri_vol2surf(
        '--mov', input_path,
        '--mni152reg',
        '--hemi', hemi,
        '--projfrac', 0.5,
        '--trgsubject', subject_id,
        '--o', out_path,
        '--reshape',
        '--interp', 'trilinear',
        _out=sys.stdout
    )
    return out_path


def project(bids_dir, bold_preprocess_dir: Path, subject_id,
            bold_ext: str = '_skip_reorient_faln_mc_bpss_resid.nii.gz'):
    """
    投影都是已mc为基础，加载register.dat文件，配准到T1空间
    默认采样到native空间
    指定trgsubject为fsaverage6，则project to fsaverage6 空间
    """
    from app.surface_projection.surface_projection import smooth_fs6, downsample_fs6_to_fs4
    task = 'rest'

    subject_dir = Path(bold_preprocess_dir) / subject_id
    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_dir), derivatives=False)
    subj_func_dir = Path(subject_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)
    subj_surf_dir = Path(subject_dir) / 'surf'
    subj_surf_dir.mkdir(parents=True, exist_ok=True)

    args = []
    bold_files = []
    if task is None:
        bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
    else:
        bids_bolds = layout.get(subject=subj, task=task, suffix='bold', extension='.nii.gz')
    for idx, bids_bold in enumerate(bids_bolds):
        bold_file = Path(bids_bold.path)
        print(f'<<< {bold_file}')
        # register.dat 包含了 subject_id 信息
        register_dat_path = subj_func_dir / bold_file.name.replace('.nii.gz',
                                                                   '_skip_reorient_faln_mc_bbregister.register.dat')
        bold_path = subj_func_dir / bold_file.name.replace('.nii.gz', bold_ext)

        if not register_dat_path.exists():
            print(f'Exists Error: {register_dat_path}')
            continue
        if not bold_path.exists():
            print(f'Exists Error: {bold_path}')
            continue

        for hemi in ['lh', 'rh']:
            # project to fsaverage6
            fs6_surf_path = subj_surf_dir / f'{hemi}.{bold_path.name.replace(".nii.gz", "_fsaverage6.nii.gz")}'
            if not fs6_surf_path.exists():
                native_project_to_fs6(bold_path, fs6_surf_path, register_dat_path, hemi)
                print(f'>>> {fs6_surf_path}')
            # smooth
            sm6_path = subj_surf_dir / fs6_surf_path.name.replace(".nii.gz", "_sm6.nii.gz")
            if not sm6_path.exists():
                smooth_fs6(fs6_surf_path, hemi)
                print(f'>>> {fs6_surf_path}')
            # down_sample
            fs5_surf_path = subj_surf_dir / sm6_path.name.replace(".nii.gz", "_fsaverage5.nii.gz")
            fs4_surf_path = subj_surf_dir / sm6_path.name.replace(".nii.gz", "_fsaverage4.nii.gz")
            if not (fs5_surf_path.exists() and fs4_surf_path.exists()):
                downsample_fs6_to_fs4(sm6_path, hemi)
                print(f'>>> {fs5_surf_path}')
                print(f'>>> {fs4_surf_path}')

            # project to native
            # surf_path = subj_surf_dir / f'{hemi}.{bold_path.name.replace(".nii.gz", "_native.nii.gz")}'
            # native_project_to_native(bold_path, surf_path, register_dat_path, hemi)


def main():
    from interface.run import set_envrion
    set_envrion()

    bids_dir = Path('/mnt/ngshare2/MSC_all/MSC')
    recon_dir = '/mnt/ngshare2/MSC_all/MSC_Recon'
    bold_preprocess_dir = Path('/mnt/ngshare2/MSC_all/MSC_BoldPreprocess')

    os.environ['SUBJECTS_DIR'] = recon_dir
    # subject_id = 'sub-1000037'
    args = []
    for subject_id in os.listdir(recon_dir):
        if not 'sub' in subject_id:
            continue
        if 'ses' in subject_id:
            continue
        if 'run' in subject_id:
            continue
        rest_cal_confounds(bids_dir, bold_preprocess_dir, subject_id)
        project(bids_dir, bold_preprocess_dir, subject_id)

        regression_MNI152(bids_dir, bold_preprocess_dir, subject_id)
        regression_surface(bids_dir, bold_preprocess_dir, subject_id)
        args.append([bids_dir, bold_preprocess_dir, subject_id])

    from multiprocessing.pool import Pool
    pool = Pool(3)
    pool.starmap(regression_MNI152, args)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
