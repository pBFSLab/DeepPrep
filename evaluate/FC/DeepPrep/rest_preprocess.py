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


def rest_preprocess(bids_path, bold_preprocess_dir: Path, subject_id):
    task = 'rest'

    subject_dir = Path(bold_preprocess_dir) / subject_id
    subj = subject_id.split('-')[1]
    layout = bids.BIDSLayout(str(bids_path), derivatives=False)
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
        mc_path = subj_func_dir / bold_file.name.replace('.nii.gz', '_skip_reorient_faln_mc.nii.gz')
        bpss_path = subj_func_dir / mc_path.name.replace('.nii.gz', '_bpss.nii.gz')

        if not bpss_path.exists():
            bold = ants.image_read(str(bold_file))
            TR = bold.spacing[3]
            bandpass_nifti(str(mc_path), TR)

        all_regressors = compile_regressors(subj_func_dir, subject_id, bold_file, bpss_path)
        regression(str(bpss_path), all_regressors)

        bold_files.append(bold_file)
        args.append([subj_func_dir, bold_file])


def main():
    bids_path = Path('/mnt/ngshare/DeepPrep_workflow_test/UKB_BIDS')
    bold_preprocess_dir = Path('/mnt/ngshare/DeepPrep_workflow_test/UKB_BoldPreprocess')
    subject_id = 'sub-1000037-ses-02'
    rest_preprocess(bids_path, bold_preprocess_dir, subject_id)


if __name__ == '__main__':
    main()
