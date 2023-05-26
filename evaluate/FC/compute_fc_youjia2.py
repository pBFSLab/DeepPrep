import os
import shutil
from pathlib import Path
import glob
import ants
import numpy as np
from scipy import stats
# from scipy.stats import PearsonRConstantInputWarning, PearsonRNearConstantInputWarning
import warnings
import bids
import sh
import pandas as pd


def set_environ():
    # FreeSurfer recon-all env
    os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
    os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
    os.environ['PATH'] = '/usr/local/freesurfer/bin:/usr/local/freesurfer/mni/bin:/usr/local/freesurfer/tktools:' + \
                         '/usr/local/freesurfer/fsfast/bin:' + os.environ['PATH']
    os.environ['MINC_BIN_DIR'] = '/usr/local/freesurfer/mni/bin'
    os.environ['MINC_LIB_DIR'] = '/usr/local/freesurfer/mni/lib'
    os.environ['PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'
    os.environ['MNI_PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'
    # FreeSurfer fsfast env
    os.environ['FSF_OUTPUT_FORMAT'] = 'nii.gz'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'


def load_vol_bolds(vol_bold_files):
    vol_bold = ants.image_read(str(vol_bold_files[0])).numpy()
    if len(vol_bold_files) > 1:
        for vol_bold_file in vol_bold_files[1:]:
            data = ants.image_read(str(vol_bold_file)).numpy()
            vol_bold = np.concatenate((vol_bold, data), axis=3)
    return vol_bold


def load_surf_bolds(surf_bold_files):
    surf_bold = ants.image_read(str(surf_bold_files[0])).numpy()
    n_frame = surf_bold.shape[3]
    n_vertex = surf_bold.shape[0] * surf_bold.shape[1] * surf_bold.shape[2]
    surf_bold = surf_bold.reshape((n_vertex, n_frame), order='F')
    if len(surf_bold_files) > 1:
        for surf_bold_file in surf_bold_files[1:]:
            data = ants.image_read(str(surf_bold_file)).numpy()
            n_frame = data.shape[3]
            n_vertex = data.shape[0] * data.shape[1] * data.shape[2]
            surf_bold = np.concatenate((surf_bold, data.reshape((n_vertex, n_frame), order='F')), axis=1)
    return surf_bold


def compute_vol_fc(seed: np.ndarray, vol_bold: np.ndarray):
    n_frame = vol_bold.shape[3]
    n = len(seed)
    if n != n_frame:
        raise ValueError('seed and surf_bold must have the same number of frames.')

    if n <= 2:
        raise ValueError('n_frame must be greater than 2.')
    if len(vol_bold.shape) != 4:
        raise ValueError('vol_bold must be a 4-dimensional ndarray')
    dtype = np.float32
    seed_mean = seed.mean(dtype=dtype)
    vol_bold_mean = vol_bold.mean(dtype=dtype, axis=3)

    seed_m = (seed.astype(dtype) - seed_mean)[np.newaxis, np.newaxis, np.newaxis, :]
    vol_bold_m = vol_bold.astype(dtype) - vol_bold_mean[:, :, :, np.newaxis]

    norm_seed_m = np.linalg.norm(seed_m)
    norm_vol_bold_m = np.linalg.norm(vol_bold_m, axis=3)[:, :, :, np.newaxis]

    vol_fc = ((vol_bold_m / norm_vol_bold_m) * (seed_m / norm_seed_m)).sum(axis=3).clip(-1, 1)

    return vol_fc



def compute_surf_fc(seed: np.ndarray, surf_bold: np.ndarray):
    n_frame = surf_bold.shape[1]
    n = len(seed)
    if n != n_frame:
        raise ValueError('seed and surf_bold must have the same number of frames.')

    # If an input is constant, the correlation coefficient is not defined.
    if (seed == seed[0]).all() or (surf_bold == surf_bold[0][0]).all():
        warnings.warn(PearsonRConstantInputWarning())
        return None

    if n <= 2:
        raise ValueError('n_frame must be greater than 2.')
    dtype = np.float32
    seed_mean = seed.mean(dtype=dtype)
    surf_bold_mean = surf_bold.mean(dtype=dtype, axis=1).reshape((-1, 1))

    seed_m = seed.astype(dtype).reshape((-1, 1)) - seed_mean
    surf_bold_m = surf_bold.astype(dtype) - surf_bold_mean

    norm_seed_m = np.linalg.norm(seed_m)
    norm_surf_bold_m = np.linalg.norm(surf_bold_m, axis=1).reshape((-1, 1))

    threshold = 1e-13
    if np.any(norm_seed_m < threshold * abs(seed_m)) or np.any(norm_surf_bold_m < threshold * abs(surf_bold_m)):
        warnings.warn(PearsonRNearConstantInputWarning())

    surf_fc = np.dot(surf_bold_m / norm_surf_bold_m, seed_m / norm_seed_m).flatten().clip(-1, 1)

    return surf_fc


def batch_vol_fc(data_path: Path, pipeline, bold_num):
    save_path = data_path / 'derivatives' / 'analysis' / pipeline
    save_path.mkdir(parents=True, exist_ok=True)
    layout = bids.BIDSLayout(str(data_path))
    subjs = sorted(layout.get_subjects())
    if pipeline == 'app':
        MNI152_T1_2mm = ants.image_read('MNI2mm_template.nii.gz')
    else:
        MNI152_T1_2mm = ants.image_read('/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz')
    MNI152_T1_2mm_ref = ants.image_read('/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz')
    for subj in subjs:
        # vol FC
        subj_path = data_path / 'derivatives' / pipeline / f'sub-{subj}'
        subj_save_path = save_path / f'sub-{subj}'
        subj_save_path.mkdir(exist_ok=True)
        if pipeline == 'deepprep':
            vol_bold_files = sorted(subj_path.glob('ses-*/func/*task-rest_*bold_resid_MIN2mm_sm6.nii.gz'))[:bold_num]
        elif pipeline == 'app':
            vol_bold_files = sorted(subj_path.glob('*/Preprocess/vol/*_resid_FS1mm_MNI1mm_MNI2mm_sm6_*.nii.gz'))
            bld_ids = list()
            for vol_bold_file in vol_bold_files:
                file_name = vol_bold_file.name
                bld_idx = vol_bold_file.name.find('bld')
                bld_ids.append(file_name[bld_idx + 3:bld_idx + 6])
            bld_selected = sorted(list(set(bld_ids)))[0:bold_num]
            vol_bold_files = [vol_bold_file for vol_bold_file, bld_id in zip(vol_bold_files, bld_ids) if
                              bld_id in bld_selected]
        elif pipeline == 'fmriprep':
            vol_bold_files = sorted(subj_path.glob(
                'ses-*/func/*task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz'))[:bold_num]
        else:
            raise Exception()
        vol_bold = load_vol_bolds(vol_bold_files)
        if pipeline == 'app':
            seed_mask_files = (Path(__file__).parent / 'MNI_ROI_128').glob('*.nii.gz')
        else:
            seed_mask_files = (Path(__file__).parent / 'MNI_ROI').glob('*.nii.gz')
        for seed_mask_file in seed_mask_files:
            seed_mask = ants.image_read(str(seed_mask_file))
            seed_maks_np = seed_mask.numpy()
            seed = vol_bold[seed_maks_np == 1, :].mean(axis=0)
            vol_fc_np = compute_vol_fc(seed, vol_bold)
            vol_fc = ants.from_numpy(vol_fc_np, MNI152_T1_2mm.origin, MNI152_T1_2mm.spacing, MNI152_T1_2mm.direction)
            vol_fc_file = subj_save_path / f'fc_{seed_mask_file.name}'
            if pipeline == 'app':
                vol_fc_resample = ants.resample_image_to_target(vol_fc, MNI152_T1_2mm_ref)
                ants.image_write(vol_fc_resample, str(vol_fc_file))
                print(f'>>> {vol_fc_file}')
            else:
                ants.image_write(vol_fc, str(vol_fc_file))
                print(f'>>> {vol_fc_file}')
        # exit()

def select_vol_fc(data_path: Path, pipeline, bold_num):
    save_path = Path('/run/user/1000/gvfs/sftp:host=30.30.30.81,user=youjia/mnt/ngshare/DeepPrep/Validation/MSC_all/vol_fc', pipeline)
    save_path.mkdir(parents=True, exist_ok=True)
    # layout = bids.BIDSLayout(str(data_path))
    # subjs = sorted(layout.get_subjects())
    # df = pd.read_csv('/mnt/ngshare2/MSC_all/MSCsub_resid_29sub.csv') # select 29 subjects
    # fp_subjects = df['fmriPrep_path_sm2trg'].tolist()
    # dp_subjects = df['DeepPrep_path'].tolist()
    subjs = [f'sub-MSC0{i}' for i in range(1, 10)]
    subjs.append('sub-MSC10')
    print(subjs)

    if pipeline == 'app':
        MNI152_T1_2mm = ants.image_read('MNI2mm_template.nii.gz')
    else:
        MNI152_T1_2mm = ants.image_read('/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz')
    MNI152_T1_2mm_ref = ants.image_read('/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz')

    for subj in subjs:
        if subj not in ['sub-MSC04', 'sub-MSC05']:
            continue
        # vol FC
        subj_save_path = save_path / f'{subj}'
        subj_save_path.mkdir(exist_ok=True)
        if pipeline == 'deepprep':
            subj_path = Path('/run/user/1000/gvfs/sftp:host=30.30.30.81,user=youjia/mnt/ngshare2/MSC_all/deepprep_BoldPreprocess', subj)
            vol_bold_files = sorted(subj_path.glob('func/*_task-rest_bold_skip_reorient_faln_mc_native_T1_2mm_MNI152_T1_2mm_bpss_resid.nii.gz'))[:bold_num]
            print('deepprep', len(vol_bold_files))
        elif pipeline == 'fmriprep':
            subj_path = Path('/run/user/1000/gvfs/sftp:host=30.30.30.81,user=youjia/mnt/ngshare2/MSC_all/fmriprep_BoldPreprocess', subj)
            vol_bold_files = sorted(subj_path.glob('ses-*/func/*_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold_bpss_resid_sm2trg.nii.gz'))[:bold_num]
            print('fmriprep', len(vol_bold_files))
        else:  # app
            raise Exception()
        vol_bold = load_vol_bolds(vol_bold_files)
        if pipeline == 'app':
            seed_mask_files = (Path(__file__).parent / 'MNI_ROI_128').glob('*.nii.gz')
        else:
            seed_mask_files = (Path(__file__).parent / 'MNI_ROI').glob('*.nii.gz')
        for seed_mask_file in seed_mask_files:
            seed_mask = ants.image_read(str(seed_mask_file))
            seed_maks_np = seed_mask.numpy()
            seed = vol_bold[seed_maks_np == 1, :].mean(axis=0)
            vol_fc_np = compute_vol_fc(seed, vol_bold)
            vol_fc = ants.from_numpy(vol_fc_np, MNI152_T1_2mm.origin, MNI152_T1_2mm.spacing, MNI152_T1_2mm.direction)
            vol_fc_file = subj_save_path / f'fc_{seed_mask_file.name}'
            if pipeline == 'app':
                vol_fc_resample = ants.resample_image_to_target(vol_fc, MNI152_T1_2mm_ref)
                ants.image_write(vol_fc_resample, str(vol_fc_file))
                print(f'>>> {vol_fc_file}')
            else:
                ants.image_write(vol_fc, str(vol_fc_file))
                print(f'>>> {vol_fc_file}')
        # exit()


def batch_surf_fc(data_path: Path, bold_path: Path, pipeline, dataset):
    # save_path = data_path / 'derivatives' / 'analysis' / pipeline
    save_path = Path(f'/mnt/ngshare/DeepPrep/Validation/{dataset}/surf_fc/{pipeline}')
    save_path.mkdir(parents=True, exist_ok=True)
    layout = bids.BIDSLayout(str(data_path))
    subjs = sorted(layout.get_subjects())

    # fsaverage4
    # motor
    lh_motor_idx = 644
    rh_motor_idx = 220
    # ACC
    lh_ACC_idx = 1999
    rh_ACC_idx = 1267
    # PCC
    lh_PCC_idx = 1803
    rh_PCC_idx = 355

    lh_seeds = list()
    rh_seeds = list()
    lh_seeds.append({'name': 'LH_Motor', 'index': lh_motor_idx})
    rh_seeds.append({'name': 'RH_Motor', 'index': rh_motor_idx})
    lh_seeds.append({'name': 'LH_ACC', 'index': lh_ACC_idx})
    rh_seeds.append({'name': 'RH_ACC', 'index': rh_ACC_idx})
    lh_seeds.append({'name': 'LH_PCC', 'index': lh_PCC_idx})
    rh_seeds.append({'name': 'RH_PCC', 'index': rh_PCC_idx})
    set_environ()
    for subj in subjs:
        print(subj)
        workdir = Path('workdir')
        # surf FC
        subj_path = bold_path / f'sub-{subj}'
        subj_save_path = save_path / f'sub-{subj}'
        subj_save_path.mkdir(exist_ok=True)
        if pipeline == 'deepprep':
            lh_surf_bold_files = sorted(
                subj_path.glob('surf/lh.*_task-rest*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:bold_num]
            rh_surf_bold_files = sorted(
                subj_path.glob('surf/rh.*_task-rest*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:bold_num]
        elif pipeline == 'app':
            lh_surf_bold_files = sorted(
                subj_path.glob('*/Preprocess/surf/lh.*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:bold_num]
            rh_surf_bold_files = sorted(
                subj_path.glob('*/Preprocess/surf/rh.*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:bold_num]
        elif pipeline == 'fmriprep':
            workdir.mkdir(exist_ok=True)
            vol_bold_files = sorted(subj_path.glob(
                'func/*task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold_*_reorient.nii.gz'))[:bold_num]
            for vol_bold_file in vol_bold_files:
                lh_surf_fc_file = workdir / f'lh_{vol_bold_file.name}'
                sh.mri_vol2surf('--mov', vol_bold_file, '--mni152reg', '--trgsubject', 'fsaverage4', '--hemi', 'lh',
                                '--o', lh_surf_fc_file)
                rh_surf_fc_file = workdir / f'rh_{vol_bold_file.name}'
                sh.mri_vol2surf('--mov', vol_bold_file, '--mni152reg', '--trgsubject', 'fsaverage4', '--hemi', 'rh',
                                '--o', rh_surf_fc_file)
            lh_surf_bold_files = sorted(workdir.glob('lh_*.nii.gz'))
            rh_surf_bold_files = sorted(workdir.glob('rh_*.nii.gz'))
        else:
            raise Exception()
        lh_surf_bold = load_surf_bolds(lh_surf_bold_files)
        rh_surf_bold = load_surf_bolds(rh_surf_bold_files)
        if workdir.exists():
            shutil.rmtree(workdir)

        for lh_seed_dict in lh_seeds:
            seed_name = lh_seed_dict['name']
            seed_idx = lh_seed_dict['index']
            lh_seed = lh_surf_bold[seed_idx, :]
            lh_lh_fc = compute_surf_fc(lh_seed, lh_surf_bold)
            lh_rh_fc = compute_surf_fc(lh_seed, rh_surf_bold)
            lh_fc = ants.from_numpy(lh_lh_fc.reshape((-1, 1, 1)))
            lh_fc_file = save_path / f'sub-{subj}' / f"lh_{seed_name}_fc.mgh"
            ants.image_write(lh_fc, str(lh_fc_file))
            print(f'>>> {lh_fc_file}')
            rh_fc = ants.from_numpy(lh_rh_fc.reshape((-1, 1, 1)))
            rh_fc_file = save_path / f'sub-{subj}' / f"rh_{seed_name}_fc.mgh"
            ants.image_write(rh_fc, str(rh_fc_file))
            print(f'>>> {rh_fc_file}')

        for rh_seed_dict in rh_seeds:
            seed_name = rh_seed_dict['name']
            seed_idx = rh_seed_dict['index']
            rh_seed = rh_surf_bold[seed_idx, :]
            rh_lh_fc = compute_surf_fc(rh_seed, lh_surf_bold)
            rh_rh_fc = compute_surf_fc(rh_seed, rh_surf_bold)
            lh_fc = ants.from_numpy(rh_lh_fc.reshape((-1, 1, 1)))
            lh_fc_file = save_path / f'sub-{subj}' / f"lh_{seed_name}_fc.mgh"
            ants.image_write(lh_fc, str(lh_fc_file))
            print(f'>>> {lh_fc_file}')
            rh_fc = ants.from_numpy(rh_rh_fc.reshape((-1, 1, 1)))
            rh_fc_file = save_path / f'sub-{subj}' / f"rh_{seed_name}_fc.mgh"
            ants.image_write(rh_fc, str(rh_fc_file))
            print(f'>>> {rh_fc_file}')
        # exit()

def grouplevel_surf_fc(data_path: Path, bold_path: Path, pipeline, dataset):
    # save_path = data_path / 'derivatives' / 'analysis' / pipeline
    save_path = Path(f'/mnt/ngshare/DeepPrep/Validation/{dataset}/surf_fc/{pipeline}')
    save_path.mkdir(parents=True, exist_ok=True)
    layout = bids.BIDSLayout(str(data_path))
    subjs = sorted(layout.get_subjects())

    # fsaverage4
    # motor
    lh_motor_idx = 644
    rh_motor_idx = 220
    # ACC
    lh_ACC_idx = 1999
    rh_ACC_idx = 1267
    # PCC
    lh_PCC_idx = 1803
    rh_PCC_idx = 355

    lh_seeds = list()
    rh_seeds = list()
    lh_seeds.append({'name': 'LH_Motor', 'index': lh_motor_idx})
    rh_seeds.append({'name': 'RH_Motor', 'index': rh_motor_idx})
    lh_seeds.append({'name': 'LH_ACC', 'index': lh_ACC_idx})
    rh_seeds.append({'name': 'RH_ACC', 'index': rh_ACC_idx})
    lh_seeds.append({'name': 'LH_PCC', 'index': lh_PCC_idx})
    rh_seeds.append({'name': 'RH_PCC', 'index': rh_PCC_idx})
    set_environ()
    lh_surf_bold_files_all = []
    rh_surf_bold_files_all = []
    for subj in subjs:
        print(subj)
        workdir = Path('workdir')
        # surf FC
        subj_path = bold_path / f'sub-{subj}'
        subj_save_path = save_path / f'sub-{subj}'
        subj_save_path.mkdir(exist_ok=True)
        if pipeline == 'deepprep':
            lh_surf_bold_files = sorted(
                subj_path.glob('surf/lh.*_task-rest*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:bold_num]
            rh_surf_bold_files = sorted(
                subj_path.glob('surf/rh.*_task-rest*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:bold_num]
            lh_surf_bold_files_all.extend(lh_surf_bold_files)
            rh_surf_bold_files_all.extend(rh_surf_bold_files)
        elif pipeline == 'app':
            lh_surf_bold_files = sorted(
                subj_path.glob('*/Preprocess/surf/lh.*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:bold_num]
            rh_surf_bold_files = sorted(
                subj_path.glob('*/Preprocess/surf/rh.*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:bold_num]
        elif pipeline == 'fmriprep':
            workdir.mkdir(exist_ok=True)
            vol_bold_files = sorted(subj_path.glob(
                'func/*task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold_*_reorient.nii.gz'))[:bold_num]
            for vol_bold_file in vol_bold_files:
                lh_surf_fc_file = workdir / f'lh_{vol_bold_file.name}'
                sh.mri_vol2surf('--mov', vol_bold_file, '--mni152reg', '--trgsubject', 'fsaverage4', '--hemi', 'lh',
                                '--o', lh_surf_fc_file)
                rh_surf_fc_file = workdir / f'rh_{vol_bold_file.name}'
                sh.mri_vol2surf('--mov', vol_bold_file, '--mni152reg', '--trgsubject', 'fsaverage4', '--hemi', 'rh',
                                '--o', rh_surf_fc_file)
            lh_surf_bold_files = sorted(workdir.glob('lh_*.nii.gz'))
            rh_surf_bold_files = sorted(workdir.glob('rh_*.nii.gz'))
            lh_surf_bold_files_all.extend(lh_surf_bold_files)
            rh_surf_bold_files_all.extend(rh_surf_bold_files)
        else:
            raise Exception()

    lh_surf_bold = load_surf_bolds(lh_surf_bold_files_all)
    rh_surf_bold = load_surf_bolds(rh_surf_bold_files_all)
    if workdir.exists():
        shutil.rmtree(workdir)

    for lh_seed_dict in lh_seeds:
        seed_name = lh_seed_dict['name']
        seed_idx = lh_seed_dict['index']
        lh_seed = lh_surf_bold[seed_idx, :]
        lh_lh_fc = compute_surf_fc(lh_seed, lh_surf_bold)
        lh_rh_fc = compute_surf_fc(lh_seed, rh_surf_bold)
        lh_fc = ants.from_numpy(lh_lh_fc.reshape((-1, 1, 1)))
        lh_fc_file = save_path / f'{dataset}_grouplevel' / f"lh_{seed_name}_fc.mgh"
        ants.image_write(lh_fc, str(lh_fc_file))
        print(f'>>> {lh_fc_file}')
        rh_fc = ants.from_numpy(lh_rh_fc.reshape((-1, 1, 1)))
        rh_fc_file = save_path / f'{dataset}_grouplevel' / f"rh_{seed_name}_fc.mgh"
        ants.image_write(rh_fc, str(rh_fc_file))
        print(f'>>> {rh_fc_file}')

    for rh_seed_dict in rh_seeds:
        seed_name = rh_seed_dict['name']
        seed_idx = rh_seed_dict['index']
        rh_seed = rh_surf_bold[seed_idx, :]
        rh_lh_fc = compute_surf_fc(rh_seed, lh_surf_bold)
        rh_rh_fc = compute_surf_fc(rh_seed, rh_surf_bold)
        lh_fc = ants.from_numpy(rh_lh_fc.reshape((-1, 1, 1)))
        lh_fc_file = save_path / f'{dataset}_grouplevel' / f"lh_{seed_name}_fc.mgh"
        ants.image_write(lh_fc, str(lh_fc_file))
        print(f'>>> {lh_fc_file}')
        rh_fc = ants.from_numpy(rh_rh_fc.reshape((-1, 1, 1)))
        rh_fc_file = save_path / f'{dataset}_grouplevel' / f"rh_{seed_name}_fc.mgh"
        ants.image_write(rh_fc, str(rh_fc_file))
        print(f'>>> {rh_fc_file}')
    # exit()


if __name__ == '__main__':
    # data_path = Path('/mnt/ngshare/DeepPrep/MSC')
    data_path = Path('/mnt/ngshare2/MSC') # bids_path
    # bold_path = Path('/mnt/ngshare/DeepPrep_MSC_all/MSC_BoldPreprocess')
    bold_path = Path('/mnt/ngshare/fMRIPrep_MSC_all/MSC_BoldPreprocess')
    dataset = 'MSC_all'
    pipeline = 'deepprep'
    # pipeline = 'fmriprep'
    bold_num = 1
    select_vol_fc(data_path, pipeline, bold_num)
    # batch_surf_fc(data_path, bold_path, pipeline, dataset)
    # grouplevel_surf_fc(data_path, bold_path, pipeline, dataset)
