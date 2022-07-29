import os
import shutil
from pathlib import Path
import glob
import ants
import numpy as np
from scipy import stats
import bids
import sh


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


def compute_vol_fc(seed, vol_bold):
    n_i, n_j, n_k = vol_bold.shape[:3]
    vol_fc = np.zeros(shape=(n_i, n_j, n_k), dtype=np.float32)
    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                r, _ = stats.pearsonr(vol_bold[i, j, k, :], seed)
                vol_fc[i, j, k] = r
    return vol_fc


def compute_surf_fc(seed, surf_bold):
    n_vertex = surf_bold.shape[0]
    surf_fc = np.zeros((n_vertex))
    for i in range(n_vertex):
        r, _ = stats.pearsonr(surf_bold[i, :], seed)
        surf_fc[i] = r
    return surf_fc


def batch_vol_fc(pipeline):
    data_path = Path('/mnt/ngshare/DeepPrep/MSC')
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
            vol_bold_files = sorted(subj_path.glob('ses-func*/func/*task-rest_bold_resid_MIN2mm_sm6.nii.gz'))[:2]
        elif pipeline == 'app':
            vol_bold_files = sorted(subj_path.glob('*/Preprocess/vol/*_resid_FS1mm_MNI1mm_MNI2mm_sm6_*.nii.gz'))
            bld_ids = list()
            for vol_bold_file in vol_bold_files:
                file_name = vol_bold_file.name
                bld_idx = vol_bold_file.name.find('bld')
                bld_ids.append(file_name[bld_idx + 3:bld_idx + 6])
            bld_selected = sorted(list(set(bld_ids)))[0:2]
            vol_bold_files = [vol_bold_file for vol_bold_file, bld_id in zip(vol_bold_files, bld_ids) if
                              bld_id in bld_selected]
        elif pipeline == 'fmriprep':
            vol_bold_files = sorted(subj_path.glob(
                'ses-func*/func/*task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz'))[:2]
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


def batch_surf_fc(pipeline):
    data_path = Path('/mnt/ngshare/DeepPrep/MSC')
    save_path = data_path / 'derivatives' / 'analysis' / pipeline
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
        workdir = Path('workdir')
        # surf FC
        subj_path = data_path / 'derivatives' / pipeline / f'sub-{subj}'
        subj_save_path = save_path / f'sub-{subj}'
        subj_save_path.mkdir(exist_ok=True)
        if pipeline == 'deepprep':
            lh_surf_bold_files = sorted(
                subj_path.glob('*/surf/lh.*_task-rest*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:2]
            rh_surf_bold_files = sorted(
                subj_path.glob('*/surf/rh.*_task-rest*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:2]
        elif pipeline == 'app':
            lh_surf_bold_files = sorted(
                subj_path.glob('*/Preprocess/surf/lh.*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:2]
            rh_surf_bold_files = sorted(
                subj_path.glob('*/Preprocess/surf/rh.*_resid_fsaverage6_sm6_fsaverage4.nii.gz'))[:2]
        elif pipeline == 'fmriprep':
            workdir.mkdir(exist_ok=True)
            vol_bold_files = sorted(subj_path.glob(
                'ses-func*/func/*task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz'))[:2]
            for vol_bold_file in vol_bold_files:
                lh_surf_fc_file = workdir / f'lh_{vol_bold_file.name}'
                sh.mri_vol2surf('--mov', vol_bold_file, '--mni152reg', '--trgsubject', 'fsaverage4', '--hemi', 'lh',
                                '--o', lh_surf_fc_file)
                rh_surf_fc_file = workdir / f'rh_{vol_bold_file.name}'
                sh.mri_vol2surf('--mov', vol_bold_file, '--mni152reg', '--trgsubject', 'fsaverage4', '--hemi', 'rh',
                                '--o', rh_surf_fc_file)
            lh_surf_bold_files = sorted(workdir.glob('lh_*.nii.gz'))[:2]
            rh_surf_bold_files = sorted(workdir.glob('rh_*.nii.gz'))[:2]
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


if __name__ == '__main__':
    pipeline = 'app'
    batch_vol_fc(pipeline)
    # batch_surf_fc(pipeline)
