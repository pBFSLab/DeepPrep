import os
from pathlib import Path
import numpy as np
import ants
from scipy import stats
import sh


def sphere_mask(img: np.ndarray, pt, radius):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                dist = np.linalg.norm([pt[0] - i, pt[1] - j, pt[2] - k])
                if dist <= radius:
                    img[i, j, k] = 1
    return img


def create_sgACC_mask():
    MNI152_2mm_file = '/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz'
    MNI152_2mm_sgACC_mask_file = 'MNI152_T1_2mm_sgACC_mask.nii.gz'
    MNI152_2mm = ants.image_read(MNI152_2mm_file)
    sgACC_mask_np = np.zeros(MNI152_2mm.shape, dtype=np.float32)
    sgACC_mask_np = sphere_mask(sgACC_mask_np, (46, 71, 32), 2)
    sgACC_mask = ants.from_numpy(sgACC_mask_np, MNI152_2mm.origin, MNI152_2mm.spacing, MNI152_2mm.direction)
    ants.image_write(sgACC_mask, MNI152_2mm_sgACC_mask_file)


def load_vol_bolds(vol_bolds_path):
    pass


def compute_vol_fc(seed, vol_bold):
    n_i, n_j, n_k = vol_bold.shape[:3]
    vol_fc = np.zeros(shape=(n_i, n_j, n_k), dtype=np.float32)
    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                r, _ = stats.pearsonr(vol_bold[i, j, k, :], seed)
                vol_fc[i, j, k] = r
    return vol_fc


def compute_sgACC_fc():
    sgACC_seed_mask_file = 'MNI152_T1_2mm_sgACC_mask.nii.gz'
    sgACC_seed_mask = ants.image_read(str(sgACC_seed_mask_file))
    sgACC_seed_mask_np = sgACC_seed_mask.numpy()
    # MNI152 2mm bold path
    vol_bolds_path = ''
    vol_bold = load_vol_bolds(vol_bolds_path)  # 拼接好的vol bold数据
    sgACC_seed = vol_bold[sgACC_seed_mask_np == 1, :].mean(axis=0)
    vol_fc = compute_vol_fc(sgACC_seed, vol_bold)
    sgACC_fc = ants.from_numpy(vol_fc, sgACC_seed_mask.origin, sgACC_seed_mask.spacing, sgACC_seed_mask.direction)
    sgACC_fc_MNI152_2mm_file = 'sgACC_fc_MNI152_2mm.nii.gz'
    ants.image_write(sgACC_fc, str(sgACC_fc_MNI152_2mm_file))

    # 将vol空间计算的sgACC FC采样到surface
    subj = 'demo'
    lh_sgACC_fc_file = 'lh_sgACC_fc.mgh'
    # --trgsubject 可以为个体(subject)或模版(fsaverage)
    sh.mri_vol2surf('--mov', sgACC_fc_MNI152_2mm_file, '--mni152reg', '--trgsubject', subj, '--hemi', 'lh',
                    '--o', lh_sgACC_fc_file)
    rh_sgACC_fc_file = 'rh_sgACC_fc.mgh'
    # --trgsubject 可以为个体(subject)或模版(fsaverage)
    sh.mri_vol2surf('--mov', sgACC_fc_MNI152_2mm_file, '--mni152reg', '--trgsubject', subj, '--hemi', 'rh',
                    '--o', lh_sgACC_fc_file)


if __name__ == '__main__':
    # create_sgACC_mask()
    compute_sgACC_fc()
