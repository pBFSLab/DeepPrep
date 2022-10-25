import os
import time
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
import nibabel as nib
import numpy as np


def reshape_bold(bold_file):
    bold = nib.load(bold_file).get_fdata()
    n_frame = bold.shape[3]
    n_vertex = bold.shape[0] * bold.shape[1] * bold.shape[2]
    bold_surf = bold.reshape((n_vertex, n_frame), order='F')
    return bold_surf


def get_roi_mean(bold_surf, annot, hemi=None):
    bold_roi_mean = dict()
    keys = np.unique(annot)
    keys = np.sort(keys)
    for key in keys:
        if key < 0:
            continue
        index = annot == key
        if not np.any(index):
            continue
        bold_roi_mean[f'{hemi}_{key}'] = np.mean(bold_surf[index, :], axis=0)
    return bold_roi_mean


if __name__ == '__main__':
    csv_data = []
    sub_list = []
    bold_data_dir = Path('/run/user/1000/gvfs/sftp:host=30.30.30.66,user=zhenyu/mnt/ngshare2/UKB/parc92_UKB')
    recon_data_dir = Path('/run/user/1000/gvfs/sftp:host=30.30.30.66,user=zhenyu/mnt/ngshare2/UKB/parc92_UKB')
    list_txt = '/run/user/1000/gvfs/sftp:host=30.30.30.66,user=zhenyu/mnt/ngshare2/UKB_50.txt'
    aparc_num = 92

    with open(list_txt) as f:
        for line in f:
            sub_list.append(line.strip())

    for sub in sub_list:
        start_time = time.time()
        print(sub, start_time)
        lh_bold_file = bold_data_dir / sub / f'lh.{sub}_ses-02_task-rest_run-01_bold_resid_fsaverage6.nii.gz'
        rh_bold_file = bold_data_dir / sub / f'rh.{sub}_ses-02_task-rest_run-01_bold_resid_fsaverage6.nii.gz'
        lh_92_parc_annot_file = recon_data_dir / sub / 'lh_parc92_fs6_surf.annot'
        rh_92_parc_annot_file = recon_data_dir / sub / 'rh_parc92_fs6_surf.annot'

        lh_92_parc_annot = nib.freesurfer.read_annot(str(lh_92_parc_annot_file))[0]
        rh_92_parc_annot = nib.freesurfer.read_annot(str(rh_92_parc_annot_file))[0]

        lh_92_parc_surf = reshape_bold(lh_bold_file)
        rh_92_parc_surf = reshape_bold(rh_bold_file)

        all_lh_92_parc_mean = get_roi_mean(lh_92_parc_surf, lh_92_parc_annot, hemi='lh')
        all_rh_92_parc_mean = get_roi_mean(rh_92_parc_surf, rh_92_parc_annot, hemi='rh')

        all_92_parc_mean = all_lh_92_parc_mean.copy()
        all_92_parc_mean.update(all_rh_92_parc_mean)

        keys = list(all_92_parc_mean.keys())

        pc_values = np.zeros((aparc_num, aparc_num))
        for i in range(aparc_num):
            for j in range(aparc_num):
                x = all_92_parc_mean[keys[i]]
                y = all_92_parc_mean[keys[j]]
                print(x.shape, y.shape)
                pc, _ = pearsonr(x, y)
                pc_values[i][j] = pc
        print('end', time.time() - start_time)

        csv_data.append(pc_values[np.tril_indices_from(pc_values, -1)])
    df = pd.DataFrame(csv_data)
    df.to_csv('/mnt/ngshare2/UKB_50_test.csv')
    print()
