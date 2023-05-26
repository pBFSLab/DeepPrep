import os
import numpy as np
from pathlib import Path
import nibabel as nib
import pandas as pd


def set_environ():
    # FreeSurfer
    os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
    os.environ['PATH'] = '/usr/local/freesurfer/bin:' + os.environ['PATH']


def aparc_dice(ref_labels, mov_labels, names):
    label_ids = np.unique(ref_labels)
    label_ids = label_ids[1:]
    dice_dict = dict()
    for label_id in label_ids:
        # print(label_id, type(label_id), names[label_id])
        ref_label_mask = ref_labels == label_id
        mov_label_mask = mov_labels == label_id
        label_dice = (2 * (np.sum(ref_label_mask & mov_label_mask))) / (np.sum(ref_label_mask) + np.sum(mov_label_mask))
        if label_id == -1:
            label_name = names[0]
        else:
            label_name = names[label_id]
        dice_dict[label_name] = label_dice

    dice_mean = np.sum([dice_dict[names[label_id]] for label_id in label_ids]) / len(label_ids)
    dice_dict['dice_mean'] = dice_mean
    dice_whole = np.sum(ref_labels == mov_labels) / ref_labels.size
    dice_dict['dice_whole'] = dice_whole

    return dice_dict


if __name__ == '__main__':
    sub_list_file = '/mnt/ngshare/SurfReg/Data_Extra/list/HNUcombine_all_list.txt'
    DP_recon_dir = Path('/mnt/ngshare/DeepPrep/ttest/HNU_DPtrt')  # 需要包含fsaverage6
    FS_recon_dir = Path('/mnt/ngshare/DeepPrep/ttest/HNU_1_combine_FS720')  # 需要包含fsaverage6
    workdir = Path('/mnt/ngshare/DeepPrep/ttest/resample_fsaverage6/HNU_DPtrt_FS')
    hemis = ['lh', 'rh']

    set_environ()

    sub_list = []
    with open(sub_list_file) as f:
        for line in f:
            sub_list.append(line.strip())

    for sub in sub_list:
        sub_aparc_output = workdir / sub / 'label'
        sub_aparc_output.mkdir(exist_ok=True, parents=True)
        for hemi in hemis:
            # DP的fs6的分区
            os.environ['SUBJECTS_DIR'] = str(DP_recon_dir)
            DP_aparc_outpath = sub_aparc_output / f'{hemi}.DP.aparc.annot'
            cmd_DP = f'mri_surf2surf --hemi {hemi} --srcsubject {sub} --sval-annot aparc.annot --trgsubject fsaverage6 --tval {DP_aparc_outpath}'
            os.system(cmd_DP)
            # FS的fs6的分区
            os.environ['SUBJECTS_DIR'] = str(FS_recon_dir)
            FS_aparc_outpath = sub_aparc_output / f'{hemi}.FS.aparc.annot'
            cmd_FS = f'mri_surf2surf --hemi {hemi} --srcsubject {sub} --sval-annot aparc.annot --trgsubject fsaverage6 --tval {FS_aparc_outpath}'
            os.system(cmd_FS)

    for hemi in hemis:
        dice_data_list = []
        for sub in sub_list:
            sub_DP_parc_annot = workdir / sub / 'label' / f'{hemi}.DP.aparc.annot'
            sub_FS_parc_annot = workdir / sub / 'label' / f'{hemi}.FS.aparc.annot'

            ref_labels, _, names = nib.freesurfer.read_annot(str(sub_DP_parc_annot))
            mov_labels, _, _ = nib.freesurfer.read_annot(str(sub_FS_parc_annot))

            ref_labels = ref_labels.astype(np.int32)
            mov_labels = mov_labels.astype(np.int32)
            names = [name.decode() for name in names]

            data_dict = dict()
            data_dict['subj'] = sub

            dice_dict = aparc_dice(ref_labels, mov_labels, names)
            data_dict.update(dice_dict)
            dice_data_list.append(data_dict)

        mean = np.mean([item['dice_mean'] for item in dice_data_list])
        mean_whole = np.mean([item['dice_whole'] for item in dice_data_list])
        items = {
            'subj': 'mean',
            'dice_mean': mean,
            'dice_whole': mean_whole
        }
        dice_data_list.append(items)
        df = pd.DataFrame(dice_data_list)
        file = workdir / f'{hemi}_dice.csv'
        df.to_csv(file, index=False)
