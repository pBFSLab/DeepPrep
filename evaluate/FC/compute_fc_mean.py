from pathlib import Path
import ants
import numpy as np


def conmpute_fc_vol_mean(model_files, data_list, save_path):
    MNI152_T1_2mm = ants.image_read('/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz')
    count = 0
    for data_file in model_files:
        if count == 0:
            save_file = str(data_file).split('/')[-1]
            save_path = save_path / save_file
        data = ants.image_read(str(data_file)).numpy()
        data_list.append(data)
        count += 1
    data_mean = np.mean(data_list, axis=0)
    data_mean_temp = ants.from_numpy(data_mean, MNI152_T1_2mm.origin, MNI152_T1_2mm.spacing, MNI152_T1_2mm.direction)
    ants.image_write(data_mean_temp, str(save_path))
    print(f'save_fc_mean_file >>> {save_path}')


def conmpute_fc_surf_mean(model_files, data_list, save_path):
    count = 0
    for data_file in model_files:
        if count == 0:
            save_file = str(data_file).split('/')[-1]
            save_path = save_path / save_file
        data = ants.image_read(str(data_file)).numpy()
        data_list.append(data)
        count += 1
    data_mean = np.mean(data_list, axis=0)
    data_mean_temp = ants.from_numpy(data_mean)
    ants.image_write(data_mean_temp, str(save_path))
    print(f'save_fc_mean_file >>> {save_path}')


def vol_fc_mean(data_path, pipeline):
    pipeline = pipeline
    save_path = data_path / 'sub-mean'
    save_path.mkdir(parents=True, exist_ok=True)

    sgACC_files = sorted(data_path.glob('*/fc_MNI152_T1_2mm_sgACC_mask.nii.gz'))
    ATN_LInsula_files = sorted(data_path.glob('*/fc_N12Trio_avg152T1_ATN_LInsula_reg8mm_-34_18_6_.nii.gz'))
    DN_LPCC_files = sorted(data_path.glob('*/fc_N12Trio_avg152T1_DN_LPCC_reg8mm_-2_-53_26_.nii.gz'))
    DN_RaMPFC_files = sorted(data_path.glob('*/fc_N12Trio_avg152T1_DN_RaMPFC_reg8mm_2_54_-4_.nii.gz'))
    DN_vMPFC_files = sorted(data_path.glob('*/fc_N12Trio_avg152T1_DN_vMPFC_reg8mm_0_42_-14_.nii.gz'))
    Motor_LHand_files = sorted(data_path.glob('*/fc_N12Trio_avg152T1_Motor_LHand_reg8mm_-42_-25_63_.nii.gz'))

    sgACC_mean = []
    ATN_LInsula_mean = []
    DN_LPCC_mean = []
    DN_RaMPFC_mean = []
    DN_vMPFC_mean = []
    Motor_LHand_mean = []

    conmpute_fc_vol_mean(sgACC_files, sgACC_mean, save_path)
    conmpute_fc_vol_mean(ATN_LInsula_files, ATN_LInsula_mean, save_path)
    conmpute_fc_vol_mean(DN_LPCC_files, DN_LPCC_mean, save_path)
    conmpute_fc_vol_mean(DN_RaMPFC_files, DN_RaMPFC_mean, save_path)
    conmpute_fc_vol_mean(DN_vMPFC_files, DN_vMPFC_mean, save_path)
    conmpute_fc_vol_mean(Motor_LHand_files, Motor_LHand_mean, save_path)


def surf_fc_mean(data_path):
    hemis = ['lh', 'rh']
    save_path = data_path / 'sub-mean'
    save_path.mkdir(parents=True, exist_ok=True)
    for hemi in hemis:
        LH_Motor_files = sorted(data_path.glob(f'*/{hemi}_LH_Motor_fc.mgh'))
        RH_Motor_files = sorted(data_path.glob(f'*/{hemi}_RH_Motor_fc.mgh'))
        LH_ACC_files = sorted(data_path.glob(f'*/{hemi}_LH_ACC_fc.mgh'))
        RH_ACC_files = sorted(data_path.glob(f'*/{hemi}_RH_ACC_fc.mgh'))
        LH_PCC_files = sorted(data_path.glob(f'*/{hemi}_LH_PCC_fc.mgh'))
        RH_PCC_files = sorted(data_path.glob(f'*/{hemi}_RH_PCC_fc.mgh'))

        LH_Motor = []
        RH_Motor = []
        LH_ACC = []
        RH_ACC = []
        LH_PCC = []
        RH_PCC = []

        conmpute_fc_surf_mean(LH_Motor_files, LH_Motor, save_path)
        conmpute_fc_surf_mean(RH_Motor_files, RH_Motor, save_path)
        conmpute_fc_surf_mean(LH_ACC_files, LH_ACC, save_path)
        conmpute_fc_surf_mean(RH_ACC_files, RH_ACC, save_path)
        conmpute_fc_surf_mean(LH_PCC_files, LH_PCC, save_path)
        conmpute_fc_surf_mean(RH_PCC_files, RH_PCC, save_path)


if __name__ == '__main__':
    dataset = 'MSC'
    pipeline = 'DeepPrep-SDC'
    data_path = Path(f'/mnt/ngshare/DeepPrep/{dataset}/derivatives/analysis/{pipeline}')
    vol_fc_mean(data_path, pipeline)
    surf_fc_mean(data_path)
