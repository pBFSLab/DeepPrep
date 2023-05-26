import os
from pathlib import Path

import nibabel as nib
from interp_fine import resample_sphere_surface_barycentric


def link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None):
    orig_subjects_dir = Path(orig_subjects_dir)
    target_subjects_dir = Path(target_subjects_dir)

    for sub in orig_subjects_dir.iterdir():
        if 'ses' not in sub.name:
            continue
        if sub.is_dir():
            # Get the paths of the directories
            source_dir = sub
            dest_dir = target_subjects_dir / sub.name

            # Create a symbolic link from the source directory to the destination directory
            if not dest_dir.exists():
                dest_dir.symlink_to(source_dir)

    source_dir = Path('/usr/local/freesurfer720/subjects/fsaverage6')
    dest_dir = target_subjects_dir / 'fsaverage6'
    if not dest_dir.exists():
        dest_dir.symlink_to(source_dir)


def traversal_link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None):
    orig_subjects_dir = Path(orig_subjects_dir)
    target_subjects_dir = Path(target_subjects_dir)

    for sub_dir in orig_subjects_dir.iterdir():
        # if 'ses' in sub_dir.name:
        #     continue

        orig_dir = sub_dir
        target_dir = target_subjects_dir.joinpath(sub_dir.name)

        for root, dirs, files in os.walk(orig_dir):
            # Create a corresponding directory in the target directory
            target_root = os.path.join(target_dir, os.path.relpath(root, orig_dir))
            os.makedirs(target_root, exist_ok=True)

            # Create symbolic links for all files in the current directory
            for file in files:
                orig_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_root, file)
                if not os.path.exists(target_file_path):
                    os.symlink(orig_file_path, target_file_path)


def link_hnu_subjects_dir():

    # orig_subjects_dir = '/mnt/ngshare2/HNU_1_all/HNU_1_combine_Reconall720'
    # target_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/HNU_1_combine_FS720'
    # if not os.path.exists(target_subjects_dir):
    #     os.mkdir(target_subjects_dir)
    # link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None)
    #
    # orig_subjects_dir = '/mnt/ngshare2/DeepPrep_HNU_1/HNU_1_Recon'
    # target_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/HNU_1_combine_DP'
    # if not os.path.exists(target_subjects_dir):
    #     os.mkdir(target_subjects_dir)
    # link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None)
    #
    # orig_subjects_dir = '/run/user/1000/gvfs/sftp:host=30.30.30.141,user=anning/mnt/ngshare/public/share/ProjData/SurfRecon/TestReTest/CoRR_HNU_300/FreeSurfer'
    # target_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/HNU_1_combine_FS600'
    # if not os.path.exists(target_subjects_dir):
    #     os.mkdir(target_subjects_dir)
    # link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None)
    #
    # orig_subjects_dir = '/mnt/ngshare2/DeepPrep_HNU_1/HNU_1_Recon'
    # target_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/HNUcombine_DPcurv'
    # if not os.path.exists(target_subjects_dir):
    #     os.mkdir(target_subjects_dir)
    # traversal_link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None)

    orig_subjects_dir = '/mnt/ngshare2/DeepPrep_HNU_1/HNU_1_Recon_allT1'
    target_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/HNU_DPtrt'
    if not os.path.exists(target_subjects_dir):
        os.mkdir(target_subjects_dir)
    traversal_link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None)

    orig_subjects_dir = '/run/user/1000/gvfs/sftp:host=30.30.30.141,user=anning/mnt/ngshare/public/share/ProjData/SurfRecon/TestReTest/CoRR_HNU_300/FreeSurfer'
    target_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/HNU_FS600'
    if not os.path.exists(target_subjects_dir):
        os.mkdir(target_subjects_dir)
    link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None)


def link_UKB_file():

    orig_subjects_dir = '/run/user/1000/gvfs/sftp:host=30.30.30.17,user=youjia/mnt/ngshare/fMRIPrep_UKB_150/UKB_150_Recon'
    target_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/UKB_150_Recon_FS720'
    if not os.path.exists(target_subjects_dir):
        os.mkdir(target_subjects_dir)
    traversal_link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None)

    orig_subjects_dir = '/run/user/1000/gvfs/sftp:host=30.30.30.17,user=youjia/mnt/ngshare/DeepPrep_UKB_150/UKB_150_Recon'
    target_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/UKB_150_Recon_DP'
    if not os.path.exists(target_subjects_dir):
        os.mkdir(target_subjects_dir)
    traversal_link_subjects_dir(orig_subjects_dir, target_subjects_dir, subjects_list=None)


def replace_sphere_reg():
    import shutil
    orig_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/SurfReg/NonRigid/HNU_combine_DPtrt'
    target_subjects_dir = '/mnt/ngshare/DeepPrep/ttest/HNU_combine_DPtrt'
    orig_subjects_dir = Path(orig_subjects_dir)
    target_subjects_dir = Path(target_subjects_dir)

    hemi = 'lh'

    for sub_dir in orig_subjects_dir.iterdir():
        orig_sphere_reg_file = sub_dir / 'surf' / f'{hemi}.sphere.reg'
        if not orig_sphere_reg_file.exists():
            continue
        target_sphere_reg_file = target_subjects_dir / sub_dir.name / 'surf' / f'{hemi}.sphere.reg'
        if target_sphere_reg_file.exists():
            target_sphere_reg_file.unlink(missing_ok=True)

        shutil.copy(orig_sphere_reg_file, target_sphere_reg_file)


def interp_morph(orig_file, target_file, orig_geo_file, target_geo_file, device='cuda'):
    import torch
    # 加载数据
    sulc_orig = nib.freesurfer.read_morph_data(orig_file).astype(np.float32)

    xyz_orig, faces_orig = nib.freesurfer.read_geometry(orig_geo_file)
    xyz_orig = xyz_orig.astype(np.float32)

    xyz_target, faces_target = nib.freesurfer.read_geometry(target_geo_file)
    xyz_target = xyz_target.astype(np.float32)

    sulc_orig_t = torch.from_numpy(sulc_orig).to(device)
    xyz_orig_t = torch.from_numpy(xyz_orig).to(device)
    faces_orig_t = torch.from_numpy(faces_orig.astype(int)).to(device)
    xyz_target_t = torch.from_numpy(xyz_target).to(device)

    sulc_interp = resample_sphere_surface_barycentric(xyz_orig_t, xyz_target_t, sulc_orig_t.unsqueeze(1),
                                                      orig_face=faces_orig_t)

    nib.freesurfer.write_morph_data(target_file, sulc_interp.squeeze().cpu().numpy())

    print(f'interp: >>> {target_file}')


def resample():
    subjects_dir_list = ['HNU_1_combine_DP',
                              'HNU_1_combine_FS720',
                              'HNU_1_combine_FS600',
                              'HNU_1_combine_DPNew',
                              'HNU_1_combine_DPNew720',
                              'HNUcombine_DPNew_susu',
                              'HNUcombine_DPcurv',
                              'UKB_150_Recon_DPcurv',
                              'UKB_150_Recon_FS720',
                              ]

    subjects_dir_list = [
                         'HNU_combine_DPtrt',
                         'HNU_1_combine_FS720',
                         ]

    hemi = 'rh'

    for subjects_dir_name in subjects_dir_list:
        subjects_dir = f'/mnt/ngshare/DeepPrep/ttest/{subjects_dir_name}'
        resample_dir = f'/mnt/ngshare/DeepPrep/ttest/resample_fsaverage6/{subjects_dir_name}'

        # set_environ
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer720'
        os.environ['SUBJECTS_DIR'] = subjects_dir
        os.environ['PATH'] = '/usr/local/freesurfer710/bin:/usr/local/workbench/bin_linux64:' + os.environ['PATH']

        for subject_id in os.listdir(subjects_dir):
            if 'fsaverage' in subject_id:
                continue
            for ftype in ['sulc', 'curv', 'thickness']:
                sfile = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.{ftype}')
                tfile = os.path.join(resample_dir, subject_id, 'surf', f'{hemi}.{ftype}')
                result_dir = os.path.dirname(tfile)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                cmd = f'mri_surf2surf --srcsubject {subject_id} --sval {sfile} ' \
                      f'--trgsubject fsaverage6 --tval {tfile} --tfmt curv --hemi {hemi}'
                if not os.path.exists(tfile):
                    os.system(cmd)

                # s_geo_file = os.path.join(subjects_dir, subject_id, 'surf', f'lh.sphere.reg')
                # t_geo_file = '/usr/local/freesurfer720/subjects/fsaverage6/surf/lh.sphere'
                # interp_morph(sfile, tfile, s_geo_file, t_geo_file)


from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def ttest_pvalue():
    subjects_dir_name_a = 'HNU_combine_DPtrt'
    subjects_dir_name_b = 'HNU_1_combine_FS720'

    # subjects_dir_name_a = 'UKB_150_Recon_DPcurv'
    # subjects_dir_name_b = 'UKB_150_Recon_FS720'

    hemi = 'rh'

    subjects_dir_a = Path(f'/mnt/ngshare/DeepPrep/ttest/resample_fsaverage6/{subjects_dir_name_a}')
    subjects_dir_b = Path(f'/mnt/ngshare/DeepPrep/ttest/resample_fsaverage6/{subjects_dir_name_b}')

    output_dir = f'/mnt/ngshare/DeepPrep/ttest/stats/pvalue/{subjects_dir_name_a}_{subjects_dir_name_b}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def get_subjects_data(subjects_dir_path, file_type, subjects_list):
        datas = []
        for sub in subjects_list:
            sub = sub.strip()
            data_file_path = subjects_dir_path / sub / 'surf' / f'{hemi}.{file_type}'
            data = nib.freesurfer.read_morph_data(str(data_file_path))
            datas.append(data)
        return np.array(datas)

    def zscore(datas):
        d_mean = datas.mean(axis=1)[:, np.newaxis]
        d_std = datas.std(axis=1)[:, np.newaxis]
        d_zscore = (datas - d_mean) / d_std
        return d_zscore

    with open(os.path.join(f'/mnt/ngshare/SurfReg/Data_Extra/list', f'HNUcombine_all_list.txt'), "r") as f:
    # with open(os.path.join(f'/mnt/ngshare/SurfReg/Data_Extra/list', f'UKB_150_all_list.txt'), "r") as f:
        subjects = f.readlines()

    for ftype in ['sulc', 'curv', 'thickness']:
    # for ftype in ['thickness']:
        datas_a = get_subjects_data(subjects_dir_a, ftype, subjects)
        datas_b = get_subjects_data(subjects_dir_b, ftype, subjects)

        # datas_a = datas_a[:, mask]
        # datas_b = datas_b[:, mask]

        # ############## p-value
        p_values = []
        for i in range(40962):
            t, p = ttest_ind(datas_a[:, i], datas_b[:, i])
            p_values.append(p)

        p_values = np.array(p_values)
        p_values[np.isnan(p_values)] = 0

        # do something with the p_values here
        pvalue_file = os.path.join(output_dir, f'{hemi}.pvalue.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, p_values)
        print(f'>>> {pvalue_file}')

        pvalue_file = os.path.join(output_dir, f'{hemi}.log.pvalue.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, -np.log(p_values))
        print(f'>>> {pvalue_file}')

        # do FDR
        from statsmodels.stats.multitest import fdrcorrection
        rejected, corrected_pvals = fdrcorrection(p_values, alpha=0.05)
        pvalue_file = os.path.join(output_dir, f'{hemi}.FDR.pvalue.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, corrected_pvals)
        print(f'>>> {pvalue_file}')

        pvalue_file = os.path.join(output_dir, f'{hemi}.log.FDR.pvalue.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, -np.log(corrected_pvals))
        print(f'>>> {pvalue_file}')

        # ############## p-value
        datas_a_zscore = zscore(datas_a)
        datas_b_zscore = zscore(datas_b)

        p_values = []
        for i in range(40962):
            t, p = ttest_ind(datas_a_zscore[:, i], datas_b_zscore[:, i])
            p_values.append(p)
        p_values = np.array(p_values)

        pvalue_file = os.path.join(output_dir, f'{hemi}.pvalue.zscore.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, p_values)
        print(f'>>> {pvalue_file}')

        pvalue_file = os.path.join(output_dir, f'{hemi}.log.pvalue.zscore.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, -np.log(p_values))
        print(f'>>> {pvalue_file}')

        # do FDR
        from statsmodels.stats.multitest import fdrcorrection
        rejected, corrected_pvals = fdrcorrection(p_values, alpha=0.05)
        pvalue_file = os.path.join(output_dir, f'{hemi}.FDR.pvalue.zscore.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, corrected_pvals)
        print(f'>>> {pvalue_file}')

        pvalue_file = os.path.join(output_dir, f'{hemi}.log.FDR.pvalue.zscore.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, -np.log(corrected_pvals))
        print(f'>>> {pvalue_file}')

        # ############## relative bias
        # relative_bias = (datas_b - datas_a) / datas_a
        # print((datas_a == 0).sum(), (datas_b == 0).sum())
        # index = np.logical_or(datas_a == 0, datas_b == 0)
        # relative_bias[index] = np.nan
        # relative_bias_mean = np.nanmean(np.abs(relative_bias), axis=0) * 100
        # relative_file = os.path.join(output_dir, f'{hemi}.relative_file.{ftype}')
        #
        # sns.histplot(relative_bias[0] * 100, binrange=(0, 2000))
        # plt.show()
        # sns.scatterplot(datas_a.mean(axis=0), datas_b.mean(axis=0), )
        # plt.show()
        #
        # datas_a_mean = datas_a.mean(axis=0)
        # datas_b_mean = datas_b.mean(axis=0)
        # relative_bias_mean = (datas_b_mean - datas_a_mean) / datas_a_mean * 100
        # relative_bias_mean = relative_bias_mean
        #
        # # 剔除大于3倍std的异常点
        # relative_bias_std = relative_bias_mean.std()
        # z = np.abs(relative_bias_mean) / relative_bias_std
        # relative_bias_mean[z > 3] = 0
        #
        # sns.scatterplot(datas_a_mean, relative_bias_mean, )
        # plt.show()
        #
        # nib.freesurfer.write_morph_data(relative_file, relative_bias_mean)
        # print(f'>>> {relative_file}')
        def get_roi(pvalue_data, reversed=False):
            annot_data, _, _ = nib.freesurfer.read_annot(f'/usr/local/freesurfer720/subjects/fsaverage6/label/{hemi}.aparc.annot')
            if reversed:
                c1 = pvalue_data > 0.001
            else:
                c1 = pvalue_data < 0.001  # 找到显著的区域
            c2 = ~np.logical_or(annot_data == 0, annot_data == 4)  # 找到有效区域
            roi = np.logical_and(c1, c2)
            return roi

        def get_fixed(ft):
            dir_fixed = '/usr/local/freesurfer720/subjects'
            data_fixed = nib.freesurfer.read_morph_data(os.path.join(dir_fixed, 'fsaverage6', 'surf', f'{hemi}.{ft}')).astype(np.float32)
            d_m = data_fixed.mean()
            d_s = data_fixed.std()
            data_fixed = (data_fixed - d_m) / d_s
            return data_fixed

        def sim_mean(predict, target):
            corr_top = ((predict - predict.mean(axis=0, keepdims=True)) * (
                        target - target.mean(axis=0, keepdims=True))).mean(axis=0, keepdims=True)
            corr_bottom = (predict.std(axis=0, keepdims=True) * target.std(axis=0, keepdims=True))
            corr = corr_top / corr_bottom
            corr = corr.mean()
            l2 = np.mean((predict - target) ** 2)
            l1 = np.mean(np.abs(predict - target))

            return corr, l2, l1

        datas_fixed_zscore = get_fixed(ftype)[np.newaxis, :]

        roi = get_roi(p_values, reversed=False)
        print(roi.sum())
        print(ftype)
        corr_a, l2_a, l1_a = sim_mean(datas_a_zscore[:, roi].T, datas_fixed_zscore[:, roi].T)
        print(corr_a, l2_a, l1_a)
        corr_b, l2_b, l1_b = sim_mean(datas_b_zscore[:, roi].T, datas_fixed_zscore[:, roi].T)
        print(corr_b, l2_b, l1_b)

        print('diff', corr_a - corr_b, l2_a - l2_b, l1_a - l1_b)

        roi = get_roi(p_values, reversed=True)
        print(roi.sum())
        corr_a, l2_a, l1_a = sim_mean(datas_a_zscore[:, roi].T, datas_fixed_zscore[:, roi].T)
        print(corr_a, l2_a, l1_a)
        corr_b, l2_b, l1_b = sim_mean(datas_b_zscore[:, roi].T, datas_fixed_zscore[:, roi].T)
        print(corr_b, l2_b, l1_b)

        print('diff', corr_a - corr_b, l2_a - l2_b, l1_a - l1_b)

        def sim(predict, target):
            corr_top = ((predict - predict.mean(axis=0, keepdims=True)) * (
                    target - target.mean(axis=0, keepdims=True))).mean(axis=0, keepdims=True)
            corr_bottom = (predict.std(axis=0, keepdims=True) * target.std(axis=0, keepdims=True))
            corr = corr_top / corr_bottom
            corr = corr
            l2 = (predict - target) ** 2
            l1 = np.abs(predict - target)

            return corr, l2, l1

        corr_a, l2_a, l1_a = sim(datas_a_zscore.T, datas_fixed_zscore.T)
        corr_b, l2_b, l1_b = sim(datas_b_zscore.T, datas_fixed_zscore.T)

        pvalue_file = os.path.join(output_dir, f'{hemi}.{subjects_dir_name_a}.l2.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, l2_a.mean(axis=1))
        print(f'>>> {pvalue_file}')

        pvalue_file = os.path.join(output_dir, f'{hemi}.{subjects_dir_name_b}.l2.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, l2_b.mean(axis=1))
        print(f'>>> {pvalue_file}')

        corr_diff = corr_a - corr_b
        print('group diff')
        print(corr_diff.mean())
        l2_diff = l2_a - l2_b
        l1_diff = l1_a - l1_b

        l2_diff_mean = l2_diff.mean(axis=1)

        pvalue_file = os.path.join(output_dir, f'{hemi}.l2_diff.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, l2_diff_mean)
        print(f'>>> {pvalue_file}')

        pvalue_file = os.path.join(output_dir, f'{hemi}.l1_diff.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, l1_diff.mean(axis=1))
        print(f'>>> {pvalue_file}')

        l2_diff_mean[l2_diff_mean < 0] = -1
        l2_diff_mean[l2_diff_mean > 0] = 1

        print((l2_diff_mean > 0).sum())

        pvalue_file = os.path.join(output_dir, f'{hemi}.binary.l2_diff.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, l2_diff_mean)
        print(f'>>> {pvalue_file}')

        pvalue_file = os.path.join(output_dir, f'{hemi}.binary.l1_diff.{ftype}')
        nib.freesurfer.write_morph_data(pvalue_file, l1_diff.mean(axis=1))
        print(f'>>> {pvalue_file}')

        print()


if __name__ == '__main__':
    # link_hnu_subjects_dir()  # link recon 数据
    # link_hnu_file()  # link recon数据，并且修改其中的sphere.reg为SurfReg中的sphere.reg
    # replace_sphere_reg()
    # resample()  # 将native空间的数据
    ttest_pvalue()

    # link_UKB_file()
    # replace_sphere_reg()
    # resample()
    # ttest_pvalue()

    # link_hnu_subjects_dir()
