import os
import time
import cv2

import ants

from image import concat_horizontal, concat_vertical
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool
from statsmodels.stats.weightstats import ztest
import shutil
from glob import glob
import torch
from pytorch3d.ops.knn import knn_points
from interface.run import set_envrion


def info_label_aseg():
    aseg_label = [0., 2., 3., 4., 5., 7., 8., 10., 11., 12., 13.,
                  14., 15., 16., 17., 18., 24., 26., 28., 30., 31., 41.,
                  42., 43., 44., 46., 47., 49., 50., 51., 52., 53., 54.,
                  58., 60., 62., 63., 77., 85., 251., 252., 253., 254., 255.]
    aseg_label_dict = {0: 0,
                       2: "Left-Cerebral-White-Matter",
                       3: "Left-Cerebral-Cortex",
                       4: "Left-Lateral-Ventricle",
                       5: "Left-Inf-Lat-Vent",
                       7: "Left-Cerebellum-White-Matter",
                       8: "Left-Cerebellum-Cortex",
                       10: "Left-Thalamus",
                       11: "Left-Caudate",
                       12: "left-Putamen",
                       13: "lefy-Pallidum",
                       14: "3rd-Ventricle",
                       15: "4th-Ventricle",
                       16: "Brain-Stem",
                       17: "Left-Hippocampus",
                       18: "Left-Amygdala",
                       24: "CSF",
                       26: "Left-Accumbens-area",
                       28: "Left-VentralDC",
                       30: "Left-vessel",
                       31: "Left-choroid-plexus",
                       41: "Right-Cerebral-White-Matter",
                       42: "Right-Cerebral-Cortex",
                       43: "Right-Lateral-Ventricle",
                       44: "Right-Inf-Lat-Vent",
                       46: "Right-Cerebellum-White-Matter",
                       47: "Right-Cerebellum-Cortex",
                       49: "Right-Thalamus",
                       50: "Right-Caudate",
                       51: "Right-Putamen",
                       52: "Right-Pallidum",
                       53: "Right-Hippocampus",
                       54: "Right-Amygdala",
                       58: "Right-Accumbens-area",
                       60: "Right-VentralDC",
                       62: "Right-vessel",
                       63: "Right-choroid-plexus",
                       77: "WM-hypointensities",
                       85: "Optic-Chiasm",
                       251: "CC_Posterior",
                       252: "CC_Mid_Posterior",
                       253: "CC_Central",
                       254: "CC_Mid_Anterior",
                       255: "CC_Anterior"}
    return aseg_label, aseg_label_dict

def info_label_aparc(parc=18, type="func"):
    if parc == 18 and type=="func":
        index = np.squeeze(ants.image_read('aparc_template/lh.Clustering_18_fs6_new.mgh').numpy())
        aparc_label = set(index)
        aparc_label_dict = {}
        for i in aparc_label:
            if i != 0:
                aparc_label_dict[i] = f'Network_{int(i)}'
            else:
                aparc_label_dict[i] = f'Unknown'
        return aparc_label, aparc_label_dict
    elif parc == 92 and type=="func":
        index = np.squeeze(ants.image_read('aparc_template/lh.Clustering_46_fs6.mgh').numpy())
        aparc_label = set(index)
        aparc_label_dict = {}
        for i in aparc_label:
            if i != 0:
                aparc_label_dict[i] = f'Network_{int(i)}'
            else:
                aparc_label_dict[i] = f'Unknown'
        return aparc_label, aparc_label_dict
    elif parc == 152 and type=="func":
        index = np.squeeze(ants.image_read('aparc_template/lh.Clustering_76_fs6.mgh').numpy())
        aparc_label = set(index)
        aparc_label_dict = {}
        for i in aparc_label:
            if i != 0:
                aparc_label_dict[i] = f'Network_{int(i)}'
            else:
                aparc_label_dict[i] = f'Unknown'
        return aparc_label, aparc_label_dict
    elif parc == 213 and type=="func":
        index_l = np.squeeze(ants.image_read('aparc_template/lh.Clustering_108_fs6.mgh').numpy())
        index_r = np.squeeze(ants.image_read('aparc_template/rh.Clustering_108_fs6.mgh').numpy())
        aparc_label_l = set(index_l)
        aparc_label_r = set(index_r)
        aparc_label_dict_l = {}
        aparc_label_dict_r = {}
        for i in aparc_label_l:
            if i != 0:
                aparc_label_dict_l[i] = f'Network_{int(i)}'
            else:
                aparc_label_dict_l[i] = f'Unknown'
        for i in aparc_label_r:
            if i != 0:
                aparc_label_dict_r[i] = f'Network_{int(i)}'
            else:
                aparc_label_dict_r[i] = f'Unknown'

        return aparc_label_l, aparc_label_dict_l, aparc_label_r, aparc_label_dict_r
    elif type=="anat":
        index = nib.freesurfer.io.read_annot('aparc_template/lh.aparc.annot')
        aparc_label = np.unique(index[0])
        aparc_label_dict = {}
        for k, v in zip(aparc_label, index[2]):
            aparc_label_dict[k] = str(v).lstrip('b').strip("''")
        return aparc_label, aparc_label_dict





class AccAndStability:
    """
    Calculate dice acc and dice std (stability)

    ants_reg: register to mni152_1mm space
    aseg_acc: use aseg results of method1 as gt to calculate the acc (dice) of method2
    """

    def __init__(self, recon_dir, dataset, method):
        self.recon_dir = recon_dir
        self.dataset = dataset
        self.method = method
        self.output_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/{self.dataset}/v1_aparc')

    def dc(self, pred, gt):
        result = np.atleast_1d(pred.astype(bool))
        reference = np.atleast_1d(gt.astype(bool))

        intersection = np.count_nonzero(result & reference)

        size_i1 = np.count_nonzero(result)
        size_i2 = np.count_nonzero(reference)

        try:
            dc = 2. * intersection / float(size_i1 + size_i2)
        except ZeroDivisionError:
            dc = 0.0

        return dc

    def compute_aseg_dice(self, gt, pred):
        num_ref = np.sum(gt)
        num_pred = np.sum(pred)

        if num_ref == 0:
            if num_pred == 0:
                return 1
            else:
                return 0
        else:
            return self.dc(pred, gt)

    def evaluate_aseg(self, arr: np.ndarray, arr_gt: np.ndarray, aseg_i):
        mask_gt = (arr_gt == aseg_i).astype(int)
        mask_pred = (arr == aseg_i).astype(int)
        return self.compute_aseg_dice(mask_gt, mask_pred)
        del mask_gt, mask_pred

    def resample_sphere_surface_nearest(self, orig_xyz, target_xyz, orig_annot, gn=False):
        assert orig_xyz.shape[0] == orig_annot.shape[0]

        p1 = target_xyz.unsqueeze(0)
        p2 = orig_xyz.unsqueeze(0)
        result = knn_points(p1, p2, K=1)

        idx_num = result[1].squeeze()
        if gn:  # 是否需要计算梯度
            dist = result[0].squeeze()
            target_annot = orig_annot[idx_num]
            return target_annot, dist
        else:
            target_annot = orig_annot[idx_num]
            return target_annot

    def interp_annot_knn(self, sphere_orig_file, sphere_target_file, orig_annot_file,
                         interp_result_file, sphere_interp_file=None, device='cuda'):
        xyz_orig, faces_orig = nib.freesurfer.read_geometry(sphere_orig_file)
        xyz_orig = xyz_orig.astype(np.float32)

        xyz_target, faces_target = nib.freesurfer.read_geometry(sphere_target_file)
        xyz_target = xyz_target.astype(np.float32)

        data_orig = nib.freesurfer.read_annot(orig_annot_file)

        # data_orig[0][data_orig[0] == 35] = 0  # 剔除无效分区
        data_orig_t = torch.from_numpy(data_orig[0].astype(np.int32)).to(device)

        xyz_orig_t = torch.from_numpy(xyz_orig).to(device)
        xyz_target_t = torch.from_numpy(xyz_target).to(device)

        data_target = self.resample_sphere_surface_nearest(xyz_orig_t, xyz_target_t, data_orig_t)
        data_target = data_target.detach().cpu().numpy()
        nib.freesurfer.write_annot(interp_result_file, data_target, data_orig[1], data_orig[2], fill_ctab=True)
        print(f'save_interp_annot_file_path: >>> {interp_result_file}')

        if sphere_interp_file is not None and not os.path.exists(sphere_interp_file):
            shutil.copyfile(sphere_target_file, sphere_interp_file)
            print(f'copy sphere_target_file to sphere_interp_file: >>> {sphere_interp_file}')




    def ants_reg(self, type_of_transform='SyN'):
        """
        Registering both brain.mgz and aseg.mgz to atlas space (default: mni152_1mm)

        Arguments
        ---------
        type_of_transform : 'SyN' -- nonlinear registration

        Returns
        -------
        save registered t1 file T1_mni152.mgz, t1_warp.nii.gz, t1_affine.mat, and transformed aseg_mni152.mgz
        """

        moving_dir = self.recon_dir
        moved_dir = Path(self.output_dir, f'aseg_{self.method}reg_to_mni152')
        print(f"moved dir: {moved_dir}")

        fixed_dir = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
        dest_dir = Path(moved_dir)
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
        for file in os.listdir(moving_dir):
            reg_output_dir = Path(dest_dir, file)
            if not reg_output_dir.exists():
                reg_output_dir.mkdir(parents=True, exist_ok=True)

            t1_moving_dir = Path(moving_dir, file, 'mri/brain.mgz')
            aseg_moving_dir = Path(moving_dir, file, 'mri/aseg.mgz')
            fixed = ants.image_read(fixed_dir)

            # register T1
            # non-rigid registration
            if type_of_transform == 'SyN':
                t1_moving = ants.image_read(str(t1_moving_dir))
                t1_moved = ants.registration(fixed, t1_moving, type_of_transform=type_of_transform)
                t1_fwdtransform = t1_moved['fwdtransforms']
                warp_dir = Path(reg_output_dir, 't1_warp.nii.gz')
                affine_dir = Path(reg_output_dir, 't1_affine.mat')
                shutil.copy(t1_fwdtransform[0], str(warp_dir))
                shutil.copy(t1_fwdtransform[1], str(affine_dir))
                t1_moved = t1_moved['warpedmovout']
                t1_movd_dir = Path(reg_output_dir, 'T1_mni152.mgz')
                ants.image_write(t1_moved, str(t1_movd_dir))

                # register aseg
                aseg_moving = ants.image_read(str(aseg_moving_dir))
                aseg_moved = ants.apply_transforms(fixed, aseg_moving, [str(warp_dir), str(affine_dir)],
                                                   interpolator='multiLabel')
                aseg_movd_dir = Path(reg_output_dir, 'aseg_mni152.mgz')
                ants.image_write(aseg_moved, str(aseg_movd_dir))
                print(f'{file} done')

            # rigid registration
            elif type_of_transform == 'Rigid':
                t1_moving = ants.image_read(str(t1_moving_dir))
                t1_moved = ants.registration(fixed, t1_moving, type_of_transform=type_of_transform)
                t1_fwdtransform = t1_moved['fwdtransforms']
                warp_dir = Path(reg_output_dir, 't1_warp.nii.gz')
                affine_dir = Path(reg_output_dir, 't1_affine.mat')
                shutil.copy(t1_fwdtransform[0], str(affine_dir))
                t1_moved = t1_moved['warpedmovout']
                t1_movd_dir = Path(reg_output_dir, 'T1_mni152.mgz')
                ants.image_write(t1_moved, str(t1_movd_dir))

                # register aseg
                aseg_moving = ants.image_read(str(aseg_moving_dir))
                aseg_moved = ants.apply_transforms(fixed, aseg_moving, [str(affine_dir)],
                                                   interpolator='multiLabel')
                aseg_movd_dir = Path(reg_output_dir, 'aseg_mni152.mgz')
                ants.image_write(aseg_moved, str(aseg_movd_dir))
                print(f'{file} done')
            else:
                raise TypeError(
                    "type_of_transform should be either 'SyN' for non-linear registration, or 'Rigid' for linear registration")

    def aseg_stability(self, method):
        """
        calculate 44 aseg stability (std of dice) for each sub

        Arguments
        ---------
        method: aseg method
        aseg: True -- get 44 aseg labels
              False -- get 18 aparc labels

        Returns
        -------
        save stability (std of dice) for all subjects (1 csv file)
        """
        input_dir = Path(self.output_dir, f'aseg_{method}reg_to_mni152')
        output_dir = Path(self.output_dir, f'aseg_{method}reg_mni152_stability.csv')

        label, label_dict = info_label_aseg()
        method_dict = {}

        for sub in os.listdir(input_dir):
            if 'ses' in sub:
                sub_id = sub.split('-ses')[0] # MSC
                # sub_id = '-'.join(sub.split('-')[:2])  # HNU_1
                if sub_id not in method_dict:
                    method_dict[sub_id] = [sub]
                else:
                    method_dict[sub_id].append(sub)

        df_dice_mean = pd.DataFrame(columns=label_dict.values(), index=sorted(method_dict.keys()))
        df_dice_std = pd.DataFrame(columns=label_dict.values(), index=sorted(method_dict.keys()))

        # calculate dice of all pairs
        for sub in sorted(method_dict.keys()):
            print(sub)
            df_sub_dice = None
            dice_dict = {}
            i = 0
            while i < len(method_dict[sub]):
                aseg_i = method_dict[sub][i]
                if aseg_i != sub:
                    for j in range(i + 1, len(method_dict[sub])):
                        aseg_j = method_dict[sub][j]
                        if aseg_j != sub:
                            i_dir = Path(input_dir, aseg_i, 'aseg_mni152.mgz')
                            j_dir = Path(input_dir, aseg_j, 'aseg_mni152.mgz')
                            print(aseg_i, aseg_j)
                            dice_dict['sub'] = aseg_i + ' & ' + aseg_j
                            i_aseg = ants.image_read(str(i_dir)).numpy()
                            j_aseg = ants.image_read(str(j_dir)).numpy()

                            for l in label:
                                dice = self.evaluate_aseg(i_aseg, j_aseg, l)
                                dice_dict[label_dict[l]] = [dice]

                            df = pd.DataFrame.from_dict(dice_dict)

                            if df_sub_dice is None:
                                df_sub_dice = df
                            else:
                                df_sub_dice = pd.concat((df_sub_dice, df), axis=0)

                i += 1

            # df_sub_dice.to_csv()
            df_dice_mean.loc[sub] = df_sub_dice.mean(axis=0)
            df_dice_std.loc[sub] = df_sub_dice.std(axis=0)

        df_dice = pd.concat([df_dice_mean, df_dice_std], axis=0)
        df_dice.to_csv(output_dir)


    def aparc_stability(self, input_dir, parc, method='DeepPrep'):
        """
        calculate 18, 92 , 152 or 213 aparc stability (std of dice) for each sub

        Arguments
        ---------
        method: aparc method
        aparc: 18 -- get 18 aparc labels
              92 -- get 92 aparc labels
              152 -- get 152 aparc labels
              213 -- get 213 aparc labels

        Returns
        -------
        save stability (std of dice) for all subjects and each of them
        """
        output_dir = Path(self.output_dir, f'aparc{parc}_{method}_csv')
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        if parc == 213:
            label_lh, label_dict_lh, label_rh, label_dict_rh = info_label_aparc(parc, type="func")
        elif parc in [18, 92, 152]:
            label, label_dict = info_label_aparc(parc, type="func")
        sub_id = [sub for sub in sorted(os.listdir(input_dir))]
        dict = {}
        for sub in sub_id:
            sub_dir = Path(input_dir, sub)
            dict[sub] = sorted(os.listdir(sub_dir))

        for hemi in ['lh', 'rh']:
            if parc == 213 and hemi == 'lh':
                label = label_lh
                label_dict = label_dict_lh
            elif parc == 213 and hemi == 'rh':
                label = label_rh
                label_dict = label_dict_rh
            else:
                pass
            df_dice_mean = pd.DataFrame(columns=label_dict.values(), index=sorted(dict.keys()))
            df_dice_std = pd.DataFrame(columns=label_dict.values(), index=sorted(dict.keys()))

            for sub in sorted(dict.keys()):
                print(sub)
                dice_dict = {}
                df_sub_dice = None

                i = 0
                while i < len(dict[sub]):
                    aparc_i = dict[sub][i]
                    for j in range(i + 1, len(dict[sub])):
                        aparc_j = dict[sub][j]
                        if parc == 18:
                            i_dir = \
                                glob(os.path.join(input_dir, sub, aparc_i,
                                                  f'parc/{aparc_i}/*/{hemi}_parc_result.annot'))[0] # App dir
                            j_dir = \
                                glob(os.path.join(input_dir, sub, aparc_j,
                                                  f'parc/{aparc_j}/*/{hemi}_parc_result.annot'))[0] # App dir
                        elif parc == 92:
                            i_dir = \
                            glob(os.path.join(input_dir, sub, aparc_i, f'parc92/{hemi}_parc92_result.annot'))[0] # App dir
                            j_dir = \
                            glob(os.path.join(input_dir, sub, aparc_j, f'parc92/{hemi}_parc92_result.annot'))[0] # App dir
                        else:
                            pass
                        print(aparc_i, aparc_j)
                        dice_dict['sub'] = aparc_i + ' & ' + aparc_j
                        i_aseg = nib.freesurfer.read_annot(i_dir)[0]
                        j_aseg = nib.freesurfer.read_annot(j_dir)[0]

                        for l in label:
                            dice = self.evaluate_aseg(i_aseg, j_aseg, l)
                            dice_dict[label_dict[l]] = [dice]

                        df = pd.DataFrame.from_dict(dice_dict)

                        if df_sub_dice is None:
                            df_sub_dice = df
                        else:
                            df_sub_dice = pd.concat((df_sub_dice, df), axis=0)

                    i += 1
                sub_output = Path(output_dir, f'{method}_{hemi}_{sub}_aparc{parc}_dice.csv')
                df_sub_dice.loc['mean'], df_sub_dice.loc['std'] = df_sub_dice.mean(axis=0), df_sub_dice.std(axis=0)
                df_sub_dice.to_csv(sub_output, index=False)
                df_dice_mean.loc[sub] = df_sub_dice.loc['mean']
                df_dice_std.loc[sub] = df_sub_dice.loc['std']

            stability_output_dir = Path(output_dir, f'{method}_{hemi}_aparc{parc}_stability.csv')
            df_dice = pd.concat([df_dice_mean, df_dice_std], axis=0)
            df_dice.to_csv(stability_output_dir)



class ScreenShot:
    """
    Take screenshots of input pipeline and dataset according to different methods and features (thickness, sulc, curv).
    """

    def __init__(self, recon_dir1, recon_dir2, dataset, method1="DeepPrep", method2=None):
        """
        Initialize the ScreenShot

        Arguments
        ---------
        pipeline : string
            the pipeline used

        dataset : string
            the datasett used
        """
        self.recon_dir1 = recon_dir1 # method 1
        self.recon_dir2 = recon_dir2 # method 2
        self.dataset = dataset
        self.method1 = method1
        self.method2 = method2
        self.feature_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/{self.dataset}/v1_feature')
        self.Multi_CPU_Num = 10

    def run_cmd(self, cmd):
        print(f'shell_run : {cmd}')
        os.system(cmd)
        print('*' * 40)

    def image_screenshot(self, surf_file, overlay_file, save_path, min, max, overlay_color='colorwheel'):
        cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color={overlay_color},inverse -cam dolly 1.4 azimuth 0 -ss {save_path}'
        print(cmd)
        os.system(cmd)

    def image_screenshot_azimuth_180(self, surf_file, overlay_file, save_path, min, max, overlay_color='colorwheel'):
        cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color={overlay_color},inverse -cam dolly 1.4 azimuth 180 -ss {save_path}'
        print(cmd)
        os.system(cmd)

    def pvalue_image_screenshot(self, surf_file, overlay_file, save_path, min, max, overlay_color='heat'):
        cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color={overlay_color}, -cam dolly 1.4 azimuth 0 -ss {save_path}'
        print(cmd)
        os.system(cmd)

    def pvalue_image_screenshot_azimuth_180(self, surf_file, overlay_file, save_path, min, max, overlay_color='heat'):
        cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color={overlay_color}, -cam dolly 1.4 azimuth 180 -ss {save_path}'
        print(cmd)
        os.system(cmd)

    def feature_screenshot(self, feature='thickness', vmin='', vmax=''):
        """
        读取FreeSurfer格式的目录，并对aparc结构进行截图
        需要 surf/?h.pial 和 label/?h.aparc.annot
        """
        for method, recon_dir in zip([self.method1, self.method2], [self.recon_dir1, self.recon_dir2]):
            if method is None:
                continue
            out_dir = Path(self.feature_dir, f'{feature}/feature_map_image_{method}')

            # HNU_1
            subject_list = [sub for sub in os.listdir(recon_dir) if sub.startswith('sub-')]

            args_list1 = []
            args_list2 = []
            for subject in subject_list:
                for hemi in ['lh', 'rh']:
                    surf_file = os.path.join(recon_dir, subject, 'surf', f'{hemi}.pial')

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    overlay_file = os.path.join(recon_dir, subject, 'surf', f'{hemi}.{feature}')
                    save_file = os.path.join(out_dir, f'{subject}_{hemi}_lateral.png')
                    args_list1.append([surf_file, overlay_file, save_file, vmin, vmax])
                    save_file = os.path.join(out_dir, f'{subject}_{hemi}_medial.png')
                    args_list2.append([surf_file, overlay_file, save_file, vmin, vmax])
            pool = Pool(self.Multi_CPU_Num)
            pool.starmap(self.image_screenshot, args_list1)
            pool.starmap(self.image_screenshot_azimuth_180, args_list2)
            pool.close()
            pool.join()

    def project_fsaverage6(self, recon_dir, feature='thickness', hemi='lh'):
        """
        project subjects to fsaverage6 space
        """
        method = self.method1
        recon_dir = Path(recon_dir)
        output_dir = Path(self.feature_dir, f'recon_interp_fsaverage6/{method}')

        target = recon_dir / 'fsaverage6'
        if not target.exists():
            target.symlink_to('/usr/local/freesurfer/subjects/fsaverage6', target_is_directory=True)

        args_list = []
        for subject in recon_dir.iterdir():
            subject_id = subject.name
            if not 'sub' in subject_id:
                continue
            if 'fsaverage' in subject_id:
                continue

            os.environ['SUBJECTS_DIR'] = str(recon_dir)

            sfile = recon_dir / subject_id / 'surf' / f'{hemi}.{feature}'

            interp_path = output_dir / subject_id / 'surf'
            interp_path.mkdir(parents=True, exist_ok=True)
            tfile = interp_path / f'{hemi}.{feature}'

            if not tfile.exists():
                cmd = f'mri_surf2surf --srcsubject {subject_id} --sval {sfile} ' \
                      f'--trgsubject fsaverage6 --tval {tfile} --tfmt curv --hemi {hemi}'
                args_list.append((cmd,))
        pool = Pool(self.Multi_CPU_Num)
        pool.starmap(self.run_cmd, args_list)
        pool.close()
        pool.join()

    def cal_individual_fsaverage6(self, feature='thickness', hemi='lh'):
        """
        calculate individual's mean and std according to different features
        """
        method = self.method1
        interp_dir = Path(self.feature_dir, f'recon_interp_fsaverage6/{method}')
        individual_dir = Path(self.feature_dir, 'recon_individual_fsaverage6')

        dp_dict = dict()
        for subject_path in interp_dir.iterdir():
            if not 'sub' in subject_path.name:
                continue
            # if 'ses' not in subject_path.name:
            #     continue

            sub_name = subject_path.name.split('-ses')[0]
            # sub_name = '-'.join(subject_path.name.split('-')[:2])

            if sub_name not in dp_dict:
                dp_dict[sub_name] = [subject_path]
            else:
                dp_dict[sub_name].append(subject_path)


        project = self.method1
        proj_dict = dp_dict

        for sub_name in proj_dict.keys():

            subjects = proj_dict[sub_name]

            data_concat = None
            for subject_path in subjects:
                feature_file = subject_path / 'surf' / f'{hemi}.{feature}'
                data = np.expand_dims(nib.freesurfer.read_morph_data(str(feature_file)), 1)
                if data_concat is None:
                    data_concat = data
                else:
                    data_concat = np.concatenate([data_concat, data], axis=1)

            data_mean = np.nanmean(data_concat, axis=1)
            data_std = np.nanstd(data_concat, axis=1)

            out_dir = individual_dir / project / sub_name / 'surf'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file_mean = out_dir / f'{hemi}.mean.{feature}'
            out_file_std = out_dir / f'{hemi}.std.{feature}'
            nib.freesurfer.write_morph_data(out_file_mean, data_mean)
            nib.freesurfer.write_morph_data(out_file_std, data_std)

    def cal_stability_fsaverage6(self, feature='thickness', hemi='lh'):
        """
        input: surf/?h.std.<feature>
        output: surf/?h.<feature>
        """
        # method = self.method1
        group_dir = Path(self.feature_dir, f'recon_individual_fsaverage6')
        stability_dir = Path(self.feature_dir, 'recon_stability_fsaverage6')

        project = self.method1

        subjects = []
        for subject_path in (group_dir / project).iterdir():
            if not 'sub' in subject_path.name:
                continue
            subjects.append(subject_path)

        data_concat = None
        for subject_path in subjects:
            feature_file = subject_path / 'surf' / f'{hemi}.std.{feature}'
            data = np.expand_dims(nib.freesurfer.read_morph_data(str(feature_file)), 1)
            if data_concat is None:
                data_concat = data
            else:
                data_concat = np.concatenate([data_concat, data], axis=1)
        data_mean = np.nanmean(data_concat, axis=1)

        out_dir = stability_dir / project / 'surf'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file_mean = out_dir / f'{hemi}.{feature}'
        nib.freesurfer.write_morph_data(out_file_mean, data_mean)

    def cal_group_fsaverage6(self, feature='thickness', hemi='lh'):
        """
        calculate group mean and std according to different features

        input: surf/?h.<feature>
        output: surf/?h.mean.<feature>
                surf/?h.std.<feature>
        """
        method = self.method1
        interp_dir = Path(self.feature_dir, f'recon_interp_fsaverage6/{method}')
        group_dir = Path(self.feature_dir, 'recon_group_fsaverage6')

        dp_list = list()
        for subject_path in interp_dir.iterdir():
            if not 'sub' in subject_path.name:
                continue
            # if 'ses' not in subject_path.name:
            #     continue

            dp_list.append(subject_path)

        project = self.method1
        proj_list = dp_list

        data_concat = None
        for subject_path in proj_list:
            feature_file = subject_path / 'surf' / f'{hemi}.{feature}'
            data = np.expand_dims(nib.freesurfer.read_morph_data(str(feature_file)), 1)
            if data_concat is None:
                data_concat = data
            else:
                data_concat = np.concatenate([data_concat, data], axis=1)

        data_mean = np.nanmean(data_concat, axis=1)
        data_std = np.nanstd(data_concat, axis=1)

        out_dir = group_dir / project / 'surf'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file_mean = out_dir / f'{hemi}.mean.{feature}'
        out_file_std = out_dir / f'{hemi}.std.{feature}'
        nib.freesurfer.write_morph_data(out_file_mean, data_mean)
        nib.freesurfer.write_morph_data(out_file_std, data_std)

    def concat_group_screenshot(self, screenshot_dir: Path, out_dir: Path, feature=''):
        """
        拼接method1和method2的分区结果图像
        """
        for stats_type in ['mean', 'std']:
            f1 = screenshot_dir / f'{self.method1}_{feature}_{stats_type}_lh_lateral.png'
            f2 = screenshot_dir / f'{self.method1}_{feature}_{stats_type}_rh_lateral.png'
            f3 = screenshot_dir / f'{self.method1}_{feature}_{stats_type}_lh_medial.png'
            f4 = screenshot_dir / f'{self.method1}_{feature}_{stats_type}_rh_medial.png'
            img_h1 = concat_vertical([f1, f3, f4, f2])

            save_file = out_dir / f'{feature}_{stats_type}.png'
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(save_file), img_h1)
            print(f'>>> save image : {save_file}')

    def group_screenshot(self, feature='thickness', vmin1='', vmax1='', vmin2='', vmax2=''):
        """

        """
        method = self.method1

        input_dir = Path(self.feature_dir, f'recon_group_fsaverage6/{method}')
        out_dir = Path(self.feature_dir, f'recon_group_fsaverage6_screenshot/{method}')
        concat_dir = Path(self.feature_dir, f'recon_group_fsaverage6_screenshot_concat/{method}')

        args_list1 = []
        args_list2 = []
        for hemi in ['lh', 'rh']:
            surf_file = os.path.join('/usr/local/freesurfer/subjects/fsaverage6', 'surf', f'{hemi}.pial')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for stats_type, (vmin, vmax) in zip(['mean', 'std'], [(vmin1, vmax1), (vmin2, vmax2)]):
                overlay_file = Path(input_dir, 'surf', f'{hemi}.{stats_type}.{feature}')
                save_file = Path(out_dir, f'{method}_{feature}_{stats_type}_{hemi}_lateral.png')
                args_list1.append([surf_file, overlay_file, save_file, vmin, vmax])
                save_file = Path(out_dir, f'{method}_{feature}_{stats_type}_{hemi}_medial.png')
                args_list2.append([surf_file, overlay_file, save_file, vmin, vmax])
        pool = Pool(self.Multi_CPU_Num)
        pool.starmap(self.image_screenshot, args_list1)
        pool.starmap(self.image_screenshot_azimuth_180, args_list2)
        pool.close()
        pool.join()

        self.concat_group_screenshot(out_dir, concat_dir, feature=feature)

    def concat_stability_screenshot(self, screenshot_dir: Path, out_dir: Path, feature=''):
        """
        拼接DeepPrep和FreeSurfer的分区结果图像
        """
        f1 = screenshot_dir / f'{self.method1}_{feature}_lh_lateral.png'
        f2 = screenshot_dir / f'{self.method1}_{feature}_rh_lateral.png'
        f3 = screenshot_dir / f'{self.method1}_{feature}_lh_medial.png'
        f4 = screenshot_dir / f'{self.method1}_{feature}_rh_medial.png'

        img_h1 = concat_vertical([f1, f3, f4, f2])

        save_file = out_dir / f'{feature}.png'
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_file), img_h1)
        print(f'>>> save image : {save_file}')

    def stability_screenshot(self, feature='thickness', vmin='', vmax=''):
        """
        """
        method = self.method1
        input_dir = Path(self.feature_dir, 'recon_stability_fsaverage6')
        out_dir = Path(self.feature_dir, 'recon_stability_fsaverage6_screenshot')
        concat_dir = Path(self.feature_dir, f'recon_stability_fsaverage6_screenshot_concat/{method}')

        args_list1 = []
        args_list2 = []

        for hemi in ['lh', 'rh']:
            surf_file = os.path.join('/usr/local/freesurfer/subjects/fsaverage6', 'surf', f'{hemi}.pial')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            overlay_file = Path(input_dir, method, 'surf', f'{hemi}.{feature}')
            save_file = Path(out_dir, f'{method}_{feature}_{hemi}_lateral.png')
            args_list1.append([surf_file, overlay_file, save_file, vmin, vmax])
            save_file = os.path.join(out_dir, f'{method}_{feature}_{hemi}_medial.png')
            args_list2.append([surf_file, overlay_file, save_file, vmin, vmax])
        pool = Pool(self.Multi_CPU_Num)
        pool.starmap(self.image_screenshot, args_list1)
        pool.starmap(self.image_screenshot_azimuth_180, args_list2)
        pool.close()
        pool.join()

        self.concat_stability_screenshot(out_dir, concat_dir, feature=feature)



if __name__ == '__main__':
    set_envrion()

    ######################## Acc & Stability ##########################
    recon_dir = '/mnt/ngshare/DeepPrep_HNU_1/HNU_1_Recon_allT1'
    cls = AccAndStability(recon_dir, 'HNU_1', 'DeepPrep')
    # cls.ants_reg('Rigid') # register to MNI152 space
    # cls.aseg_stability('DeepPrep')
    # cls.aparc_stability()

    ####################### project to fs6 ##########################
    method1 = 'FreeSurfer'
    # method1 = 'DeepPrep'
    method2 = None
    recon_dir1 = '/mnt/ngshare/FreeSurfer_HNU_1/HNU_1_Recon_allT1'
    # recon_dir1 = '/mnt/ngshare/DeepPrep_HNU_1/HNU_1_Recon_allT1'
    recon_dir2 = None
    dataset = 'HNU_1'
    screenshot = ScreenShot(recon_dir1, recon_dir2, dataset, method1, method2)
    for feature, (vmin, vmax), (vmin2, vmax2), (vmin3, vmax3) in zip(['thickness', 'curv', 'sulc'],
                                                                     [('1', '3.5'), ('-0.5', '0.25'), ('-13', '13'), ],
                                                                     [('0', '0.8'), ('0', '0.15'), ('0', '4'), ],
                                                                     [('0', '0.6'), ('0', '0.08'), ('0', '1.3')], ):
        print(feature, vmin, vmax, vmin2, vmax2, vmin3, vmax3)
        # screenshot.feature_screenshot(feature)

        # for hemi in ['lh', 'rh']:
            # screenshot.project_fsaverage6(recon_dir1, feature=feature, hemi=hemi)
            # screenshot.cal_individual_fsaverage6(feature=feature, hemi=hemi)
            # screenshot.cal_stability_fsaverage6(feature=feature, hemi=hemi)
            # screenshot.cal_group_fsaverage6(feature=feature, hemi=hemi)

        screenshot.group_screenshot(feature=feature, vmin1=vmin, vmax1=vmax, vmin2=vmin2, vmax2=vmax2)
        screenshot.stability_screenshot(feature=feature, vmin=vmin3, vmax=vmax3)

    # ####################### project to fs6 ##########################
    # method1 = 'fMRIPrep'
    # method2 = None
    # recon_dir1 = '/mnt/ngshare/fMRIPrep_UKB_50/UKB_50_Recon'
    # recon_dir2 = None
    # dataset = 'UKB_50'
    # screenshot = ScreenShot(recon_dir1, recon_dir2, dataset, method1, method2)
    # for feature, (vmin, vmax), (vmin2, vmax2), (vmin3, vmax3) in zip(['thickness', 'curv', 'sulc'],
    #                                                                  [('1', '3.5'), ('-0.5', '0.25'), ('-13', '13'), ],
    #                                                                  [('0', '0.8'), ('0', '0.15'), ('0', '4'), ],
    #                                                                  [('0', '0.6'), ('0', '0.08'), ('0', '1.3')], ):
    #     print(feature, vmin, vmax, vmin2, vmax2, vmin3, vmax3)
    #     # screenshot.feature_screenshot(feature)
    #
    #     # for hemi in ['lh', 'rh']:
    #         # screenshot.project_fsaverage6(recon_dir1, feature=feature, hemi=hemi)
    #         # screenshot.cal_group_fsaverage6(feature=feature, hemi=hemi)
    #
    #     # screenshot.group_screenshot(feature=feature, vmin1=vmin, vmax1=vmax, vmin2=vmin2, vmax2=vmax2)

    # ######################## project to fs6 ##########################
    # method1 = 'DeepPrep'
    # method2 = None
    # recon_dir1 = '/mnt/ngshare/DeepPrep_UKB_50/UKB_50_Recon'
    # recon_dir2 = None
    # dataset = 'UKB_50'
    # screenshot = ScreenShot(recon_dir1, recon_dir2, dataset, method1, method2)
    # for feature, (vmin, vmax), (vmin2, vmax2), (vmin3, vmax3) in zip(['thickness', 'curv', 'sulc'],
    #                                                                  [('1', '3.5'), ('-0.5', '0.25'),
    #                                                                   ('-13', '13'), ],
    #                                                                  [('0', '0.8'), ('0', '0.15'), ('0', '4'), ],
    #                                                                  [('0', '0.6'), ('0', '0.08'), ('0', '1.3')], ):
    #     print(feature, vmin, vmax, vmin2, vmax2, vmin3, vmax3)
    #     screenshot.feature_screenshot(feature)
    #
    #     for hemi in ['lh', 'rh']:
    #         screenshot.project_fsaverage6(recon_dir1, feature=feature, hemi=hemi)
    #         screenshot.cal_group_fsaverage6(feature=feature, hemi=hemi)
    #
    #     screenshot.group_screenshot(feature=feature, vmin1=vmin, vmax1=vmax, vmin2=vmin2, vmax2=vmax2)
