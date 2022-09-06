import os
import time

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

def set_environ():
    # FreeSurfer
    value = os.environ.get('FREESURFER_HOME')
    if value is None:
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
        os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
        os.environ['PATH'] = '/usr/local/freesurfer/bin:' + os.environ['PATH']

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


def info_label_aparc(parc):
    if parc == 18:
        index = np.squeeze(ants.image_read('aparc_template/lh.Clustering_18_fs6_new.mgh').numpy())
        aparc_label = set(index)
        aparc_label_dict = {}
        for i in aparc_label:
            if i != 0:
                aparc_label_dict[i] = f'Network_{int(i)}'
            else:
                aparc_label_dict[i] = f'Unknown'
        return aparc_label, aparc_label_dict
    elif parc == 92:
        index = np.squeeze(ants.image_read('aparc_template/lh.Clustering_46_fs6.mgh').numpy())
        aparc_label = set(index)
        aparc_label_dict = {}
        for i in aparc_label:
            if i != 0:
                aparc_label_dict[i] = f'Network_{int(i)}'
            else:
                aparc_label_dict[i] = f'Unknown'
        return aparc_label, aparc_label_dict
    else:
        raise RuntimeError("parc = 18 or 92")

class AccAndStability:
    """
    Calculate dice acc and dice std (stability)

    ants_reg: register to mni152_1mm space
    aseg_acc: use aseg results of method1 as gt to calculate the acc (dice) of method2
    """
    def __init__(self, dataset, method):
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
        moving_dir = Path(f'/mnt/ngshare/DeepPrep/{self.dataset}/derivatives/{self.method}/Recon')
        moved_dir = Path(self.output_dir, f'aseg_{self.method}reg_to_mni152')

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
                raise TypeError("type_of_transform should be either 'SyN' for non-linear registration, or 'Rigid' for linear registration")

    def aseg_acc(self, method1, method2):
        """
        calculate 44 aseg acc (dice) from method2, while using aseg results from method1 as ground truth

        Arguments
        ---------
        method1: aseg results from method1 is treated as ground truth for the following calculation
        method2: calculate the aseg acc (dice)

        Returns
        -------
        save acc (dice) score
        """
        method1_dir = Path(self.output_dir, f'aseg_{method1}reg_to_mni152')
        method2_dir = Path(self.output_dir, f'aseg_{method2}reg_to_mni152')
        output_csv = Path(self.output_dir, f'aseg_{method1}reg_{method2}reg_mni152_dc.csv')

        label, label_dict = info_label_aseg()
        df_dice = None

        for sub in os.listdir(method1_dir):
            dice_dict = {}
            dice_dict['sub'] = sub
            method1_aseg_dir = Path(method1_dir, sub, 'aseg_mni152.mgz')
            method2_aseg_dir = Path(method2_dir, sub, 'aseg_mni152.mgz')
            method1_aseg = ants.image_read(str(method1_aseg_dir)).numpy()
            if not method2_aseg_dir.exists():
                print(f"{sub} found in {method1_dir}, NOT in {method2_dir}")
                continue
            method2_aseg = ants.image_read(str(method2_aseg_dir)).numpy()

            for i in label:
                dice = self.evaluate_aseg(method1_aseg, method2_aseg, i)
                dice_dict[label_dict[i]] = [dice]

            df = pd.DataFrame.from_dict(dice_dict)

            if df_dice is None:
                df_dice = df
            else:
                df_dice = pd.concat((df_dice, df), axis=0)
            print(f'{sub} done')

        df_dice.loc['mean'], df_dice.loc['std'] = df_dice.mean(axis=0), df_dice.std(axis=0)
        df_dice.to_csv(output_csv, index=False)

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
            sub_id = sub.split('_')[0]
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
        calculate 18 or 92 aparc stability (std of dice) for each sub

        Arguments
        ---------
        method: aparc method
        aparc: 18 -- get 18 aparc labels
              92 -- get 92 aparc labels

        Returns
        -------
        save stability (std of dice) for all subjects and each of them
        """
        output_dir = Path(self.output_dir, f'aparc{parc}_{method}_csv')
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        label, label_dict = info_label_aparc(parc)
        sub_id = [sub for sub in sorted(os.listdir(input_dir))]
        dict = {}
        for sub in sub_id:
            sub_dir = Path(input_dir, sub)
            dict[sub] = sorted(os.listdir(sub_dir))

        for hemi in ['lh', 'rh']:

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
                        if parc != 92:
                            i_dir = \
                            glob(os.path.join(input_dir, sub, aparc_i, f'parc/{aparc_i}/*/{hemi}_parc_result.annot'))[0]
                            j_dir = \
                            glob(os.path.join(input_dir, sub, aparc_j, f'parc/{aparc_j}/*/{hemi}_parc_result.annot'))[0]
                        elif parc == 92:
                            i_dir = glob(os.path.join(input_dir, sub, aparc_i, f'parc92/{hemi}_parc92_result.annot'))[0]
                            j_dir = glob(os.path.join(input_dir, sub, aparc_j, f'parc92/{hemi}_parc92_result.annot'))[0]
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
    def __init__(self, pipeline, dataset, method1="DeepPrep", method2='FreeSurfer'):
        """
        Initialize the ScreenShot

        Arguments
        ---------
        pipeline : string
            the pipeline used

        dataset : string
            the datasett used
        """
        self.pipeline = pipeline
        self.dataset = dataset
        self.method1 = method1
        self.method2 = method2
        self.derivative_dir = Path(f'/mnt/ngshare/Data_Mirror/{self.pipeline}/{self.dataset}/derivatives')
        self.feature_dir = Path(f'/mnt/ngshare/Data_Mirror/{self.pipeline}/Validation/{self.dataset}/v1_feature')
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
        for method in [self.method1, self.method2]:
            recon_dir = Path(self.derivative_dir, f'{method}/Recon')
            out_dir = Path(self.feature_dir, f'{feature}/feature_map_image_{method}')

            subject_list = os.listdir(recon_dir)
            args_list1 = []
            args_list2 = []
            for subject in subject_list:
                if not 'sub' in subject:
                    continue
                # if 'ses' in subject:
                #     continue

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

    def concat_screenshot(self, feature='thickness'):
        """
        拼接method1和method2的分区结果图像
        """
        screenshot_dir = Path(self.feature_dir, f'{feature}')

        filenames = os.listdir(os.path.join(screenshot_dir, f'feature_map_image_{self.method1}'))
        subjects = set(['_'.join(i.split('_')[0:-2]) for i in filenames])

        for subject in subjects:
            f1 = os.path.join(screenshot_dir, f'feature_map_image_{self.method1}', f'{subject}_lh_lateral.png')
            f2 = os.path.join(screenshot_dir, f'feature_map_image_{self.method1}', f'{subject}_rh_lateral.png')
            f3 = os.path.join(screenshot_dir, f'feature_map_image_{self.method1}', f'{subject}_lh_medial.png')
            f4 = os.path.join(screenshot_dir, f'feature_map_image_{self.method1}', f'{subject}_rh_medial.png')
            f5 = os.path.join(screenshot_dir, f'feature_map_image_{self.method2}', f'{subject}_lh_lateral.png')
            f6 = os.path.join(screenshot_dir, f'feature_map_image_{self.method2}', f'{subject}_rh_lateral.png')
            f7 = os.path.join(screenshot_dir, f'feature_map_image_{self.method2}', f'{subject}_lh_medial.png')
            f8 = os.path.join(screenshot_dir, f'feature_map_image_{self.method2}', f'{subject}_rh_medial.png')

            # img_h1 = concat_vertical([f1, f2, f5, f6])
            # img_h2 = concat_vertical([f3, f4, f7, f8])
            img_h1 = concat_vertical([f1, f3, f4, f2])
            img_h2 = concat_vertical([f5, f7, f8, f6])
            save_path = os.path.join(screenshot_dir, 'feature_map_image_concat')
            save_file = os.path.join(save_path, f'{subject}.png')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            img = concat_horizontal([img_h1, img_h2], save_file)

    def ln_subject(self):
        """
        将method1的Recon结果和method2的Recon结果link到一个目录下（mris_surf2surf需要）
        """
        method1_dir = Path(self.derivative_dir, f'{self.method1}/Recon')
        method2_dir = Path(self.derivative_dir, f'{self.method2}/Recon')
        concat_dir = Path(self.feature_dir, f'recon_dir_concat_{self.method1}_and_{self.method2}')

        concat_dir.mkdir(parents=True, exist_ok=True)
        for subject_path in method1_dir.iterdir():
            target = concat_dir / f'DeepPrep__{subject_path.name}'
            if not target.exists():
                print(subject_path)
                target.symlink_to(subject_path, target_is_directory=True)

        for subject_path in method2_dir.iterdir():
            target = concat_dir / f'FreeSurfer__{subject_path.name}'
            if not target.exists():
                print(subject_path)
                target.symlink_to(subject_path, target_is_directory=True)

    def project_fsaverage6(self, feature='thickness', hemi='lh'):
        """
        project subjects to fsaverage6 space
        """
        recon_dir = Path(self.feature_dir, f'recon_dir_concat_{self.method1}_and_{self.method2}')
        output_dir = Path(self.feature_dir, 'recon_interp_fsaverage6')

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
        interp_dir = Path(self.feature_dir, 'recon_interp_fsaverage6')
        individual_dir = Path(self.feature_dir, 'recon_individual_fsaverage6')

        dp_dict = dict()
        fs_dict = dict()
        for subject_path in interp_dir.iterdir():
            if not 'sub' in subject_path.name:
                continue
            if 'ses' not in subject_path.name:
                continue

            sub_name = subject_path.name.split('__')[1].split('_')[0]

            if method1 in subject_path.name:
                if sub_name not in dp_dict:
                    dp_dict[sub_name] = [subject_path]
                else:
                    dp_dict[sub_name].append(subject_path)
            else:
                if sub_name not in fs_dict:
                    fs_dict[sub_name] = [subject_path]
                else:
                    fs_dict[sub_name].append(subject_path)

        for project, proj_dict in zip([self.method1, self.method2], [dp_dict, fs_dict]):

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
        group_dir = Path(self.feature_dir, 'recon_individual_fsaverage6')
        stability_dir = Path(self.feature_dir, 'recon_stability_fsaverage6')

        for project in [self.method1, self.method2]:
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
        interp_dir = Path(self.feature_dir, 'recon_interp_fsaverage6')
        group_dir = Path(self.feature_dir, 'recon_group_fsaverage6')

        dp_list = list()
        fs_list = list()
        for subject_path in interp_dir.iterdir():
            if not 'sub' in subject_path.name:
                continue
            if 'ses' not in subject_path.name:
                continue

            if self.method1 in subject_path.name:
                dp_list.append(subject_path)
            else:
                fs_list.append(subject_path)

        for project, proj_list in zip([self.method1, self.method2], [dp_list, fs_list]):

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

    def cal_group_difference(self, feature='thickness', hemi='lh'):
        """
        Calculate the significance of difference (p-value) between subjects processed using DeepPrep & FreeSurfer on fs6.

        input: surf/?h.<feature>
        output: ?h_pvalue.<feature>
        """
        input_dir = Path(self.feature_dir, 'recon_interp_fsaverage6')
        output_dir = Path(self.feature_dir, 'recon_interp_fsaverage6_pvalue')

        folders = os.listdir(str(input_dir))
        method1_data = None
        method2_data = None
        for folder in folders:
            if folder.startswith(self.method1):
                file = Path(input_dir, folder, 'surf', f'{hemi}.{feature}')
                data = np.expand_dims(nib.freesurfer.read_morph_data(file), 1)
                if method1_data is None:
                    method1_data = data
                else:
                    method1_data = np.concatenate([method1_data, data], axis=1)
            elif folder.startswith(self.method2):
                file = Path(input_dir, folder, 'surf', f'{hemi}.{feature}')
                data = np.expand_dims(nib.freesurfer.read_morph_data(file), 1)
                if method1_data is None:
                    method2_data = data
                else:
                    method2_data = np.concatenate([method2_data, data], axis=1)
            else:
                raise SyntaxError(f"{folder} does not belong to either {self.method1} or {self.method2}")
        p_value = []
        for i in range(method1_data.shape[0]):
            _, p = ztest(method1_data[i], method2_data[i], alternative='two-sided')
            p_value.append(-np.log10(p))
        p_value = np.asarray(p_value)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        nib.freesurfer.write_morph_data(os.path.join(output_dir, f'{hemi}_pvalue.{feature}'), p_value)
        print(os.path.join(output_dir, f'{hemi}_pvalue.{feature}'))

    def concat_group_screenshot(self, screenshot_dir: Path, out_dir: Path, feature=''):
        """
        拼接method1和method2的分区结果图像
        """
        for stats_type in ['mean', 'std']:
            f1 = screenshot_dir / f'{self.method1}_{feature}_{stats_type}_lh_lateral.png'
            f2 = screenshot_dir / f'{self.method1}_{feature}_{stats_type}_rh_lateral.png'
            f3 = screenshot_dir / f'{self.method1}_{feature}_{stats_type}_lh_medial.png'
            f4 = screenshot_dir / f'{self.method1}_{feature}_{stats_type}_rh_medial.png'
            f5 = screenshot_dir / f'{self.method2}_{feature}_{stats_type}_lh_lateral.png'
            f6 = screenshot_dir / f'{self.method2}_{feature}_{stats_type}_rh_lateral.png'
            f7 = screenshot_dir / f'{self.method2}_{feature}_{stats_type}_lh_medial.png'
            f8 = screenshot_dir / f'{self.method2}_{feature}_{stats_type}_rh_medial.png'

            img_h1 = concat_vertical([f1, f3, f4, f2])
            img_h2 = concat_vertical([f5, f7, f8, f6])
            save_file = out_dir / f'{feature}_{stats_type}.png'
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
            img = concat_horizontal([img_h1, img_h2], str(save_file))
    def group_screenshot(self, feature='thickness', vmin1='', vmax1='', vmin2='', vmax2=''):
        """

        """
        for method in [self.method1, self.method2]:
            recon_dir = Path(self.feature_dir, f'recon_group_fsaverage6/{method}')
            out_dir = Path(self.feature_dir, f'recon_group_fsaverage6_screenshot/{method}')
            concat_dir = Path(self.feature_dir, 'recon_group_fsaverage6_screenshot_concat')

            subject_list = os.listdir(str(recon_dir))
            args_list1 = []
            args_list2 = []
            for subject in subject_list:
                for hemi in ['lh', 'rh']:
                    surf_file = os.path.join('/usr/local/freesurfer/subjects/fsaverage6', 'surf', f'{hemi}.pial')

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    for stats_type, (vmin, vmax) in zip(['mean', 'std'], [(vmin1, vmax1), (vmin2, vmax2)]):
                        overlay_file = Path(recon_dir, subject, 'surf', f'{hemi}.{stats_type}.{feature}')
                        save_file = Path(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_lateral.png')
                        args_list1.append([surf_file, overlay_file, save_file, vmin, vmax])
                        save_file = Path(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_medial.png')
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
        f5 = screenshot_dir / f'{self.method2}_{feature}_lh_lateral.png'
        f6 = screenshot_dir / f'{self.method2}_{feature}_rh_lateral.png'
        f7 = screenshot_dir / f'{self.method2}_{feature}_lh_medial.png'
        f8 = screenshot_dir / f'{self.method2}_{feature}_rh_medial.png'

        img_h1 = concat_vertical([f1, f3, f4, f2])
        img_h2 = concat_vertical([f5, f7, f8, f6])
        save_file = out_dir / f'{feature}.png'
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        img = concat_horizontal([img_h1, img_h2], str(save_file))
    def stability_screenshot(self, feature='thickness', vmin='', vmax=''):
        """
        """
        recon_dir = Path(self.feature_dir, 'recon_stability_fsaverage6')
        out_dir = Path(self.feature_dir, 'recon_stability_fsaverage6_screenshot')
        concat_dir = Path(self.feature_dir, 'recon_stability_fsaverage6_screenshot_concat')

        subject_list = os.listdir(recon_dir)
        args_list1 = []
        args_list2 = []
        for subject in subject_list:

            for hemi in ['lh', 'rh']:
                surf_file = os.path.join('/usr/local/freesurfer/subjects/fsaverage6', 'surf', f'{hemi}.pial')

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                overlay_file = Path(recon_dir, subject, 'surf', f'{hemi}.{feature}')
                save_file = Path(out_dir, f'{subject}_{feature}_{hemi}_lateral.png')
                args_list1.append([surf_file, overlay_file, save_file, vmin, vmax])
                save_file = os.path.join(out_dir, f'{subject}_{feature}_{hemi}_medial.png')
                args_list2.append([surf_file, overlay_file, save_file, vmin, vmax])
        pool = Pool(self.Multi_CPU_Num)
        pool.starmap(self.image_screenshot, args_list1)
        pool.starmap(self.image_screenshot_azimuth_180, args_list2)
        pool.close()
        pool.join()

        self.concat_stability_screenshot(out_dir, concat_dir, feature=feature)

    def concat_pvalue_screenshot(self, screenshot_dir: str, feature='thickness'):
        """
       拼接DeepPrep和FreeSurfer的p_value结果图像
       """
        lh_medial = os.path.join(screenshot_dir, f'lh_pvalue_{feature}_medial.png')
        lh_lateral = os.path.join(screenshot_dir, f'lh_pvalue_{feature}_lateral.png')
        rh_medial = os.path.join(screenshot_dir, f'rh_pvalue_{feature}_medial.png')
        rh_lateral = os.path.join(screenshot_dir, f'rh_pvalue_{feature}_lateral.png')

        img_h1 = concat_vertical([lh_lateral, lh_medial])
        img_h2 = concat_vertical([rh_lateral, rh_medial])
        save_path = os.path.join(os.path.dirname(screenshot_dir), os.path.basename(screenshot_dir) + '_concat')
        save_file = os.path.join(save_path, f'pvalue_{feature}.png')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = concat_horizontal([img_h1, img_h2], save_file)

    def p_value_screenshot(self, feature='thickness', vmin1='2.0', vmax1='5.0'):
        """
        读取p_value路径，并对fs6的p_value进行截图
        需要 ?h.sulc, ?h.curv, ?h.thickness
        """
        p_value_dir = Path(self.feature_dir, 'recon_interp_fsaverage6_pvalue')
        out_dir = Path(self.feature_dir, 'recon_interp_fsaverage6_pvalue_screenshot')

        args_list1 = []
        args_list2 = []
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for hemi in ['lh', 'rh']:
            if hemi == 'lh':
                surf_file = f'/usr/local/freesurfer/subjects/fsaverage6/surf/{hemi}.pial'
                overlay_file = os.path.join(p_value_dir, f'{hemi}_pvalue.{feature}')
                save_file = os.path.join(out_dir, f'{hemi}_pvalue_{feature}_lateral.png')
                args_list1.append([surf_file, overlay_file, save_file, vmin1, vmax1, 'heat'])
                save_file = os.path.join(out_dir, f'{hemi}_pvalue_{feature}_medial.png')
                args_list2.append([surf_file, overlay_file, save_file, vmin1, vmax1, 'heat'])
            else:
                surf_file = f'/usr/local/freesurfer/subjects/fsaverage6/surf/{hemi}.pial'
                overlay_file = os.path.join(p_value_dir, f'{hemi}_pvalue.{feature}')
                save_file = os.path.join(out_dir, f'{hemi}_pvalue_{feature}_medial.png')
                args_list1.append([surf_file, overlay_file, save_file, vmin1, vmax1, 'heat'])
                save_file = os.path.join(out_dir, f'{hemi}_pvalue_{feature}_lateral.png')
                args_list2.append([surf_file, overlay_file, save_file, vmin1, vmax1, 'heat'])
        pool = Pool(self.Multi_CPU_Num)
        pool.starmap(self.pvalue_image_screenshot, args_list1)
        pool.starmap(self.pvalue_image_screenshot_azimuth_180, args_list2)
        pool.close()
        pool.join()

        self.concat_pvalue_screenshot(out_dir, feature=feature)

class NegTriangleCount:
    """
    Calculate area of negative triangle and sum/average statistically
    """
    def __init__(self, dataset, method):
        self.dataset = dataset
        self.method = method
        if method == 'deepprep':
            self.subject_path = Path(f'/mnt/ngshare/DeepPrep/{self.dataset}/derivatives/{self.method}/Recon')
            self.save_path = Path(f'/mnt/ngshare/DeepPrep/Validation/{self.dataset}/v1_feature/neg/{self.method}/Recon')
        elif method == 'FreeSurfer':
            self.subject_path = Path(f'/mnt/ngshare/DeepPrep/{self.dataset}/derivatives/{self.method}')
            self.save_path = Path(f'/mnt/ngshare/DeepPrep/Validation/{self.dataset}/v1_feature/neg/{self.method}')
    def negative_area(self,faces, xyz):
        n = faces[:, 0]
        n0 = faces[:, 2]
        n1 = faces[:, 1]

        v0 = xyz[n] - xyz[n0]
        v1 = xyz[n1] - xyz[n]

        d1 = -v1[:, 1] * v0[:, 2] + v0[:, 1] * v1[:, 2]
        d2 = v1[:, 0] * v0[:, 2] - v0[:, 0] * v1[:, 2]
        d3 = -v1[:, 0] * v0[:, 1] + v0[:, 0] * v1[:, 1]

        dot = xyz[n][:, 0] * d1 + xyz[n][:, 1] * d2 + xyz[n][:, 2] * d3  # dot neg area neg
        area = torch.sqrt(d1 * d1 + d2 * d2 + d3 * d3)
        area[dot < 0] *= -1
        return area
    def negative_area_1(self,faces, xyz):
        n = faces[:, 0]
        n0 = faces[:, 2]
        n1 = faces[:, 1]

        v0 = xyz[n] - xyz[n0]
        v1 = xyz[n1] - xyz[n]

        d1 = -v1[:, 1] * v0[:, 2] + v0[:, 1] * v1[:, 2]
        d2 = v1[:, 0] * v0[:, 2] - v0[:, 0] * v1[:, 2]
        d3 = -v1[:, 0] * v0[:, 1] + v0[:, 0] * v1[:, 1]

        dot = xyz[n][:, 0] * d1 + xyz[n][:, 1] * d2 + xyz[n][:, 2] * d3  # dot neg area neg
        mask = dot < 0
        count = torch.zeros(xyz.shape[0])
        for i in range(len(mask)):
            if mask[i] == True:
                for j in range(len(faces[i])):
                    count[faces[i][j]] = 1
        return count
    def count_negative_area(self):
        """
        Count the negative triangle area and save it as neg and annot format
        """
        subject_list = os.listdir(self.subject_path)
        subject_list.sort()
        i = 0
        while i < len(subject_list):
            if 'fsaverage' in subject_list[i]:
                subject_list.remove(subject_list[i])
                i -= 1
            i += 1
        count_num = 1
        for hemi in ['lh', 'rh']:
            fs6_annot_path = Path(
                f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon/fsaverage6/label/{hemi}.aparc.annot')
            _, ctab, names = nib.freesurfer.io.read_annot(fs6_annot_path)
            for subject in subject_list:
                if not (self.subject_path / subject).is_dir():
                    continue
                sphere_file = self.subject_path / subject / 'surf' / f'{hemi}.sphere.reg'
                xyz_sphere, faces_sphere = nib.freesurfer.read_geometry(str(sphere_file))
                device = 'cuda'
                xyz_sphere = torch.from_numpy(xyz_sphere.astype(np.float32)).to(device)
                faces_sphere = torch.from_numpy(faces_sphere.astype(int)).to(device)
                count= self.negative_area_1(faces_sphere, xyz_sphere).numpy()
                file_like_path = self.save_path / subject / 'surf'
                if not file_like_path.exists():
                    file_like_path.mkdir(parents=True, exist_ok=True)
                nib.freesurfer.io.write_morph_data(os.path.join(file_like_path,f'{hemi}.neg'), count, fnum=0)
                nib.freesurfer.io.write_annot(os.path.join(file_like_path,f'{hemi}.neg.annot'), count.astype('int64'),
                                              ctab.astype('int64'), names)
                print('count_num:',count_num)
                count_num += 1

    def remove_negative_area(self,faces, xyz, device='cuda'):
        """
        基于laplacian smoothing的原理
        https://en.wikipedia.org/wiki/Laplacian_smoothing
        """
        from torch_scatter import scatter_mean
        area = self.negative_area(faces, xyz)
        index = area < 0
        count = index.sum()  # 面积为负的面
        print(f'negative area count : {count}')

        remove_times = 0
        dt_weight_init = 1  # 初始值

        x = np.expand_dims(faces.cpu()[:, 0], 1)
        y = np.expand_dims(faces.cpu()[:, 1], 1)
        z = np.expand_dims(faces.cpu()[:, 2], 1)

        a = np.concatenate([x, y], axis=1)
        b = np.concatenate([y, x], axis=1)
        c = np.concatenate([x, z], axis=1)
        d = np.concatenate([z, x], axis=1)
        e = np.concatenate([y, z], axis=1)
        f = np.concatenate([z, y], axis=1)

        edge_index = np.concatenate([a, b, c, d, e, f])
        edge_index = torch.from_numpy(edge_index).to(device)
        edge_index = edge_index.t().contiguous()

        row, col = edge_index

        while count > 0:
            dt_weight = dt_weight_init - count % 10 * 0.01  # 按比例减小

            xyz_dt = scatter_mean(xyz[col], row, dim=0) - xyz
            neg_faces = faces[index]
            index = neg_faces.flatten()
            xyz[index] = xyz[index] + xyz_dt[index] * dt_weight
            xyz = xyz / torch.norm(xyz, dim=1, keepdim=True) * 100

            area = self.negative_area(faces, xyz)
            index = area < 0
            count = index.sum()  # 面积为负的面
            print(f'negative area count : {count}')
            remove_times += 1

            if remove_times >= 1000:
                break

        return xyz, count, remove_times
    def remove_negative_area_and_save_sphere(self):
        """
        Remove area of negative triangle and save as new sphere
        """
        subject_list = os.listdir(self.subject_path)
        subject_list.sort()
        i = 0
        while i < len(subject_list):
            if 'fsaverage' in subject_list[i]:
                subject_list.remove(subject_list[i])
                i -= 1
            i += 1
        count_num = 1
        for hemi in ['lh', 'rh']:
            for subject in subject_list:
                if not (self.subject_path / subject).is_dir():
                    continue
                sphere_file = self.subject_path / subject / 'surf' / f'{hemi}.sphere.reg'
                xyz_sphere, faces_sphere = nib.freesurfer.read_geometry(str(sphere_file))
                device = 'cuda'
                xyz_sphere = torch.from_numpy(xyz_sphere.astype(np.float32)).to(device)
                faces_sphere = torch.from_numpy(faces_sphere.astype(int)).to(device)
                area = self.negative_area(faces_sphere, xyz_sphere)
                index = area < 0
                count_orig = index.sum()  # 面积为负的面
                print(f'negative area count : {count_orig}')
                times = count_final = 0
                if count_orig > 0:
                    xyz_sphere, count_final, times = self.remove_negative_area(faces_sphere, xyz_sphere)

                xyz_sphere = xyz_sphere.cpu().numpy()
                faces_sphere = faces_sphere.cpu().numpy()
                file_like = self.subject_path / subject / 'surf' / f'{hemi}.rna.sphere.reg'
                nib.freesurfer.io.write_geometry(file_like, xyz_sphere, faces_sphere)
                print('count_num:', count_num)
                count_num += 1
    def set_environ(self,dataset,method):
        # FreeSurfer
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
        if method == 'deepprep':
            os.environ['SUBJECTS_DIR'] = f'/mnt/ngshare/DeepPrep/{dataset}/derivatives/{method}/Recon'
        elif method == 'FreeSurfer':
            os.environ['SUBJECTS_DIR'] = f'/mnt/ngshare/DeepPrep/{dataset}/derivatives/{method}'

        os.environ['PATH'] = '/usr/local/freesurfer/bin:' + os.environ['PATH']
    def surf2surf_use_annot(self, dataset, method, type_of_sphere='ori'):
        """
        type_of_sphere : 'ori' or 'rna'
        """
        self.set_environ(dataset, method)
        subject_list = os.listdir(self.subject_path)
        subject_list.sort()

        i = 0
        while i < len(subject_list):
            if 'fsaverage' in subject_list[i]:
                subject_list.remove(subject_list[i])
                i -= 1
            i += 1
        for hemi in ['lh', 'rh']:
            for subject in subject_list:
                if not (self.subject_path / subject).is_dir():
                    continue
                annot1 = self.save_path / subject / 'surf' / f'{hemi}.neg.annot'
                if type_of_sphere == 'ori':
                    tval = self.save_path / subject / 'surf' / f'{hemi}.neg.40962.annot'
                    cmd = f'mri_surf2surf --srcsubject {subject} --sval-annot {annot1} ' \
                          f'--trgsubject fsaverage6 --tval {tval} --hemi {hemi} --surfreg sphere.reg'
                elif type_of_sphere == 'rna':
                    tval = self.save_path / subject / 'surf' / f'{hemi}.rna.neg.40962.annot'
                    cmd = f'mri_surf2surf --srcsubject {subject} --sval-annot {annot1} ' \
                      f'--trgsubject fsaverage6 --tval {tval} --hemi {hemi} --surfreg rna.sphere.reg'
                os.system(cmd)

    def statistic_native_area(self,dataset,metohd,type_of_sphere='ori'):
        subject_list = os.listdir(self.subject_path)
        subject_list.sort()

        i = 0
        while i < len(subject_list):
            if 'fsaverage' in subject_list[i]:
                subject_list.remove(subject_list[i])
                i -= 1
            i += 1
        sum = np.zeros(40962)
        for hemi in ['lh', 'rh']:
            fs6_annot_path = Path(
                f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon/fsaverage6/label/{hemi}.aparc.annot')
            _, ctab, names = nib.freesurfer.io.read_annot(fs6_annot_path)
            for subject in subject_list:
                if not (self.subject_path / subject).is_dir():
                    continue
                if type_of_sphere == 'ori':
                    label_path = self.save_path / subject / 'surf' / f'{hemi}.neg.40962.annot'
                elif type_of_sphere == 'rna':
                    label_path = self.subject_path / subject / 'surf' / f'{hemi}.rna.neg.40962.annot'
                label, _, _ = nib.freesurfer.io.read_annot(label_path)
                sum += label
            path = f'/mnt/ngshare/DeepPrep/Validation/{dataset}/v1_feature/neg'
            file_like_path = Path(os.path.join(path, f'statistic_native_area_{metohd}'))
            if not file_like_path.exists():
                file_like_path.mkdir(parents=True, exist_ok=True)
            nib.freesurfer.io.write_morph_data(os.path.join(file_like_path,f'{hemi}.neg.rna.statistic.sulc'),
                                               sum.astype('int64'))
if __name__ == '__main__':
    set_environ()
    # Calculate Accuracy and stability
    neg = NegTriangleCount('MSC','deepprep')
    # neg.count_negative_area()
    # neg.remove_negative_area_and_save_sphere()
    # neg.surf2surf_use_annot('MSC','deepprep','rna')
    # neg.statistic_native_area('MSC','deepprep')
    exit()
    cls = AccAndStability('MSC', 'DeepPrep')
    cls.ants_reg('Rigid')
    cls.aseg_acc('FreeSurfer', 'DeepPrep')
    cls.aseg_stability('FreeSurfer', aseg=True)
    cls.aseg_stability('DeepPrep', aseg=True)
    cls.aparc_stability('/run/user/1000/gvfs/sftp:host=30.30.30.66,user=zhenyu/home/zhenyu/workdata/App/MSC_DeepPrep_processed',
                        92, method="DeepPrep")
    cls.aparc_stability(
        '/run/user/1000/gvfs/sftp:host=30.30.30.66,user=zhenyu/mnt/ngshare2/App/MSC_app',
        92, method="App")
    exit()


    method1 = 'DeepPrep'
    method2 = 'FreeSurfer'
    pipeline = 'FreeSurferFastSurferFastCSRFeatReg'
    dataset = 'MSC'
    screenshot = ScreenShot(pipeline, dataset, method1, method2)
    for feature, (vmin, vmax), (vmin2, vmax2), (vmin3, vmax3) in zip(['thickness', 'curv', 'sulc'],
                                     [('1', '3.5'), ('-0.5', '0.25'), ('-13', '13'),],
                                     [('0', '0.35'), ('0', '0.05'), ('0', '1.3'),],
                                     [('0', '0.35'), ('0', '0.05'), ('0', '1.3')],):
        screenshot.feature_screenshot(feature)
        screenshot.feature_screenshot(feature)
        screenshot.ln_subject()

        for hemi in ['lh', 'rh']:
            screenshot.project_fsaverage6(feature=feature, hemi=hemi)
            screenshot.cal_individual_fsaverage6(feature=feature, hemi=hemi)
            screenshot.cal_stability_fsaverage6(feature=feature, hemi=hemi)
            screenshot.cal_group_fsaverage6(feature=feature, hemi=hemi)
            screenshot.cal_group_difference(feature=feature, hemi=hemi)

        screenshot.group_screenshot(feature=feature,  vmin1=vmin, vmax1=vmax, vmin2=vmin2,vmax2=vmax2)
        screenshot.stability_screenshot(feature=feature, vmin=vmin3, vmax=vmax3)
        screenshot.p_value_screenshot(feature=feature, vmin1='2.0', vmax1='5.0')

