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


def set_environ():
    # FreeSurfer
    value = os.environ.get('FREESURFER_HOME')
    if value is None:
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
        os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
        os.environ['PATH'] = '/usr/local/freesurfer/bin:' + os.environ['PATH']


def run_cmd(cmd):
    print(f'shell_run : {cmd}')
    os.system(cmd)
    print('*' * 40)


def image_screenshot(surf_file, overlay_file, save_path, min, max, overlay_color='colorwheel'):
    cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color={overlay_color},inverse -cam dolly 1.4 azimuth 0 -ss {save_path}'
    # cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color=colorwheel,inverse -colorscale -cam dolly 1.4 azimuth 0 -ss {save_path}'
    # cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f "{surf_file}":annotation="{overlay_file}" -cam dolly 1.4 azimuth 0 -ss {save_path}'
    print(cmd)
    os.system(cmd)


def image_screenshot_azimuth_180(surf_file, overlay_file, save_path, min, max, overlay_color='colorwheel'):
    cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color={overlay_color},inverse -cam dolly 1.4 azimuth 180 -ss {save_path}'
    # cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color=colorwheel,inverse -colorscale -cam dolly 1.4 azimuth 180 -ss {save_path}'
    # cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f "{surf_file}":annotation="{overlay_file}" -cam dolly 1.4 azimuth 180 -ss {save_path}'
    print(cmd)
    os.system(cmd)


def feature_screenshot(recon_dir, out_dir, feature='thickness', vmin='', vmax=''):
    """
    读取FreeSurfer格式的目录，并对aparc结构进行截图
    需要 surf/?h.pial 和 label/?h.aparc.annot
    """

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
            # image_screenshot(surf_file, overlay_file, save_file, min=vmin, max=vmax)
            save_file = os.path.join(out_dir, f'{subject}_{hemi}_medial.png')
            args_list2.append([surf_file, overlay_file, save_file, vmin, vmax])
            # image_screenshot_azimuth_180(surf_file, overlay_file, save_file, min=vmin, max=vmax)
    pool = Pool(Multi_CPU_Num)
    pool.starmap(image_screenshot, args_list1)
    pool.starmap(image_screenshot_azimuth_180, args_list2)
    pool.close()
    pool.join()


def concat_screenshot(screenshot_dir: str):
    """
    拼接DeepPrep和FreeSurfer的分区结果图像
    """

    filenames = os.listdir(os.path.join(screenshot_dir, 'feature_map_image_DeepPrep'))
    subjects = set(['_'.join(i.split('_')[0:-2]) for i in filenames])

    for subject in subjects:
        f1 = os.path.join(screenshot_dir, 'feature_map_image_DeepPrep', f'{subject}_lh_lateral.png')
        f2 = os.path.join(screenshot_dir, 'feature_map_image_DeepPrep', f'{subject}_rh_lateral.png')
        f3 = os.path.join(screenshot_dir, 'feature_map_image_DeepPrep', f'{subject}_lh_medial.png')
        f4 = os.path.join(screenshot_dir, 'feature_map_image_DeepPrep', f'{subject}_rh_medial.png')
        f5 = os.path.join(screenshot_dir, 'feature_map_image_FreeSurfer', f'{subject}_lh_lateral.png')
        f6 = os.path.join(screenshot_dir, 'feature_map_image_FreeSurfer', f'{subject}_rh_lateral.png')
        f7 = os.path.join(screenshot_dir, 'feature_map_image_FreeSurfer', f'{subject}_lh_medial.png')
        f8 = os.path.join(screenshot_dir, 'feature_map_image_FreeSurfer', f'{subject}_rh_medial.png')

        # img_h1 = concat_vertical([f1, f2, f5, f6])
        # img_h2 = concat_vertical([f3, f4, f7, f8])
        img_h1 = concat_vertical([f1, f3, f4, f2])
        img_h2 = concat_vertical([f5, f7, f8, f6])
        save_path = os.path.join(screenshot_dir, 'feature_map_image_concat')
        save_file = os.path.join(save_path, f'{subject}.png')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = concat_horizontal([img_h1, img_h2], save_file)
        # img = cv2.resize(img, (1568, 718))
        # cv2.imshow('imshow', img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("imshow")
        # break


def ln_subject(deepprep_dir: Path, freesurfer_dir: Path, concat_dir: Path):
    concat_dir.mkdir(parents=True, exist_ok=True)
    for subject_path in deepprep_dir.iterdir():
        target = concat_dir / f'DeepPrep__{subject_path.name}'
        if not target.exists():
            print(subject_path)
            target.symlink_to(subject_path, target_is_directory=True)

    for subject_path in freesurfer_dir.iterdir():
        target = concat_dir / f'FreeSurfer__{subject_path.name}'
        if not target.exists():
            print(subject_path)
            target.symlink_to(subject_path, target_is_directory=True)


def project_fsaverage6(recon_dir: Path, output_dir: Path, feature='thickness', hemi='lh'):
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
            # print(f'shell_run : {cmd}')
            # os.system(cmd)
            # print('*' * 40)
            args_list.append((cmd,))
    pool = Pool(Multi_CPU_Num)
    pool.starmap(run_cmd, args_list)
    pool.close()
    pool.join()


def cal_individual_fsaverage6(interp_dir: Path, individual_dir: Path, feature='thickness', hemi='lh'):
    dp_dict = dict()
    fs_dict = dict()
    for subject_path in interp_dir.iterdir():
        if not 'sub' in subject_path.name:
            continue
        if 'ses' not in subject_path.name:
            continue

        sub_name = subject_path.name.split('__')[1].split('_')[0]

        if 'DeepPrep' in subject_path.name:
            if sub_name not in dp_dict:
                dp_dict[sub_name] = [subject_path]
            else:
                dp_dict[sub_name].append(subject_path)
        else:
            if sub_name not in fs_dict:
                fs_dict[sub_name] = [subject_path]
            else:
                fs_dict[sub_name].append(subject_path)

    for project, proj_dict in zip(['DeepPrep', 'FreeSurfer'], [dp_dict, fs_dict]):

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


def individual_screenshot(recon_dir, out_dir, feature='thickness', vmin1='', vmax1='', vmin2='', vmax2=''):
    """
    读取FreeSurfer格式的目录，并对aparc结构进行截图
    需要 surf/?h.pial 和 label/?h.aparc.annot
    """

    subject_list = os.listdir(recon_dir)
    args_list1 = []
    args_list2 = []
    for subject in subject_list:
        if not 'sub' in subject:
            continue
        # if 'ses' in subject:
        #     continue

        for hemi in ['lh', 'rh']:
            surf_file = os.path.join('/usr/local/freesurfer/subjects/fsaverage6', 'surf', f'{hemi}.pial')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for stats_type, (vmin, vmax) in zip(['mean', 'std'], [(vmin1, vmax1), (vmin2, vmax2)]):

                overlay_file = os.path.join(recon_dir, subject, 'surf', f'{hemi}.{stats_type}.{feature}')
                save_file = os.path.join(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_lateral.png')
                if not os.path.exists(save_file):
                    # image_screenshot(surf_file, overlay_file, save_file, min=vmin, max=vmax)
                    args_list1.append([surf_file, overlay_file, save_file, vmin, vmax])
                save_file = os.path.join(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_medial.png')
                if not os.path.exists(save_file):
                    # image_screenshot_azimuth_180(surf_file, overlay_file, save_file, min=vmin, max=vmax)
                    args_list2.append([surf_file, overlay_file, save_file, vmin, vmax])
    pool = Pool(Multi_CPU_Num)
    pool.starmap(image_screenshot, args_list1)
    pool.starmap(image_screenshot_azimuth_180, args_list2)
    pool.close()
    pool.join()


def cal_stability_fsaverage6(group_dir: Path, stability_dir: Path, feature='thickness', hemi='lh'):
    for project in ['DeepPrep', 'FreeSurfer']:
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


def stability_screenshot(recon_dir, out_dir, feature='thickness', vmin='', vmax=''):
    """
    读取FreeSurfer格式的目录，并对aparc结构进行截图
    需要 surf/?h.pial 和 label/?h.aparc.annot
    """

    subject_list = os.listdir(recon_dir)
    args_list1 = []
    args_list2 = []
    for subject in subject_list:

        for hemi in ['lh', 'rh']:
            surf_file = os.path.join('/usr/local/freesurfer/subjects/fsaverage6', 'surf', f'{hemi}.pial')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            overlay_file = os.path.join(recon_dir, subject, 'surf', f'{hemi}.{feature}')
            save_file = os.path.join(out_dir, f'{subject}_{feature}_{hemi}_lateral.png')
            # image_screenshot(surf_file, overlay_file, save_file, min=vmin, max=vmax)
            args_list1.append([surf_file, overlay_file, save_file, vmin, vmax])
            save_file = os.path.join(out_dir, f'{subject}_{feature}_{hemi}_medial.png')
            # image_screenshot_azimuth_180(surf_file, overlay_file, save_file, min=vmin, max=vmax)
            args_list2.append([surf_file, overlay_file, save_file, vmin, vmax])
    pool = Pool(Multi_CPU_Num)
    pool.starmap(image_screenshot, args_list1)
    pool.starmap(image_screenshot_azimuth_180, args_list2)
    pool.close()
    pool.join()


def concat_stability_screenshot(screenshot_dir: Path, out_dir: Path, feature=''):
    """
    拼接DeepPrep和FreeSurfer的分区结果图像
    """

    f1 = screenshot_dir / f'DeepPrep_{feature}_lh_lateral.png'
    f2 = screenshot_dir / f'DeepPrep_{feature}_rh_lateral.png'
    f3 = screenshot_dir / f'DeepPrep_{feature}_lh_medial.png'
    f4 = screenshot_dir / f'DeepPrep_{feature}_rh_medial.png'
    f5 = screenshot_dir / f'FreeSurfer_{feature}_lh_lateral.png'
    f6 = screenshot_dir / f'FreeSurfer_{feature}_rh_lateral.png'
    f7 = screenshot_dir / f'FreeSurfer_{feature}_lh_medial.png'
    f8 = screenshot_dir / f'FreeSurfer_{feature}_rh_medial.png'

    # img_h1 = concat_vertical([f1, f2, f5, f6])
    # img_h2 = concat_vertical([f3, f4, f7, f8])
    img_h1 = concat_vertical([f1, f3, f4, f2])
    img_h2 = concat_vertical([f5, f7, f8, f6])
    save_file = out_dir / f'{feature}.png'
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    img = concat_horizontal([img_h1, img_h2], str(save_file))


def cal_group_fsaverage6(interp_dir: Path, group_dir: Path, feature='thickness', hemi='lh'):
    dp_list = list()
    fs_list = list()
    for subject_path in interp_dir.iterdir():
        if not 'sub' in subject_path.name:
            continue
        if 'ses' not in subject_path.name:
            continue

        if 'DeepPrep' in subject_path.name:
            dp_list.append(subject_path)
        else:
            fs_list.append(subject_path)

    for project, proj_list in zip(['DeepPrep', 'FreeSurfer'], [dp_list, fs_list]):

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


def group_screenshot(recon_dir, out_dir, feature='thickness', vmin1='', vmax1='', vmin2='', vmax2=''):
    """
    读取FreeSurfer格式的目录，并对aparc结构进行截图
    需要 surf/?h.pial 和 label/?h.aparc.annot
    """

    subject_list = os.listdir(recon_dir)
    args_list1 = []
    args_list2 = []
    for subject in subject_list:
        for hemi in ['lh', 'rh']:
            surf_file = os.path.join('/usr/local/freesurfer/subjects/fsaverage6', 'surf', f'{hemi}.pial')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for stats_type, (vmin, vmax) in zip(['mean', 'std'], [(vmin1, vmax1), (vmin2, vmax2)]):

                overlay_file = os.path.join(recon_dir, subject, 'surf', f'{hemi}.{stats_type}.{feature}')
                save_file = os.path.join(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_lateral.png')
                # image_screenshot(surf_file, overlay_file, save_file, min=vmin, max=vmax)
                args_list1.append([surf_file, overlay_file, save_file, vmin, vmax])
                save_file = os.path.join(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_medial.png')
                # image_screenshot_azimuth_180(surf_file, overlay_file, save_file, min=vmin, max=vmax)
                args_list2.append([surf_file, overlay_file, save_file, vmin, vmax])
    pool = Pool(Multi_CPU_Num)
    pool.starmap(image_screenshot, args_list1)
    pool.starmap(image_screenshot_azimuth_180, args_list2)
    pool.close()
    pool.join()


def concat_group_screenshot(screenshot_dir: Path, out_dir: Path, feature=''):
    """
    拼接DeepPrep和FreeSurfer的分区结果图像
    """
    for stats_type in ['mean', 'std']:

        f1 = screenshot_dir / f'DeepPrep_{feature}_{stats_type}_lh_lateral.png'
        f2 = screenshot_dir / f'DeepPrep_{feature}_{stats_type}_rh_lateral.png'
        f3 = screenshot_dir / f'DeepPrep_{feature}_{stats_type}_lh_medial.png'
        f4 = screenshot_dir / f'DeepPrep_{feature}_{stats_type}_rh_medial.png'
        f5 = screenshot_dir / f'FreeSurfer_{feature}_{stats_type}_lh_lateral.png'
        f6 = screenshot_dir / f'FreeSurfer_{feature}_{stats_type}_rh_lateral.png'
        f7 = screenshot_dir / f'FreeSurfer_{feature}_{stats_type}_lh_medial.png'
        f8 = screenshot_dir / f'FreeSurfer_{feature}_{stats_type}_rh_medial.png'

        # img_h1 = concat_vertical([f1, f2, f5, f6])
        # img_h2 = concat_vertical([f3, f4, f7, f8])
        img_h1 = concat_vertical([f1, f3, f4, f2])
        img_h2 = concat_vertical([f5, f7, f8, f6])
        save_file = out_dir / f'{feature}_{stats_type}.png'
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        img = concat_horizontal([img_h1, img_h2], str(save_file))

def cal_group_difference(fs6_deepprep_freesurfer: Path, output_dir: Path, feature='thickness', hemi='lh'):
    """
    Calculate the significance of difference (p-value) between subjects processed using DeepPrep & FreeSurfer on fs6.
    """
    folders = os.listdir(fs6_deepprep_freesurfer)
    deepprep_data = None
    freesurfer_data = None
    for folder in folders:
        if folder.lower().startswith('deepprep'):
            file = Path(fs6_deepprep_freesurfer, folder, 'surf', f'{hemi}.{feature}')
            data = np.expand_dims(nib.freesurfer.read_morph_data(file), 1)
            if deepprep_data is None:
                deepprep_data = data
            else:
                deepprep_data = np.concatenate([deepprep_data, data], axis=1)
        elif folder.lower().startswith('freesurfer'):
            file = Path(fs6_deepprep_freesurfer, folder, 'surf', f'{hemi}.{feature}')
            data = np.expand_dims(nib.freesurfer.read_morph_data(file), 1)
            if deepprep_data is None:
                freesurfer_data = data
            else:
                freesurfer_data = np.concatenate([freesurfer_data, data], axis=1)
        else:
            raise SyntaxError(f"{folder} does not belong to either DeepPrep or FreeSurfer")
    print(deepprep_data.shape)
    print(freesurfer_data.shape)
    p_value = []
    for i in range(deepprep_data.shape[0]):
        _, p = ztest(deepprep_data[i], freesurfer_data[i], alternative='two-sided')
        p_value.append(-np.log10(p))
    p_value = np.asarray(p_value)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    nib.freesurfer.write_morph_data(os.path.join(output_dir, f'{hemi}_pvalue.{feature}'), p_value)
    print(os.path.join(output_dir, f'{hemi}_pvalue.{feature}'))

def pvalue_image_screenshot(surf_file, overlay_file, save_path, min, max, overlay_color='heat'):
    cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color={overlay_color}, -cam dolly 1.4 azimuth 0 -ss {save_path}'
    print(cmd)
    os.system(cmd)
def pvalue_image_screenshot_azimuth_180(surf_file, overlay_file, save_path, min, max, overlay_color='heat'):
    cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color={overlay_color}, -cam dolly 1.4 azimuth 180 -ss {save_path}'
    print(cmd)
    os.system(cmd)
def p_value_screenshot(p_value_dir, out_dir, feature='thickness', vmin1='2.0', vmax1='5.0'):
    """
    读取p_value路径，并对fs6的p_value进行截图
    需要 ?h.sulc, ?h.curv, ?h.thickness
    """
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
    pool = Pool(Multi_CPU_Num)
    pool.starmap(pvalue_image_screenshot, args_list1)
    pool.starmap(pvalue_image_screenshot_azimuth_180, args_list2)
    pool.close()
    pool.join()

def concat_pvalue_screenshot(screenshot_dir: str, feature='thickness'):
    """
   拼接DeepPrep和FreeSurfer的p_value结果图像
   """
    lh_medial = os.path.join(screenshot_dir, f'lh_pvalue_{feature}_medial.png')
    lh_lateral = os.path.join(screenshot_dir, f'lh_pvalue_{feature}_lateral.png')
    rh_medial = os.path.join(screenshot_dir, f'rh_pvalue_{feature}_medial.png')
    rh_lateral = os.path.join(screenshot_dir, f'rh_pvalue_{feature}_lateral.png')

    img_h1 = concat_vertical([lh_lateral, lh_medial])
    img_h2 = concat_vertical([rh_lateral, rh_medial])
    save_path = os.path.join(os.path.dirname(screenshot_dir), os.path.basename(screenshot_dir)+'_concat')
    save_file = os.path.join(save_path, f'pvalue_{feature}.png')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = concat_horizontal([img_h1, img_h2], save_file)
    print()


def ants_reg(moving_dir: Path, dest_dir: Path, type_of_transform='SyN'):
    fixed_dir = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
    dest_dir = Path(dest_dir)
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
        aseg_moved = ants.apply_transforms(fixed, aseg_moving, [str(warp_dir), str(affine_dir)], interpolator='multiLabel')
        aseg_movd_dir = Path(reg_output_dir, 'aseg_mni152.mgz')
        ants.image_write(aseg_moved, str(aseg_movd_dir))
        print(f'{file} done')

def info_label(aseg = True):
    if aseg:
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
    else:
        aparc_label = [-1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18]
        aparc_label_dict = {-1: 'unknown',
                            1: 'bankssts',
                            2: 'caudalanteriorcingulate',
                            3: 'caudalmiddlefrontal',
                            4: 'corpuscallosum',
                            5: 'cuneus',
                            6: 'entorhinal',
                            7: 'fusiform',
                            8: 'inferiorparietal',
                            9: 'inferiortemporal',
                            10: 'isthmuscingulate',
                            11: 'lateraloccipital',
                            12: 'lateralorbitofrontal',
                            13: 'lingual',
                            14: 'medialorbitofrontal',
                            15: 'middletemporal',
                            16: 'parahippocampal',
                            17: 'paracentral',
                            18: 'parsopercularis'}
        return aparc_label, aparc_label_dict

def dc(pred, gt):
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


def compute_aseg_dice(gt, pred):
    num_ref = np.sum(gt)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 1
        else:
            return 0
    else:
        return dc(pred, gt)

def evaluate_aseg(arr: np.ndarray, arr_gt: np.ndarray, aseg_i):
    mask_gt = (arr_gt == aseg_i).astype(int)
    mask_pred = (arr == aseg_i).astype(int)
    return compute_aseg_dice(mask_gt, mask_pred)
    del mask_gt, mask_pred

def aseg_acc(fs_dir: Path, deepprep_dir: Path, output_dir: Path):
    label, label_dict = info_label(aseg=True)
    df_dice = None

    for sub in os.listdir(fs_dir):
        dice_dict = {}
        dice_dict['sub'] = sub
        fs_aseg_dir = Path(fs_dir, sub, 'aseg_mni152.mgz')
        deepprep_aseg_dir = Path(deepprep_dir, sub, 'aseg_mni152.mgz')
        fs_aseg = ants.image_read(str(fs_aseg_dir)).numpy()
        if not deepprep_aseg_dir.exists():
            print(f"{sub} found in {fs_dir}, NOT in {deepprep_dir}")
            continue
        deepprep_aseg = ants.image_read(str(deepprep_aseg_dir)).numpy()

        for i in label:
            dice = evaluate_aseg(fs_aseg, deepprep_aseg, i)
            dice_dict[label_dict[i]] = [dice]

        df = pd.DataFrame.from_dict(dice_dict)

        if df_dice is None:
            df_dice = df
        else:
            df_dice = pd.concat((df_dice, df), axis=0)
        print(f'{sub} done')

    df_dice.loc['mean'] = df_dice.mean(axis=0)
    df_dice.loc['std'] = df_dice.std(axis=0)
    df_dice.to_csv(output_dir, index=False)

def aseg_stability(fs_dir, output_dir, aseg=True):
    label, label_dict = info_label(aseg=aseg)
    fs_dict = {}

    for sub in os.listdir(fs_dir):
        sub_id = sub.split('_')[0]
        if sub_id not in fs_dict:
            fs_dict[sub_id] = [sub]
        else:
            fs_dict[sub_id].append(sub)

    df_dice = pd.DataFrame(columns=label_dict.values(), index=sorted(fs_dict.keys()))

    # calculate dice of all pairs
    for sub in sorted(fs_dict.keys()):
        print(sub)
        df_sub_dice = None
        dice_dict = {}
        i = 0
        while i < len(fs_dict[sub]):
            aseg_i = fs_dict[sub][i]
            if aseg_i != sub:
                for j in range(i+1, len(fs_dict[sub])):
                    aseg_j = fs_dict[sub][j]
                    if aseg_j != sub:
                        i_dir = Path(fs_dir, aseg_i, 'aseg_mni152.mgz')
                        j_dir = Path(fs_dir, aseg_j, 'aseg_mni152.mgz')
                        print(aseg_i, aseg_j)
                        dice_dict['sub'] = aseg_i + ' & ' + aseg_j
                        i_aseg = ants.image_read(str(i_dir)).numpy()
                        j_aseg = ants.image_read(str(j_dir)).numpy()

                        for l in label:
                            dice = evaluate_aseg(i_aseg, j_aseg, l)
                            dice_dict[label_dict[l]] = [dice]

                        df = pd.DataFrame.from_dict(dice_dict)

                        if df_sub_dice is None:
                            df_sub_dice = df
                        else:
                            df_sub_dice = pd.concat((df_sub_dice, df), axis=0)

            i += 1

        # df_sub_dice.to_csv()
        df_dice.loc[sub] = df_sub_dice.std(axis=0)

    df_dice.to_csv(output_dir)

def aparc_stability(input_dir, output_dir, aseg):
    label, label_dict = info_label(aseg=aseg)
    sub_id = [sub for sub in sorted(os.listdir(input_dir))]
    dict = {}
    for sub in sub_id:
        sub_dir = Path(input_dir, sub)
        dict[sub] = sorted(os.listdir(sub_dir))

    for hemi in ['lh', 'rh']:

        for sub in sorted(dict.keys()):
            print(sub)

            i = 0
            while i < len(dict[sub]):
                aparc_i = dict[sub][i]
                for j in range(i+1, len(dict[sub])):
                    aparc_j = dict[sub][j]
                    i_dir = glob(os.path.join(input_dir, sub, aparc_i, f'parc/{aparc_i}/*/{hemi}_parc_result.annot'))[0]
                    j_dir = glob(os.path.join(input_dir, sub, aparc_j, f'parc/{aparc_j}/*/{hemi}_parc_result.annot'))[0]
                    print(aparc_i, aparc_j)
                i += 1





if __name__ == '__main__':
    set_environ()
    Multi_CPU_Num = 10

    t1_moving_dir = '/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon'
    dest_dir = '/mnt/ngshare/DeepPrep/Validation/MSC/v1_aparc/aparc_deepprepreg_to_mni152'

    # ants_reg(t1_moving_dir, dest_dir, type_of_transform='SyN')

    # DeepPrep和FreeSurfer的结果计算精度DICE
    fs_dir = '/mnt/ngshare/DeepPrep/Validation/MSC/v1_aparc/aparc_fsreg_to_mni152'
    deepprep_dir = '/mnt/ngshare/DeepPrep/Validation/MSC/v1_aparc/aparc_deepprepreg_to_mni152'
    output_dir = '/mnt/ngshare/DeepPrep/Validation/MSC/v1_aparc/aparc_fsreg_deepprep_mni152_dc.csv'
    # aseg_acc(fs_dir, deepprep_dir, output_dir)

    # DeepPrep和FreeSurfer的结果计算稳定性
    fs_dir = '/mnt/ngshare/DeepPrep/Validation/MSC/v1_aparc/aparc_fsreg_to_mni152'
    deepprep_dir = '/mnt/ngshare/DeepPrep/Validation/MSC/v1_aparc/aparc_deepprepreg_to_mni152'
    fs_output_dir = '/mnt/ngshare/DeepPrep/Validation/MSC/v1_aparc/aseg_fsreg_mni152_stability.csv'
    deepprep_output_dir = '/mnt/ngshare/DeepPrep/Validation/MSC/v1_aparc/aseg_deepprepreg_mni152_stability.csv'
    # aseg_stability(fs_dir, fs_output_dir)
    # aseg_stability(deepprep_dir, deepprep_output_dir)

    # 功能分区稳定性
    input_dir = '/run/user/1000/gvfs/sftp:host=30.30.30.66,user=zhenyu/home/zhenyu/workdata/App/MSC'
    output_dir = '/mnt/ngshare/DeepPrep/Validation/MSC/v1_aparc/aparc_MSC_stability.csv'
    aparc_stability(input_dir, output_dir, aseg=False)


    # ############# 分区截图
    # for feature in ['thickness', 'sulc', 'curv']:
    # for feature, (vmin, vmax) in zip(['thickness'], [('1', '3.5')]):
    for feature, (vmin, vmax), (vmin2, vmax2), (vmin3, vmax3) in zip(['thickness', 'curv', 'sulc'],
                                     [('1', '3.5'), ('-0.5', '0.25'), ('-13', '13'),],
                                     [('0', '0.35'), ('0', '0.05'), ('0', '1.3'),],
                                     [('0', '0.35'), ('0', '0.05'), ('0', '1.3')],):
        # if feature in ['thickness', 'curv']:
        #     continue

        method = 'DeepPrep'
        src_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/MSC/derivatives/deepprep/Recon'
        screenshot_result_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/{feature}/feature_map_image_{method}'
        # feature_screenshot(src_dir, screenshot_result_dir, feature=feature, vmin=vmin, vmax=vmax)

        method = 'FreeSurfer'
        src_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/MSC/derivatives/FreeSurfer'
        screenshot_result_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/{feature}/feature_map_image_{method}'
        # feature_screenshot(src_dir, screenshot_result_dir, feature=feature, vmin=vmin, vmax=vmax)

        # ############# cat screenshot
        # concat_screenshot(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/{feature}')

        # # # ############# cal DICE, save to csv
        # # 将DeepPrep的Recon结果和FreeSurfer的Recon结果link到一个目录下（mris_surf2surf需要）
        deepprep_recon_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/MSC/derivatives/deepprep/Recon')
        freesurfer_recon_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/MSC/derivatives/FreeSurfer')
        concat_dp_and_fs_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_dir_concat_DeepPrep_and_FreeSurfer')
        # ln_subject(deepprep_recon_dir, freesurfer_recon_dir, concat_dp_and_fs_dir)

        ############# 将结果投影到fs6
        for hemi in ['lh', 'rh']:
            # ## 投影到fs6
            native_interp_fsaverage6_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/'
                                                f'recon_interp_fsaverage6')
            # project_fsaverage6(concat_dp_and_fs_dir, native_interp_fsaverage6_dir, feature, hemi=hemi)

            # # ## 在fs6 space计算个体水平
            individual_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_individual_fsaverage6')
            # cal_individual_fsaverage6(interp_dir=native_interp_fsaverage6_dir, individual_dir=individual_dir, feature=feature, hemi=hemi)

            # # ## 在fs6 space计算稳定性
            stability_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_stability_fsaverage6')
            # cal_stability_fsaverage6(individual_dir, stability_dir, feature=feature, hemi=hemi)

            # # ## 在fs6 space计算组水平
            group_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_group_fsaverage6')
            # cal_group_fsaverage6(interp_dir=native_interp_fsaverage6_dir, group_dir=group_dir, feature=feature, hemi=hemi)

            # 计算差异组水平显著性p_value
            fs6_deepprep_freesurfer = '/mnt/ngshare/Data_Mirror/FreeSurferFastCSR/Validation/MSC/v1_feature/recon_interp_fsaverage6'
            output_dir = '/mnt/ngshare/Data_Mirror/FreeSurferFastCSR/Validation/MSC/v1_feature/recon_interp_fsaverage6_pvalue'

            # cal_group_difference(fs6_deepprep_freesurfer, output_dir, feature=feature, hemi=hemi)


        # ## individual_screenshot
        for project in ['DeepPrep', 'FreeSurfer']:
            individual_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_individual_fsaverage6/{project}')
            individual_screenshot_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_individual_fsaverage6_screenshot/{project}')
            # group_screenshot(individual_dir, individual_screenshot_dir, feature=feature, vmin1=vmin, vmax1=vmax, vmin2=vmin2,vmax2=vmax2)

        # TODO 个体水平的拼接

        ## group_screenshot
        group_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_group_fsaverage6')
        group_screenshot_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_group_fsaverage6_screenshot')
        # group_screenshot(group_dir, group_screenshot_dir, feature=feature, vmin1=vmin, vmax1=vmax, vmin2=vmin2, vmax2=vmax2)

        group_screenshot_concat_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_group_fsaverage6_screenshot_concat')
        # concat_group_screenshot(group_screenshot_dir, group_screenshot_concat_dir, feature=feature)

        # ## stability_screenshot
        stability_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_stability_fsaverage6')
        stability_screenshot_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_stability_fsaverage6_screenshot')
        # stability_screenshot(stability_dir, stability_screenshot_dir, feature=feature, vmin=vmin3, vmax=vmax3)

        stability_screenshot_concat_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/Validation/MSC/v1_feature/recon_stability_fsaverage6_screenshot_concat')
        # concat_stability_screenshot(stability_screenshot_dir, stability_screenshot_concat_dir, feature=feature)

        ## p_value screenshot
        p_value_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastCSR/Validation/MSC/v1_feature/recon_interp_fsaverage6_pvalue')
        p_value_dir_screenshot_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastCSR/Validation/MSC/v1_feature/recon_interp_fsaverage6_pvalue_screenshot')
        # p_value_screenshot(p_value_dir, p_value_dir_screenshot_dir, feature=feature)
        # print("DONE")

        p_value_dir_screenshot_concat_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastCSR/Validation/MSC/v1_feature/recon_interp_fsaverage6_pvalue_screenshot')
        # concat_pvalue_screenshot(p_value_dir_screenshot_concat_dir, feature=feature)

    # break
