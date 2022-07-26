import os
import time

from image import concat_horizontal, concat_vertical
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path


def set_environ():
    # FreeSurfer
    value = os.environ.get('FREESURFER_HOME')
    if value is None:
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer710'
        os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer710/subjects'
        os.environ['PATH'] = '/usr/local/freesurfer710/bin:' + os.environ['PATH']


def image_screenshot(surf_file, overlay_file, save_path, min, max):
    cmd = f'freeview --viewsize 600 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color=colorwheel,inverse -cam dolly 1.4 azimuth 0 -ss {save_path}'
    # cmd = f'freeview --viewsize 600 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color=colorwheel,inverse -colorscale -cam dolly 1.4 azimuth 0 -ss {save_path}'
    # cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f "{surf_file}":annotation="{overlay_file}" -cam dolly 1.4 azimuth 0 -ss {save_path}'
    print(cmd)
    os.system(cmd)


def image_screenshot_azimuth_180(surf_file, overlay_file, save_path, min, max):
    cmd = f'freeview --viewsize 600 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color=colorwheel,inverse -cam dolly 1.4 azimuth 180 -ss {save_path}'
    # cmd = f'freeview --viewsize 600 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:overlay={overlay_file}:overlay_threshold={min},{max}:overlay_color=colorwheel,inverse -colorscale -cam dolly 1.4 azimuth 180 -ss {save_path}'
    # cmd = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f "{surf_file}":annotation="{overlay_file}" -cam dolly 1.4 azimuth 180 -ss {save_path}'
    print(cmd)
    os.system(cmd)


def feature_screenshot(recon_dir, out_dir, feature='thickness', vmin='', vmax=''):
    """
    读取FreeSurfer格式的目录，并对aparc结构进行截图
    需要 surf/?h.pial 和 label/?h.aparc.annot
    """

    subject_list = os.listdir(recon_dir)
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
            image_screenshot(surf_file, overlay_file, save_file, min=vmin, max=vmax)
            save_file = os.path.join(out_dir, f'{subject}_{hemi}_medial.png')
            image_screenshot_azimuth_180(surf_file, overlay_file, save_file, min=vmin, max=vmax)


def concat_screenshot(screenshot_dir: str):
    """
    拼接DeepPrep和FreeSurfer的分区结果图像
    """

    filenames = os.listdir(os.path.join(screenshot_dir, 'feature_map_image_FreeSurfer'))
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
    concat_dir.mkdir(exist_ok=True)
    for subject_path in deepprep_dir.iterdir():
        print(subject_path)
        target = concat_dir / f'DeepPrep__{subject_path.name}'
        if not target.exists():
            target.symlink_to(subject_path, target_is_directory=True)

    for subject_path in freesurfer_dir.iterdir():
        print(subject_path)
        target = concat_dir / f'FreeSurfer__{subject_path.name}'
        if not target.exists():
            target.symlink_to(subject_path, target_is_directory=True)


def project_fsaverage6(recon_dir: Path, output_dir: Path, feature='thickness', hemi='lh'):
    target = recon_dir / 'fsaverage6'
    if not target.exists():
        target.symlink_to('/usr/local/freesurfer600/subjects/fsaverage6', target_is_directory=True)

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
            print(f'shell_run : {cmd}')
            os.system(cmd)
            print('*' * 40)


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
    for subject in subject_list:
        if not 'sub' in subject:
            continue
        # if 'ses' in subject:
        #     continue

        for hemi in ['lh', 'rh']:
            surf_file = os.path.join('/usr/local/freesurfer600/subjects/fsaverage6', 'surf', f'{hemi}.pial')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for stats_type, (vmin, vmax) in zip(['mean', 'std'], [(vmin1, vmax1), (vmin2, vmax2)]):

                overlay_file = os.path.join(recon_dir, subject, 'surf', f'{hemi}.{stats_type}.{feature}')
                save_file = os.path.join(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_lateral.png')
                if not os.path.exists(save_file):
                    image_screenshot(surf_file, overlay_file, save_file, min=vmin, max=vmax)
                save_file = os.path.join(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_medial.png')
                if not os.path.exists(save_file):
                    image_screenshot_azimuth_180(surf_file, overlay_file, save_file, min=vmin, max=vmax)


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
    for subject in subject_list:

        for hemi in ['lh', 'rh']:
            surf_file = os.path.join('/usr/local/freesurfer600/subjects/fsaverage6', 'surf', f'{hemi}.pial')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            overlay_file = os.path.join(recon_dir, subject, 'surf', f'{hemi}.{feature}')
            save_file = os.path.join(out_dir, f'{subject}_{feature}_{hemi}_lateral.png')
            image_screenshot(surf_file, overlay_file, save_file, min=vmin, max=vmax)
            save_file = os.path.join(out_dir, f'{subject}_{feature}_{hemi}_medial.png')
            image_screenshot_azimuth_180(surf_file, overlay_file, save_file, min=vmin, max=vmax)


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
    for subject in subject_list:
        for hemi in ['lh', 'rh']:
            surf_file = os.path.join('/usr/local/freesurfer600/subjects/fsaverage6', 'surf', f'{hemi}.pial')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for stats_type, (vmin, vmax) in zip(['mean', 'std'], [(vmin1, vmax1), (vmin2, vmax2)]):

                overlay_file = os.path.join(recon_dir, subject, 'surf', f'{hemi}.{stats_type}.{feature}')
                save_file = os.path.join(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_lateral.png')
                image_screenshot(surf_file, overlay_file, save_file, min=vmin, max=vmax)
                save_file = os.path.join(out_dir, f'{subject}_{feature}_{stats_type}_{hemi}_medial.png')
                image_screenshot_azimuth_180(surf_file, overlay_file, save_file, min=vmin, max=vmax)


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


if __name__ == '__main__':
    set_environ()

    # DeepPrep和FreeSurfer的结果计算DICE

    # ############# 分区截图
    # for feature in ['thickness', 'sulc', 'curv']:
    # for feature, (vmin, vmax) in zip(['thickness'], [('1', '3.5')]):
    for feature, (vmin, vmax), (vmin2, vmax2), (vmin3, vmax3) in zip(['thickness', 'curv', 'sulc'],
                                     [('1', '3.5'), ('-0.5', '0.25'), ('-13', '13'),],
                                     [('0', '0.35'), ('0', '0.05'), ('0', '1.3'),],
                                     [('0', '0.35'), ('0', '0.05'), ('0', '1.3')],):
        # method = 'DeepPrep'
        # src_dir = f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon'
        # screenshot_result_dir = f'/mnt/ngshare/DeepPrep/Validation/v1_feature/{feature}/feature_map_image_{method}'
        # feature_screenshot(src_dir, screenshot_result_dir, feature=feature, vmin=vmin, vmax=vmax)
        #
        # method = 'FreeSurfer'
        # src_dir = f'/mnt/ngshare/DeepPrep/MSC/derivatives/FreeSurfer'
        # screenshot_result_dir = f'/mnt/ngshare/DeepPrep/Validation/v1_feature/{feature}/feature_map_image_{method}'
        # feature_screenshot(src_dir, screenshot_result_dir, feature=feature, vmin=vmin, vmax=vmax)
        #
        # # ############# cat screenshot
        # concat_screenshot(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/{feature}')

        # # ############# cal DICE, save to csv
        # # 将DeepPrep的Recon结果和FreeSurfer的Recon结果link到一个目录下（mris_surf2surf需要）
        # deepprep_recon_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
        # freesurfer_recon_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/FreeSurfer')
        # concat_dp_and_fs_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_dir_concat_DeepPrep_and_FreeSurfer')
        # ln_subject(deepprep_recon_dir, freesurfer_recon_dir, concat_dp_and_fs_dir)

        # ############# 将结果投影到fs6
        for hemi in ['lh', 'rh']:
            # 投影到fs6
            native_interp_fsaverage6_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/'
                                                f'recon_interp_fsaverage6')
            # project_fsaverage6(concat_dp_and_fs_dir, native_interp_fsaverage6_dir, feature, hemi=hemi)

            # # 在fs6 space计算组水平
            # individual_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_individual_fsaverage6')
            # cal_individual_fsaverage6(interp_dir=native_interp_fsaverage6_dir, individual_dir=individual_dir, feature=feature, hemi=hemi)
            #
            # # 在fs6 space计算稳定性
            # stability_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_stability_fsaverage6')
            # cal_stability_fsaverage6(individual_dir, stability_dir, feature=feature, hemi=hemi)

            # # 在fs6 space计算组水平
            # group_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_group_fsaverage6')
            # cal_group_fsaverage6(interp_dir=native_interp_fsaverage6_dir, group_dir=group_dir, feature=feature, hemi=hemi)

        # individual_screenshot
        for project in ['DeepPrep', 'FreeSurfer']:
            individual_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_individual_fsaverage6/{project}')
            individual_screenshot_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_individual_fsaverage6_screenshot/{project}')
            group_screenshot(individual_dir, individual_screenshot_dir, feature=feature, vmin1=vmin, vmax1=vmax, vmin2=vmin2,vmax2=vmax2)

        # group_screenshot
        # group_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_group_fsaverage6')
        # group_screenshot_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_group_fsaverage6_screenshot')
        # group_screenshot(group_dir, group_screenshot_dir, feature=feature, vmin1=vmin, vmax1=vmax, vmin2=vmin2, vmax2=vmax2)

        # group_screenshot_concat_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_group_fsaverage6_screenshot_concat')
        # concat_group_screenshot(group_screenshot_dir, group_screenshot_concat_dir, feature=feature)

        # # stability_screenshot
        # stability_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_stability_fsaverage6')
        # stability_screenshot_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_stability_fsaverage6_screenshot')
        # # stability_screenshot(stability_dir, stability_screenshot_dir, feature=feature, vmin=vmin3, vmax=vmax3)
        #
        # stability_screenshot_concat_dir = Path(f'/mnt/ngshare/DeepPrep/Validation/v1_feature/recon_stability_fsaverage6_screenshot_concat')
        # # concat_stability_screenshot(stability_screenshot_dir, stability_screenshot_concat_dir, feature=feature)

