import os
from image import concat_horizontal, concat_vertical
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool


def set_environ():
    # FreeSurfer
    value = os.environ.get('FREESURFER_HOME')
    if value is None:
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer710'
        os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer710/subjects'
        os.environ['PATH'] = '/usr/local/freesurfer710/bin:' + os.environ['PATH']


def image_screenshot(surf_file, overlay_file, save_path, min, max):
    # cmd_medial = f'freeview --viewsize 600 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:annotation={overlay_file}:overlay_threshold={min},{max} -cam dolly 1.4 azimuth 0 -ss {save_path}'
    cmd_medial = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f "{surf_file}":annotation="{overlay_file}" -cam dolly 1.4 azimuth 0 -ss {save_path}'
    os.system(cmd_medial)


def image_screenshot_azimuth_180(surf_file, overlay_file, save_path, min, max):
    # cmd_medial = f'freeview --viewsize 600 600 -viewport 3D  -layout 1 -hide-3d-slices -f {surf_file}:annotation={overlay_file}:overlay_threshold={min},{max} -cam dolly 1.4 azimuth 180 -ss {save_path}'
    cmd_medial = f'freeview --viewsize 800 600 -viewport 3D  -layout 1 -hide-3d-slices -f "{surf_file}":annotation="{overlay_file}" -cam dolly 1.4 azimuth 180 -ss {save_path}'
    os.system(cmd_medial)


def aparc_screenshot(recon_dir, out_dir):
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

            overlay_file = os.path.join(recon_dir, subject, 'label', f'{hemi}.aparc.annot')
            save_file = os.path.join(out_dir, f'{subject}_{hemi}_lateral.png')
            # image_screenshot(surf_file, overlay_file, save_file, min='', max='')
            if not os.path.exists(save_file):
                args_list1.append([surf_file, overlay_file, save_file, '', ''])
            save_file = os.path.join(out_dir, f'{subject}_{hemi}_medial.png')
            # image_screenshot_azimuth_180(surf_file, overlay_file, save_file, min='', max='')
            if not os.path.exists(save_file):
                args_list2.append([surf_file, overlay_file, save_file, '', ''])
    pool = Pool(10)
    pool.starmap(image_screenshot, args_list1)
    pool.starmap(image_screenshot_azimuth_180, args_list2)
    pool.close()
    pool.join()


def concat_screenshot(screenshot_dir: str):
    """
    拼接DeepPrep和FreeSurfer的分区结果图像
    """

    filenames = os.listdir(os.path.join(screenshot_dir, 'aparc_map_image_DeepPrep'))
    subjects = set(['_'.join(i.split('_')[0:-2]) for i in filenames])

    for subject in subjects:
        f1 = os.path.join(screenshot_dir, 'aparc_map_image_DeepPrep', f'{subject}_lh_lateral.png')
        f2 = os.path.join(screenshot_dir, 'aparc_map_image_DeepPrep', f'{subject}_rh_lateral.png')
        f3 = os.path.join(screenshot_dir, 'aparc_map_image_DeepPrep', f'{subject}_lh_medial.png')
        f4 = os.path.join(screenshot_dir, 'aparc_map_image_DeepPrep', f'{subject}_rh_medial.png')
        f5 = os.path.join(screenshot_dir, 'aparc_map_image_FreeSurfer', f'{subject}_lh_lateral.png')
        f6 = os.path.join(screenshot_dir, 'aparc_map_image_FreeSurfer', f'{subject}_rh_lateral.png')
        f7 = os.path.join(screenshot_dir, 'aparc_map_image_FreeSurfer', f'{subject}_lh_medial.png')
        f8 = os.path.join(screenshot_dir, 'aparc_map_image_FreeSurfer', f'{subject}_rh_medial.png')

        # img_h1 = concat_vertical([f1, f2, f5, f6])
        # img_h2 = concat_vertical([f3, f4, f7, f8])
        img_h1 = concat_vertical([f1, f3, f4, f2])
        img_h2 = concat_vertical([f5, f7, f8, f6])
        save_path = os.path.join(screenshot_dir, 'aparc_map_image_concat')
        save_file = os.path.join(save_path, f'{subject}.png')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = concat_horizontal([img_h1, img_h2], save_file)
        # img = cv2.resize(img, (1568, 718))
        # cv2.imshow('imshow', img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("imshow")
        # break


def aparc_dice(ref_labels, mov_labels, names, drop_num=35):
    if drop_num is not None:
        index = ref_labels != drop_num
        ref_labels = ref_labels[index]
        mov_labels = mov_labels[index]
    label_ids = np.unique(ref_labels)
    # label_ids = np.delete(label_ids, [34])  # 去掉‘insula’分区进行比较
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


def dice(concat_dir: Path, interp_dir: Path, dice_out_csv: Path, hemi: str):
    dp_dict = dict()
    fs_dict = dict()
    for subject in concat_dir.iterdir():
        if not 'sub' in subject.name:
            continue
        if 'DeepPrep' in subject.name:
            dp_dict[subject.name.split('__')[1]] = subject
        else:
            fs_dict[subject.name.split('__')[1]] = subject

    data_list = list()
    for subject_id in dp_dict.keys():

        os.environ['SUBJECTS_DIR'] = str(concat_dir)

        deepprep_subject = dp_dict[subject_id]
        if subject_id in fs_dict:
            freesurfer_subject = fs_dict[subject_id]
        else:
            continue

        # for hemi in ['lh', 'rh']:
        data_dict = dict()
        data_dict['subj'] = f'{subject_id}_{hemi}'

        annot1 = deepprep_subject / 'label' / f'{hemi}.aparc.annot'
        annot2 = freesurfer_subject / 'label' / f'{hemi}.aparc.annot'

        interp_path = interp_dir / subject_id / 'label'
        interp_path.mkdir(parents=True, exist_ok=True)

        tval = interp_path / f'{hemi}.aparc.annot'

        if not tval.exists():
            cmd = f'mri_surf2surf --srcsubject DeepPrep__{subject_id} --sval-annot {annot1} ' \
                  f'--trgsubject FreeSurfer__{subject_id} --tval {tval} --hemi {hemi}'
            os.system(cmd)

        annot1_data, _, names1 = nib.freesurfer.read_annot(str(tval))
        annot2_data, _, names2 = nib.freesurfer.read_annot(str(annot2))

        dice_result = aparc_dice(annot1_data, annot2_data, names1, drop_num=None)
        data_dict.update(dice_result)
        data_list.append(data_dict)

    dice_mean = np.mean([item['dice_mean'] for item in data_list])
    dice_whole = np.mean([item['dice_whole'] for item in data_list])
    items = {
        'subj': 'mean',
        'dice_mean': dice_mean,
        'dice_whole': dice_whole
    }
    data_list.append(items)
    df = pd.DataFrame(data_list)
    df.to_csv(dice_out_csv, index=False)


if __name__ == '__main__':
    set_environ()

    # DeepPrep和FreeSurfer的结果计算DICE

    # 分区截图
    method = 'DeepPrep'
    src_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/MSC/derivatives/deepprep/Recon'
    screenshot_result_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/Validation/MSC/v1_aparc/aparc_map_image_{method}'
    aparc_screenshot(src_dir, screenshot_result_dir)

    method = 'FreeSurfer'
    src_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/MSC/derivatives/FreeSurfer'
    screenshot_result_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/Validation/MSC/v1_aparc/aparc_map_image_{method}'
    aparc_screenshot(src_dir, screenshot_result_dir)

    # cat screenshot
    concat_screenshot(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/Validation/MSC/v1_aparc')

    # # cal DICE
    deepprep_recon_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/MSC/derivatives/deepprep/Recon')
    freesurfer_recon_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/MSC/derivatives/FreeSurfer')
    deepprep_interp_freesurfer = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/Validation/MSC/v1_aparc/recon_aparc_DeepPrep_interp_FreeSufer')
    concat_dp_and_fs_dir = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/Validation/MSC/v1_aparc/recon_dir_concat_DeepPrep_and_FreeSurfer')
    ln_subject(deepprep_recon_dir, freesurfer_recon_dir, concat_dp_and_fs_dir)
    dice_csv = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/Validation/MSC/v1_aparc/DICE_DeepPrep_FreeSurfer/dice_lh.csv')
    if not dice_csv.parent.exists():
        dice_csv.parent.mkdir(parents=True, exist_ok=True)
    dice(concat_dp_and_fs_dir, deepprep_interp_freesurfer, dice_csv, 'lh')
    dice_csv = Path(f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/Validation/MSC/v1_aparc/DICE_DeepPrep_FreeSurfer/dice_rh.csv')
    dice(concat_dp_and_fs_dir, deepprep_interp_freesurfer, dice_csv, 'rh')
