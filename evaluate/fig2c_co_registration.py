import os
import sh
import cv2
import numpy as np
import ants
from pathlib import Path


def plot_voxel_enhance_brats(arr, arr_mask=None, out_file: str = None, plot_type=None):  # arr: zxy
    arr *= (255.0 / arr.max())

    rows = cols = int(round(np.sqrt(arr.shape[0])))
    img_height = arr.shape[1]
    img_width = arr.shape[2]
    # assert img_width == img_height
    res_img = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)
    if arr_mask is not None:
        res_mask_img = np.zeros(
            (rows * img_height, cols * img_width), dtype=np.uint8)
    else:
        res_mask_img = None
    for row in range(rows):
        for col in range(cols):
            if (row * cols + col) >= arr.shape[0]:
                continue
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = arr[row * cols + col]
            if res_mask_img is not None:
                res_mask_img[target_y:target_y + img_height, target_x:target_x + img_width] = arr_mask[row * cols + col]
    if arr_mask is None or plot_type == 'origin':
        cv2.imwrite(out_file, res_img)
    else:
        if plot_type == 'contours':  # 添加掩码轮廓图
            res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2BGR)
            for thres, color in zip([1], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):  # 遍历标签，三种标签 zip([1,3,2]。
                res_mask_color = np.zeros_like(res_mask_img, dtype=np.uint8)
                res_mask_color[res_mask_img == thres] = 2
                ret, binary = cv2.threshold(res_mask_color, 1, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(res_img, contours, -1, color, 1)
            cv2.imwrite(out_file, res_img)
        elif plot_type == 'fillpoly':  # 添加掩码填充图
            res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2BGR)
            for thres, color in zip([1], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                res_mask_color = np.zeros_like(res_mask_img, dtype=np.uint8)
                res_mask_color[res_mask_img == thres] = 2
                ret, binary = cv2.threshold(res_mask_color, 1, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                layer = np.zeros_like(res_img, dtype=np.uint8)
                cv2.fillPoly(layer, contours, color)
                res_img = cv2.addWeighted(res_img, 1, layer, 0.3, 0)
            cv2.imwrite(out_file, res_img)
        else:
            raise RuntimeError(f'plot_type is error: {plot_type}')


def plot_bold_native_t1(bold_preprocess_dir, subjects_dir):

    for subject_id in os.listdir(bold_preprocess_dir):
        bold_files = glob(f'{bold_preprocess_dir}/{subject_id}/**/*_native_t1_2mm.nii.gz', recursive=True)
        if not bold_files:
            continue
        bold_file = bold_files[0]

        seg_file_1mm = subjects_dir / subject_id / 'mri/aparc+aseg.mgz'
        if not seg_file_1mm.exists():
            continue
        seg_file_2mm = bold_file.replace('_native_t1_2mm.nii.gz', '_aparc+aseg_T1_2mm.mgz')
        if not os.path.exists(seg_file_2mm):
            sh.mri_convert('-ds', 2, 2, 2,
                           '--resample_type', 'nearest',
                           '-i', seg_file_1mm,
                           '-o', seg_file_2mm)

        background_data = ants.image_read(str(bold_file))[:, :, :, 0]
        seg_data = ants.image_read(str(seg_file_2mm)).numpy()
        mask_data = np.zeros_like(seg_data, dtype='uint8')
        mask_data[seg_data == 41] = 1
        mask_data[seg_data == 2] = 1
        contours_img_file = bold_file + '.wm_contours_x.png'
        plot_voxel_enhance_brats(background_data, mask_data, out_file=contours_img_file, plot_type='contours')

        background_data_y = background_data.swapaxes(0, 2)
        mask_data_y = mask_data.swapaxes(0, 2)
        contours_img_file = bold_file + '.wm_contours_y.png'
        plot_voxel_enhance_brats(background_data_y, mask_data_y, out_file=contours_img_file, plot_type='contours')

        background_data_z = background_data.swapaxes(0, 1)
        mask_data_z = mask_data.swapaxes(0, 1)
        background_data_z = np.rot90(background_data_z, axes=(1, 2))
        mask_data_z = np.rot90(mask_data_z, axes=(1, 2))
        contours_img_file = bold_file + '.wm_contours_z.png'
        plot_voxel_enhance_brats(background_data_z, mask_data_z, out_file=contours_img_file, plot_type='contours')


def plot_bold_mni152(bold_preprocess_dir, subject_dir_mni152):

    for subject_id in os.listdir(bold_preprocess_dir):
        bold_files = glob(f'{bold_preprocess_dir}/{subject_id}/**/*_MNI2mm.nii.gz', recursive=True)
        if not bold_files:
            continue
        bold_file = bold_files[0]

        seg_file_1mm = subject_dir_mni152 / 'mri/aparc+aseg.mgz'
        if not seg_file_1mm.exists():
            continue
        seg_file_2mm = bold_file.replace('_MNI2mm.nii.gz', '_aparc+aseg_MNI152_2mm.mgz')
        if not os.path.exists(seg_file_2mm):
            src_img = ants.image_read(str(seg_file_1mm))
            target_img = ants.image_read('MNI152_T1_2mm_brain.nii.gz')
            seg_2mm_img = ants.resample_image_to_target(src_img, target_img, interp_type='nearestNeighbor')
            ants.image_write(seg_2mm_img, seg_file_2mm)

        background_data = ants.image_read(str(bold_file))[:, :, :, 0]
        seg_data = ants.image_read(str(seg_file_2mm)).numpy()
        mask_data = np.zeros_like(seg_data, dtype='uint8')
        mask_data[seg_data == 41] = 1
        mask_data[seg_data == 2] = 1
        background_data_x = np.rot90(background_data, axes=(1, 2))
        mask_data_x = np.rot90(mask_data, axes=(1, 2))
        contours_img_file = bold_file + '.wm_contours_x.png'
        plot_voxel_enhance_brats(background_data_x, mask_data_x, out_file=contours_img_file, plot_type='contours')

        background_data_y = background_data.swapaxes(0, 2)
        mask_data_y = mask_data.swapaxes(0, 2)
        contours_img_file = bold_file + '.wm_contours_y.png'
        plot_voxel_enhance_brats(background_data_y, mask_data_y, out_file=contours_img_file, plot_type='contours')

        background_data_z = background_data.swapaxes(0, 1)
        mask_data_z = mask_data.swapaxes(0, 1)
        background_data_z = np.rot90(background_data_z, axes=(1, 2))
        mask_data_z = np.rot90(mask_data_z, axes=(1, 2))
        contours_img_file = bold_file + '.wm_contours_z.png'
        plot_voxel_enhance_brats(background_data_z, mask_data_z, out_file=contours_img_file, plot_type='contours')


if __name__ == '__main__':
    from glob import glob
    from interface.run import set_envrion
    set_envrion()

    bold_preprocess_dir = Path('/mnt/ngshare/DeepPrep_workflow_test/UKB_BoldPreprocess')
    subjects_dir = Path('/mnt/ngshare/DeepPrep_workflow_test/UKB_Recon')
    plot_bold_native_t1(bold_preprocess_dir, subjects_dir)

    bold_preprocess_dir = Path('/mnt/ngshare/DeepPrep_workflow_test/UKB_BoldPreprocess')
    subject_dir_mni152 = Path('/mnt/ngshare/DeepPrep/MNI152_T1_1mm')
    plot_bold_mni152(bold_preprocess_dir, subject_dir_mni152)
