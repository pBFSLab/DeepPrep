import os
import argparse
from pathlib import Path
from collections import OrderedDict
import nighres
import tempfile
import nibabel as nib
from scipy.ndimage import binary_fill_holes, binary_dilation
import ants
import numpy as np


# 从模型预测得到的levelset文件重建出surface
def levelset2surf(fastcsr_subjects_dir, subj, hemi, suffix):
    print(f'subject: {subj}, hemi: {hemi}')
    os.makedirs(fastcsr_subjects_dir / subj / 'surf', exist_ok=True)
    orig_file = fastcsr_subjects_dir / subj / 'mri' / 'orig.mgz'

    # 计算mask，对模型预测的levelset进行后处理，以增强结果稳健性
    brainmask_file = fastcsr_subjects_dir / subj / 'mri' / 'brainmask.mgz'
    brainmask = ants.image_read((str(brainmask_file)))
    brainmask_np = brainmask.numpy()

    brain_mask = brainmask_np > 0
    brain_mask = binary_fill_holes(brain_mask)

    aseg_file = fastcsr_subjects_dir / subj / 'mri' / 'aseg.presurf.mgz'

    aseg = ants.image_read(str(aseg_file))
    aseg_np = aseg.numpy()

    if hemi == 'lh':
        brainmask_aseg_idx = [2, 4, 5, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31]
    else:
        brainmask_aseg_idx = [41, 43, 44, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63]

    aseg_mask = np.zeros(aseg_np.shape, bool)
    for idx in brainmask_aseg_idx:
        aseg_mask = aseg_mask | (aseg_np == idx)

    aseg_mask = binary_dilation(aseg_mask, iterations=6)
    aseg_mask = binary_fill_holes(aseg_mask)
    mask_np = brain_mask & aseg_mask

    levelset = ants.image_read(str(fastcsr_subjects_dir / subj / 'mri' / f'{hemi}_levelset.nii.gz'))
    levelset_np = levelset.numpy()

    levelset_fix_np = np.ones_like(levelset_np)
    levelset_fix_np[mask_np] = levelset_np[mask_np]
    levelset_fix_np = levelset_fix_np * 3
    levelset_fix = ants.from_numpy(levelset_fix_np, levelset.origin, levelset.spacing, levelset.direction)
    temp_file = tempfile.mktemp(suffix='.nii.gz')
    ants.image_write(levelset_fix, temp_file)

    # 使用nighres进行surface重建
    img = nighres.io.load_volume(temp_file)
    tc_ret = nighres.shape.topology_correction(img, 'signed_distance_function',
                                               propagation='background->object',
                                               connectivity='6/18')
    os.remove(temp_file)
    l2m_ret = nighres.surface.levelset_to_mesh(tc_ret['corrected'], connectivity='6/18')
    print()

    vertices = l2m_ret['result']['points']
    faces = l2m_ret['result']['faces']
    faces = faces[:, [2, 1, 0]]

    norm_nib = nib.load(orig_file)
    vox2ras_tkr = norm_nib.header.get_vox2ras_tkr()
    vox2ras = norm_nib.header.get_vox2ras()
    points_ras = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    points_ras = np.dot(vox2ras_tkr, points_ras.T).T[:, :3]

    # 准备?h.orig.nofix文件写入时的header信息
    volume_info = OrderedDict()
    volume_info['head'] = np.array([2, 0, 20], dtype=np.int32)
    volume_info['valid'] = '1  # volume info valid'
    if hemi == 'lh':
        volume_info['filename'] = '../mri/filled-pretess255.mgz'
    else:
        volume_info['filename'] = '../mri/filled-pretess127.mgz'
    volume_info['volume'] = np.array([256, 256, 256], dtype=np.int64)
    volume_info['voxelsize'] = np.array([1, 1, 1], dtype=np.float64)
    volume_info['xras'] = vox2ras[:3, 0].astype(np.float64)
    volume_info['yras'] = vox2ras[:3, 1].astype(np.float64)
    volume_info['zras'] = vox2ras[:3, 2].astype(np.float64)
    volume_info['cras'] = norm_nib.header.get('Pxyz_c').astype(np.float64)

    # ？h.orig.nofix文件保存路径
    surf_file = fastcsr_subjects_dir / subj / 'surf' / f'{hemi}.{suffix}'
    nib.freesurfer.write_geometry(surf_file, points_ras, faces, volume_info=volume_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fastcsr_subjects_dir', required=True)
    parser.add_argument('--subj', required=True)
    parser.add_argument('--hemi', required=True, choices=['lh', 'rh'])
    parser.add_argument('--suffix', default='orig.nofix', choices=['orig.nofix', 'orig', 'white.preaparc'])
    args = parser.parse_args()
    fastcsr_subjects_dir = Path(args.fastcsr_subjects_dir)
    subj = args.subj
    hemi = args.hemi
    suffix = args.suffix

    levelset2surf(fastcsr_subjects_dir, subj, hemi, suffix)
