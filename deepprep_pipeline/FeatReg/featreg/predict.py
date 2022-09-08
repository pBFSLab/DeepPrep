import os
import argparse

import torch
from torch.utils.data import DataLoader

from utils.interp import interp_sulc_curv_barycentric

from dataset import SphericalDataset, get_dataset_item
from utils.rotate_matrix import apply_rigid_rotate_matrix, apply_norigid_rotate_matrix, save_rotate_matrix
from utils.negative_area_triangle import single_remove_negative_area

abspath_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(abspath_dir))
MODEL = {
    'lh_rigid': f'{root_dir}/model/FeatReg/lh_Rigid.model',
    'rh_rigid': f'{root_dir}/model/FeatReg/rh_Rigid.model',
    'lh_norigid': f'{root_dir}/model/FeatReg/lh_NoRigid.model',
    'rh_norigid': f'{root_dir}/model/FeatReg/rh_NoRigid.model',
}


def rigid_interp(surf_dir, tmp_dir, fsaverage_dir, sub_id, hemispheres=None, device='cuda'):
    surf_dir_interp = os.path.join(tmp_dir, sub_id, 'surf')
    if not os.path.exists(surf_dir_interp):
        os.makedirs(surf_dir_interp)

    if hemispheres is None:
        hemispheres = ['lh', 'rh']
    if isinstance(hemispheres, str):
        hemispheres = [hemispheres]

    for hemisphere in hemispheres:
        sphere_fixed_file = os.path.join(fsaverage_dir, 'surf', f'{hemisphere}.sphere')

        sulc_moving_file = os.path.join(surf_dir, f'{hemisphere}.sulc')
        curv_moving_file = os.path.join(surf_dir, f'{hemisphere}.curv')
        sphere_moving_file = os.path.join(surf_dir, f'{hemisphere}.sphere')

        sulc_moving_interp_file = os.path.join(surf_dir_interp, f'{hemisphere}.orig.interp_fs.sulc')
        curv_moving_interp_file = os.path.join(surf_dir_interp, f'{hemisphere}.orig.interp_fs.curv')
        sphere_moving_interp_file = os.path.join(surf_dir_interp, f'{hemisphere}.orig.interp_fs.sphere')

        if not os.path.exists(sulc_moving_file):
            continue

        if not os.path.exists(sulc_moving_interp_file):
            interp_sulc_curv_barycentric(
                sulc_moving_file, curv_moving_file, sphere_moving_file,
                sphere_fixed_file,
                sulc_moving_interp_file, curv_moving_interp_file, sphere_moving_interp_file, device=device)
            print(f'interp: >>> {sulc_moving_interp_file}')
            print(f'interp: >>> {curv_moving_interp_file}')
            print(f'interp: >>> {sphere_moving_interp_file}')


def rigid_predict(surf_dir, tmp_dir, fsaverage_dir, sub_id, hemispheres=None, device='cuda'):
    surf_dir_interp = os.path.join(tmp_dir)

    if hemispheres is None:
        hemispheres = ['lh', 'rh']
    if isinstance(hemispheres, str):
        hemispheres = [hemispheres]
    for hemisphere in hemispheres:
        # 2. load model
        checkpoint = torch.load(MODEL[f'{hemisphere}_rigid'])
        model = checkpoint['model']
        model.to(device)
        model.eval()

        data_input, moving_data, fixed_data, xyz_moving, xyz_fixed = get_dataset_item(
            surf_dir_interp, fsaverage_dir, sub_id, hemisphere, 'sucu', device=device)

        pred = model(data_input, xyz_moving)
        moved_xyz, rotate_matrix = pred[:2]

        # 保存旋转矩阵文件
        # 原始moving应用旋转矩阵
        rotate_matrix_rigid_file = os.path.join(surf_dir_interp, sub_id,
                                                'surf', f'{hemisphere}.sphere.rigid_rotate_metrix.npz')
        save_rotate_matrix(rotate_matrix.detach().cpu().numpy(), rotate_matrix_rigid_file)

        # 03 原始结果应用旋转矩阵：得到rigid
        sphere_moving_file = os.path.join(surf_dir,  f'{hemisphere}.sphere')
        sphere_moving_rigid_file = os.path.join(surf_dir_interp, sub_id, 'surf', f'{hemisphere}.rigid.sphere')
        apply_rigid_rotate_matrix(sphere_moving_file, rotate_matrix_rigid_file, sphere_moving_rigid_file)


def norigid_interp(surf_dir, tmp_dir, fsaverage_dir, sub_id, hemispheres=None, device='cuda'):
    """
    预处理：将native空间插值到fsaverage空间
    """
    surf_dir_interp = os.path.join(tmp_dir, sub_id, 'surf')
    if not os.path.exists(surf_dir_interp):
        os.makedirs(surf_dir_interp)

    if hemispheres is None:
        hemispheres = ['lh', 'rh']
    if isinstance(hemispheres, str):
        hemispheres = [hemispheres]

    for hemisphere in hemispheres:
        sphere_fixed_file = os.path.join(fsaverage_dir, 'surf', f'{hemisphere}.sphere')

        sulc_moving_file = os.path.join(surf_dir, f'{hemisphere}.sulc')
        curv_moving_file = os.path.join(surf_dir, f'{hemisphere}.curv')
        sphere_moving_file = os.path.join(surf_dir_interp, f'{hemisphere}.rigid.sphere')  # 跑完刚性配准以后有这个文件
        if not os.path.exists(sphere_moving_file):
            continue

        sulc_moving_interp_file = os.path.join(surf_dir_interp, f'{hemisphere}.rigid.interp_fs.sulc')
        curv_moving_interp_file = os.path.join(surf_dir_interp, f'{hemisphere}.rigid.interp_fs.curv')
        sphere_moving_interp_file = os.path.join(surf_dir_interp, f'{hemisphere}.rigid.interp_fs.sphere')
        if not os.path.exists(sulc_moving_interp_file):
            interp_sulc_curv_barycentric(sulc_moving_file, curv_moving_file, sphere_moving_file, sphere_fixed_file,
                                         sulc_moving_interp_file, curv_moving_interp_file,
                                         sphere_moving_interp_file, device=device)
            print(f'interp: >>> {sulc_moving_interp_file}')
            print(f'interp: >>> {curv_moving_interp_file}')
            print(f'interp: >>> {sphere_moving_interp_file}')


def norigid_predict(surf_dir, tmp_dir, fsaverage_dir, sub_id, hemispheres=None, device='cuda'):
    if isinstance(hemispheres, str):
        hemispheres = [hemispheres]

    # 获取config_train的配置
    feature = 'sucu'  # 加载的数据类型

    for hemisphere in hemispheres:
        # 2. load model
        checkpoint = torch.load(MODEL[f'{hemisphere}_norigid'])
        model = checkpoint['model']
        model.to(device)
        model.eval()

        model.eval()

        dataset_train = SphericalDataset(None, fsaverage_dir, tmp_dir,
                                         hemisphere, feature=feature, norm_type='SD')
        dataset_train.sub_dirs = [sub_id]
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, num_workers=0)

        for moving_data, fixed_data, xyz_moving, sub_ids, sphere_moving_rigid_files in dataloader_train:
            sub_id = sub_ids[0]

            fixed_data = fixed_data.squeeze(0).to(device)
            moving_data = moving_data.squeeze(0).to(device)
            data_x = torch.cat((moving_data, fixed_data), 1).squeeze()
            xyz_fixed = xyz_moving = xyz_moving.squeeze(0).to(device)  # 因为数据插值到了fsaverage，所以xyz_moving和xyz_fixed相同

            pred = model(data_x, xyz_moving)

            # 保存非刚性配准旋转矩阵
            surf_dir_path = os.path.join(tmp_dir, sub_id, 'surf')
            rotate_matrix_norigid_file = os.path.join(surf_dir_path, f'{hemisphere}.sphere.norigid_rotate_metrix.npz')
            save_rotate_matrix(pred[1].detach().cpu().numpy(), rotate_matrix_norigid_file,
                               xyz=xyz_fixed.detach().cpu().numpy())

            # apply rotate matrix to rigid.interp_fs.sphere
            sphere_rigid_fs_file = os.path.join(surf_dir_path, f'{hemisphere}.rigid.interp_fs.sphere')
            sphere_moved_fs_file = os.path.join(surf_dir_path, f'{hemisphere}.rigid.interp_fs.sphere.reg')
            apply_norigid_rotate_matrix(sphere_rigid_fs_file, rotate_matrix_norigid_file,
                                        sphere_moved_fs_file, device=device)

            # apply rotate matrix to rigid.native.sphere
            sphere_rigid_native_file = os.path.join(surf_dir_path, f'{hemisphere}.rigid.sphere')
            sphere_moved_native_file = os.path.join(surf_dir, f'{hemisphere}.sphere.reg')
            apply_norigid_rotate_matrix(sphere_rigid_native_file, rotate_matrix_norigid_file,
                                        sphere_moved_native_file, device=device)

            # remove negative area triangle of sphere.reg
            single_remove_negative_area(sphere_moved_native_file, sphere_moved_native_file)


def predict(surf_dir, tmp_dir, fsaverage_dir, sub_id, hemispheres=None, device='cuda'):
    rigid_interp(surf_dir, tmp_dir, fsaverage_dir, sub_id, hemispheres, device=device)
    rigid_predict(surf_dir, tmp_dir, fsaverage_dir, sub_id, hemispheres=hemispheres, device=device)
    norigid_interp(surf_dir, tmp_dir, fsaverage_dir, sub_id, hemispheres, device=device)
    norigid_predict(surf_dir, tmp_dir, fsaverage_dir, sub_id, hemispheres=hemispheres, device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', required=True, help='Subject ID for directory inside $SUBJECTS_DIR to be created')
    parser.add_argument('--sd', default=os.environ.get('SUBJECTS_DIR'),
                        help='Output directory $SUBJECTS_DIR (pass via environment or here)')
    parser.add_argument('--fsd', default=os.environ.get('FREESURFER_HOME'),
                        help='Output directory $FREESURFER_HOME (pass via environment or here)')
    parser.add_argument('--hemi', help="which hemisphere")
    parser.add_argument('--verbose', default=False, action='store_true', help="Whether to output detailed log")

    args = parser.parse_args()
    if args.sd is None:
        raise ValueError('Subjects dir need to set via $SUBJECTS_DIR environment or --sd parameter')
    else:
        os.environ['SUBJECTS_DIR'] = args.sd
    subj_dir = os.path.join(args.sd, args.sid)
    if not os.path.exists(subj_dir):
        raise ValueError(f'{subj_dir} is not exists, please check.')
    args_dict = vars(args)

    if args.hemi is None:
        args_dict['hemi'] = ['lh', 'rh']
    else:
        args_dict['hemi'] = [args.hemi]

    if args.fsd is None:
        args_dict['fsd'] = '/usr/local/freesurfer'

    return argparse.Namespace(**args_dict)


if __name__ == '__main__':
    args = parse_args()
    surf_dir = os.path.join(args.sd, args.sid, 'surf')
    tmp_dir = os.path.join(args.sd, args.sid, 'tmp')
    fsaverage_dir = os.path.join(args.fsd, 'subjects', 'fsaverage6')
    hemis = args.hemi

    predict(surf_dir, tmp_dir, fsaverage_dir, args.sid, hemispheres=hemis)
