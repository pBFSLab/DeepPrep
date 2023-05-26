import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_multi_level_multi_fixed_predict import SphericalDataset
from utils.interp_fine import interp_sulc_curv_barycentric
from utils.rotate_matrix import apply_rotate_matrix
from utils.negative_area_triangle import count_negative_area
from utils.interp import resample_sphere_surface_barycentric, upsample_std_sphere_torch
from utils.auxi_data import get_points_num_by_ico_level


def interp_dir_single(dir_recon: str, dir_rigid: str, dir_fixed: str, ico_level: str, is_rigid=False):
    """
    预处理：将native空间插值到fsaverage空间
    """

    surf_dir_recon = os.path.join(dir_recon, 'surf')
    surf_dir_rigid = os.path.join(dir_rigid, 'surf')
    if not os.path.exists(surf_dir_rigid):
        os.makedirs(surf_dir_rigid)
    for hemisphere in ['lh', 'rh']:
        sphere_fixed_file = os.path.join(dir_fixed, ico_level, 'surf', f'{hemisphere}.sphere')

        # ## 将刚性配准的结果插值到fsaverageN
        sulc_moving_file = os.path.join(surf_dir_recon, f'{hemisphere}.sulc')
        curv_moving_file = os.path.join(surf_dir_recon, f'{hemisphere}.curv')
        if is_rigid:
            data_type = 'orig'
            sphere_moving_file = os.path.join(surf_dir_recon, f'{hemisphere}.sphere')
        else:
            data_type = 'rigid'
            sphere_moving_file = os.path.join(surf_dir_rigid, f'{hemisphere}.rigid.sphere')  # 跑完刚性配准以后有这个文件

        if not os.path.exists(sphere_moving_file):
            continue

        sulc_moving_interp_file = os.path.join(surf_dir_rigid, f'{hemisphere}.{data_type}.interp_{ico_level}.sulc')
        curv_moving_interp_file = os.path.join(surf_dir_rigid, f'{hemisphere}.{data_type}.interp_{ico_level}.curv')
        sphere_moving_interp_file = os.path.join(surf_dir_rigid,
                                                 f'{hemisphere}.{data_type}.interp_{ico_level}.sphere')
        # if not os.path.exists(sulc_moving_interp_file):
        interp_sulc_curv_barycentric(sulc_moving_file, curv_moving_file, sphere_moving_file, sphere_fixed_file,
                                     sulc_moving_interp_file, curv_moving_interp_file,
                                     sphere_moving_interp_file)
        print(f'interp: >>> {sulc_moving_interp_file}')
        print(f'interp: >>> {curv_moving_interp_file}')
        print(f'interp: >>> {sphere_moving_interp_file}')


def save_sphere_reg(config, hemisphere, xyz_moved, euler_angle, dir_recon, dir_rigid, dir_result, device):
    # #################################################################################################### #
    faces = config['face']
    xyzs = config['xyz']

    if config['is_rigid']:
        data_type = 'orig'
        surf_dir_out = os.path.join(dir_rigid, 'surf')
    else:
        data_type = 'rigid'
        surf_dir_out = os.path.join(dir_rigid, 'surf')
        surf_res_out = os.path.join(dir_result, 'surf')

    # # save sphere.reg in f saverage6 not apply rotate matrix
    faces_fs = faces[config["ico_level"]].cpu().numpy()
    if not os.path.exists(surf_dir_out):
        os.makedirs(surf_dir_out)
    sphere_moved_fs_file = os.path.join(surf_dir_out,
                                        f'{hemisphere}.{data_type}.interp_{config["ico_level"]}.sphere.reg')
    nib.freesurfer.write_geometry(sphere_moved_fs_file, xyz_moved.detach().cpu().numpy() * 100, faces_fs)
    print(f'sphere_moved_fs_file >>> {sphere_moved_fs_file}')

    # # apply rotate matrix to rigid.interp_fs5.sphere
    if config["ico_level"] == 'fsaverage3':

        xyz_moved_tmp = upsample_std_sphere_torch(xyz_moved.detach(), norm=True)
        sphere_moved_fs_file = os.path.join(surf_dir_out,
                                            f'{hemisphere}.{data_type}.interp_{config["ico_level"]}.up_fsaverage4.sphere.reg')
        faces_fs4 = faces["fsaverage4"].cpu().numpy()
        nib.freesurfer.write_geometry(sphere_moved_fs_file, xyz_moved_tmp.cpu().numpy() * 100, faces_fs4)
        print(f'sphere_moved_fs_file >>> {sphere_moved_fs_file}')

        xyz_moved_tmp = upsample_std_sphere_torch(xyz_moved_tmp, norm=True)

        sphere_moved_fs_file = os.path.join(surf_dir_out,
                                            f'{hemisphere}.{data_type}.interp_{config["ico_level"]}.up_fsaverage5.sphere.reg')
        faces_fs5 = faces["fsaverage5"].cpu().numpy()
        nib.freesurfer.write_geometry(sphere_moved_fs_file, xyz_moved_tmp.cpu().numpy() * 100, faces_fs5)
        print(f'sphere_moved_fs_file >>> {sphere_moved_fs_file}')

        xyz_moved_tmp = upsample_std_sphere_torch(xyz_moved_tmp, norm=True)

        sphere_moved_fs_file = os.path.join(surf_dir_out,
                                            f'{hemisphere}.{data_type}.interp_{config["ico_level"]}.up_fsaverage6.sphere.reg')
        faces_fs6 = faces["fsaverage6"].cpu().numpy()
        nib.freesurfer.write_geometry(sphere_moved_fs_file, xyz_moved_tmp.cpu().numpy() * 100, faces_fs6)
        print(f'sphere_moved_fs_file >>> {sphere_moved_fs_file}')

    elif config["ico_level"] == 'fsaverage4':

        xyz_moved_tmp = upsample_std_sphere_torch(xyz_moved.detach(), norm=True)
        sphere_moved_fs_file = os.path.join(surf_dir_out,
                                            f'{hemisphere}.{data_type}.interp_{config["ico_level"]}.up_fsaverage5.sphere.reg')
        faces_fs5 = faces["fsaverage5"].cpu().numpy()
        nib.freesurfer.write_geometry(sphere_moved_fs_file, xyz_moved_tmp.cpu().numpy() * 100, faces_fs5)
        print(f'sphere_moved_fs_file >>> {sphere_moved_fs_file}')

        xyz_moved_tmp = upsample_std_sphere_torch(xyz_moved_tmp, norm=True)

        sphere_moved_fs_file = os.path.join(surf_dir_out,
                                            f'{hemisphere}.{data_type}.interp_{config["ico_level"]}.up_fsaverage6.sphere.reg')
        faces_fs6 = faces["fsaverage6"].cpu().numpy()
        nib.freesurfer.write_geometry(sphere_moved_fs_file, xyz_moved_tmp.cpu().numpy() * 100, faces_fs6)
        print(f'sphere_moved_fs_file >>> {sphere_moved_fs_file}')

    elif config["ico_level"] == 'fsaverage5':
        sphere_moved_fs_file = os.path.join(surf_dir_out,
                                            f'{hemisphere}.{data_type}.interp_{config["ico_level"]}.up_fsaverage6.sphere.reg')
        xyz_moved_tmp = upsample_std_sphere_torch(xyz_moved.detach(), norm=True)
        faces_fs6 = faces["fsaverage6"].cpu().numpy()
        nib.freesurfer.write_geometry(sphere_moved_fs_file, xyz_moved_tmp.cpu().numpy() * 100, faces_fs6)
        print(f'sphere_moved_fs_file >>> {sphere_moved_fs_file}')
    else:
        xyz_moved_tmp = xyz_moved

    # if output_native is True:
    xyz_fixed_fs6 = xyzs['fsaverage6'].to(device)

    # interp sphere.reg to native space
    if not config['is_rigid']:
        sphere_rigid_native_file = os.path.join(dir_rigid, 'surf', f'{hemisphere}.rigid.sphere')
        sphere_moved_native_file = os.path.join(surf_res_out, f'{hemisphere}.sphere.reg')

        xyz_native, faces_native = nib.freesurfer.read_geometry(sphere_rigid_native_file)
        xyz_native = torch.from_numpy(xyz_native).float().to(device) / 100
        xyz_moved_native = resample_sphere_surface_barycentric(xyz_fixed_fs6, xyz_native, xyz_moved_tmp,
                                                               device=device)
        xyz_moved_native = xyz_moved_native / torch.norm(xyz_moved_native, dim=1, keepdim=True)

    else:
        sphere_rigid_native_file = os.path.join(dir_recon, 'surf', f'{hemisphere}.sphere')
        sphere_moved_native_file = os.path.join(surf_dir_out, f'{hemisphere}.rigid.sphere')
        xyz_native, faces_native = nib.freesurfer.read_geometry(sphere_rigid_native_file)
        xyz_native = torch.from_numpy(xyz_native).float().to(device) / 100
        xyz_moved_native = apply_rotate_matrix(euler_angle, xyz_native, norm=True)

    nib.freesurfer.write_geometry(sphere_moved_native_file, xyz_moved_native.detach().cpu().numpy() * 100,
                                  faces_native)
    print(f'sphere.reg >>> {sphere_moved_native_file}')


def infer(moving_datas, fixed_datas, models, faces, ico_levels, features, device='cuda'):
    assert len(moving_datas) > 0

    # negative_area_triangle_list = list()

    sulc_moving_fs6, curv_moving_fs6, xyz_moving_fs6, faces_moving_fs6, seg_moving_fs6 = moving_datas
    sulc_fixed_fs6, curv_fixed_fs6, xyz_fixed_fs6, faces_fixed_fs6, seg_fixed_fs6 = fixed_datas

    sulc_moving_fs6 = sulc_moving_fs6.T.to(device)
    sulc_fixed_fs6 = sulc_fixed_fs6.T.to(device)

    curv_moving_fs6 = curv_moving_fs6.T.to(device)
    curv_fixed_fs6 = curv_fixed_fs6.T.to(device)

    seg_moving_fs6 = seg_moving_fs6.squeeze().to(device)
    seg_fixed_fs6 = seg_fixed_fs6.squeeze().to(device)

    xyz_moving_fs6 = xyz_moving_fs6.squeeze().to(device)
    xyz_fixed_fs6 = xyz_fixed_fs6.squeeze().to(device)

    # NAMIC dont have 35
    if not torch.any(seg_moving_fs6 == 0):
        seg_fixed_fs6[seg_fixed_fs6 == 0] = 35

    # 904.et dont have 4
    if not torch.any(seg_moving_fs6 == 4):
        seg_fixed_fs6[seg_fixed_fs6 == 4] = 0

    xyz_moved = None
    seg_moving_lap = None
    for idx, model in enumerate(models):
        feature = features[idx]
        if feature == 'sulc':
            data_moving_fs6 = sulc_moving_fs6.to(device)
            data_fixed_fs6 = sulc_fixed_fs6.to(device)
        elif feature == 'curv':
            data_moving_fs6 = curv_moving_fs6.to(device)
            data_fixed_fs6 = curv_fixed_fs6.to(device)
        else:
            data_moving_fs6 = torch.cat((sulc_moving_fs6, curv_moving_fs6), 1).to(device)
            data_fixed_fs6 = torch.cat((sulc_fixed_fs6, curv_fixed_fs6), 1).to(device)

        ico_level = ico_levels[idx]
        points_num = get_points_num_by_ico_level(ico_level)
        faces_sphere = faces[ico_level].to(device)
        data_moving = data_moving_fs6[:points_num]
        data_fixed = data_fixed_fs6[:points_num]
        seg_moving = seg_moving_fs6[:points_num]
        seg_fixed = seg_fixed_fs6[:points_num]
        xyz_moving = xyz_moving_fs6[:points_num]
        xyz_fixed = xyz_fixed_fs6[:points_num]

        if xyz_moved is None:
            data_x = torch.cat((data_moving, data_fixed), 1).to(device)
            data_x = data_x.detach()

            xyz_moved_lap, euler_angle = model(data_x, xyz_moving, face=faces_sphere)

            xyz_moved = apply_rotate_matrix(euler_angle, xyz_moving, norm=True,
                                            en=model.en, face=faces_sphere)

            data_moving_lap = data_moving
            if seg_moving.sum() > 0:
                seg_moving_lap = seg_moving = F.one_hot(seg_moving).float().to(device)
            else:
                seg_moving_lap = None
        else:
            # upsample xyz_moved
            xyz_moved_upsample = upsample_std_sphere_torch(xyz_moved, norm=True)
            xyz_moved_upsample = xyz_moved_upsample.detach()

            # moved数据重采样
            moving_data_resample = resample_sphere_surface_barycentric(xyz_moved_upsample, xyz_fixed, data_moving)

            data_x = torch.cat((moving_data_resample, data_fixed), 1).to(device)

            xyz_moved_lap, euler_angle = model(data_x, xyz_moving, face=faces_sphere)

            if euler_angle.shape[1] == 3:
                euler_angle_interp_moved_upsample = resample_sphere_surface_barycentric(xyz_fixed, xyz_moved_upsample,
                                                                                        euler_angle)
                # euler_angle_interp_moved_upsample = bilinearResampleSphereSurf(xyz_moved_upsample, euler_angle, device)
                xyz_moved = apply_rotate_matrix(euler_angle_interp_moved_upsample, xyz_moved_upsample, norm=True,
                                                face=faces_sphere)
            else:  # 如果使用的是切平面的位移，不能对变形场进行插值，只能对结果坐标进行插值
                xyz_moved = resample_sphere_surface_barycentric(xyz_fixed, xyz_moved_upsample, xyz_moved_lap)
                xyz_moved = xyz_moved / (torch.norm(xyz_moved, dim=1, keepdim=True).repeat(1, 3))

            if seg_moving.sum() > 0:
                seg_moving = F.one_hot(seg_moving).float().to(device)
                seg_moving_resample = resample_sphere_surface_barycentric(xyz_moved_upsample, xyz_fixed, seg_moving)
            else:
                seg_moving_resample = None

            data_moving_lap = moving_data_resample
            seg_moving_lap = seg_moving_resample

        # negative_area_triangle = count_negative_area(faces_sphere, xyz_moved)
        # negative_area_triangle_list.append(negative_area_triangle)

    if seg_moving_lap is not None:
        seg_fixed = F.one_hot(seg_fixed).float().to(device)
    else:
        seg_fixed = False

    return xyz_fixed, xyz_moved, xyz_moved_lap, data_fixed, data_moving, data_moving_lap, euler_angle, \
        seg_moving, seg_moving_lap, seg_fixed


def rd_sample_data(data_level6):
    sulc_moving_fs6, curv_moving_fs6, xyz_moving_fs6, faces_moving_fs6, seg_moving_fs6 = data_level6
    points_num = np.random.choice([642, 2562, 10242, 40962])
    if points_num != 40962:
        sulc_moving_fs6 = sulc_moving_fs6.T.squeeze()[:points_num]
        while len(sulc_moving_fs6) < 40962:
            sulc_moving_fs6 = upsample_std_sphere_torch(sulc_moving_fs6)
        data_level6[0] = sulc_moving_fs6.T.unsqueeze(0)
    return data_level6


def run_epoch(models, faces, config, dataloader,
              save_result=False, dir_recon=None, dir_rigid=None, dir_result=None, is_train=False):
    device = config['device']
    features = config['feature']
    ico_levels = config['ico_levels']
    subs_loss = []
    for datas_moving, datas_fixed in dataloader:
        datas_moving = datas_moving[0]

        rd_sample = config['rd_sample']
        if rd_sample and is_train:
            datas_moving = rd_sample_data(datas_moving)

        datas_fixed = datas_fixed[0]

        xyz_fixed, xyz_moved, xyz_moved_lap, fixed_data, data_moving, data_moving_lap, euler_angle, \
            seg_moving, seg_moving_lap, seg_fixed, \
            = infer(datas_moving, datas_fixed, models, faces, ico_levels, features, device)


        if save_result:
            hemisphere = config["hemisphere"]
            save_sphere_reg(config, hemisphere, xyz_moved, euler_angle, dir_recon, dir_rigid, dir_result, device)

    return subs_loss


@torch.no_grad()
def hemisphere_predict(models, config, hemisphere,
                       dir_recon, dir_rigid=None, dir_result=None,
                       seg=False):
    for model in models:
        model.eval()
    # 获取config_train的配置
    feature = config['feature']  # 加载的数据类型
    faces = config['face']

    # 数据目录
    dir_fixed = config["dir_fixed"]  # fixed数据目录

    dataset_train = SphericalDataset(dir_fixed, dir_rigid,
                                     hemisphere, feature=feature, norm_type=config["normalize_type"],
                                     ico_levels=['fsaverage6'],
                                     seg=seg, is_train=False, is_da=False, is_rigid=config['is_rigid'])

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, num_workers=0)


    subs_loss = run_epoch(models, faces, config, dataloader_train,
                          save_result=True, dir_recon=dir_recon, dir_rigid=dir_rigid, dir_result=dir_result)

    return subs_loss


def train_val(config):
    # 获取config_train的配置
    device = config['device']  # 使用的硬件

    if config['validation'] is True:
        # 1. interp file
        interp_dir_single(config["dir_predict_recon"], config["dir_predict_rigid"], config["dir_fixed"],
                          'fsaverage6', is_rigid=config['is_rigid'])

        models = []
        for model_file in config['model_files'][:config["ico_index"] + 1]:
            print(f'<<< model : {model_file}')
            model = torch.load(model_file)['model']
            model.to(device)
            model.eval()
            models.append(model)

        hemisphere_predict(models, config, config['hemisphere'],
                           dir_recon=config['dir_predict_recon'],
                           dir_rigid=config['dir_predict_rigid'],
                           dir_result=config['dir_predict_result'],
                           seg=False)

        return
