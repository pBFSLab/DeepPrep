import os
import nibabel as nib
import numpy as np
import torch
from pytorch3d.ops.knn import knn_points
import shutil

abspath = os.path.abspath(os.path.dirname(__file__))


def resample_sphere_surface_barycentric(orig_xyz, target_xyz, orig_value, device='cuda'):
    """
    Interpolate moving points using fixed points and its feature

    orig_xyz:          N*3, torch cuda tensor, known fixed sphere points
    target_xyz,         N*3, torch cuda tensor, points to be interpolated
    orig_value:         N*3, torch cuda tensor, known feature corresponding to fixed points
    device:             'torch.device('cpu')', or torch.device('cuda:0'), or ,torch.device('cuda:1')

    """
    assert orig_xyz.shape[0] == orig_value.shape[0]

    p1 = target_xyz.unsqueeze(0)
    p2 = orig_xyz.unsqueeze(0)
    result_all = knn_points(p1, p2, K=3)
    top3_near_vertex_index = result_all[1].squeeze()
    top3_near_vertex_0 = orig_xyz[top3_near_vertex_index[:, 0], :]
    top3_near_vertex_1 = orig_xyz[top3_near_vertex_index[:, 1], :]
    top3_near_vertex_2 = orig_xyz[top3_near_vertex_index[:, 2], :]

    # find the triangle face that the inersection is in, if the intersection
    # is in, the area of 3 small triangles is equal to the whole triangle
    area_bcp = torch.norm(torch.cross(top3_near_vertex_1 - target_xyz, top3_near_vertex_2 - target_xyz),
                          2, dim=1) / 2.0
    area_acp = torch.norm(torch.cross(top3_near_vertex_2 - target_xyz, top3_near_vertex_0 - target_xyz),
                          2, dim=1) / 2.0
    area_abp = torch.norm(torch.cross(top3_near_vertex_0 - target_xyz, top3_near_vertex_1 - target_xyz),
                          2, dim=1) / 2.0

    w = torch.cat((area_bcp.unsqueeze(1), area_acp.unsqueeze(1), area_abp.unsqueeze(1)), 1)
    w[w.sum(1) == 0] = 1
    inter_weight = w / w.sum(1).unsqueeze(1)

    target_value = torch.sum(inter_weight.unsqueeze(2) * orig_value[top3_near_vertex_index], 1)

    # target_value[area_bcp == 0] = orig_value[area_bcp == 0]  # 11-24 如果PABC共线，该目标位置点的值使用重叠点的值

    return target_value


def resample_sphere_surface_nearest(orig_xyz, target_xyz, orig_annot, gn=False):
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


def interp_annot_knn(sphere_orig_file, sphere_target_file, orig_annot_file,
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

    data_target = resample_sphere_surface_nearest(xyz_orig_t, xyz_target_t, data_orig_t)
    data_target = data_target.detach().cpu().numpy()
    nib.freesurfer.write_annot(interp_result_file, data_target, data_orig[1], data_orig[2], fill_ctab=True)
    print(f'save_interp_annot_file_path: >>> {interp_result_file}')

    if sphere_interp_file is not None and not os.path.exists(sphere_interp_file):
        shutil.copyfile(sphere_target_file, sphere_interp_file)
        print(f'copy sphere_target_file to sphere_interp_file: >>> {sphere_interp_file}')


def interp_morph_barycentric(sphere_orig_file, sphere_target_file, gt_orig_morph_file,
                             interp_result_file, sphere_interp_file=None, device='cuda'):
    # 加载数据
    data_orig = nib.freesurfer.read_morph_data(gt_orig_morph_file).astype(np.float32)
    xyz_orig, faces_orig = nib.freesurfer.read_geometry(sphere_orig_file)
    xyz_orig = xyz_orig.astype(np.float32)

    xyz_target, faces_target = nib.freesurfer.read_geometry(sphere_target_file)
    xyz_target = xyz_target.astype(np.float32)

    data_orig_t = torch.from_numpy(data_orig).to(device)
    xyz_orig_t = torch.from_numpy(xyz_orig).to(device)
    xyz_target_t = torch.from_numpy(xyz_target).to(device)

    data_interp = resample_sphere_surface_barycentric(xyz_orig_t, xyz_target_t, data_orig_t.unsqueeze(1))

    nib.freesurfer.write_morph_data(interp_result_file, data_interp.squeeze().cpu().numpy())
    print(f'save_interp_morph_file_path: >>> {interp_result_file}')

    if sphere_interp_file is not None and not os.path.exists(sphere_interp_file):
        shutil.copyfile(sphere_target_file, sphere_interp_file)
        print(f'copy sphere_target_file to sphere_interp_file: >>> {sphere_interp_file}')


def interp_sulc_curv_barycentric(sulc_orig_file, curv_orig_file, sphere_orig_file, sphere_target_file,
                                 sulc_interp_file, curv_interp_file, sphere_interp_file=None, device='cuda'):
    # 加载数据
    sulc_orig = nib.freesurfer.read_morph_data(sulc_orig_file).astype(np.float32)
    curv_orig = nib.freesurfer.read_morph_data(curv_orig_file).astype(np.float32)
    xyz_orig, faces_orig = nib.freesurfer.read_geometry(sphere_orig_file)
    xyz_orig = xyz_orig.astype(np.float32)

    xyz_target, faces_target = nib.freesurfer.read_geometry(sphere_target_file)
    xyz_target = xyz_target.astype(np.float32)

    sulc_orig_t = torch.from_numpy(sulc_orig).to(device)
    curv_orig_t = torch.from_numpy(curv_orig).to(device)
    xyz_orig_t = torch.from_numpy(xyz_orig).to(device)
    xyz_target_t = torch.from_numpy(xyz_target).to(device)

    sulc_interp = resample_sphere_surface_barycentric(xyz_orig_t, xyz_target_t, sulc_orig_t.unsqueeze(1))
    curv_interp = resample_sphere_surface_barycentric(xyz_orig_t, xyz_target_t, curv_orig_t.unsqueeze(1))

    nib.freesurfer.write_morph_data(sulc_interp_file, sulc_interp.squeeze().cpu().numpy())
    nib.freesurfer.write_morph_data(curv_interp_file, curv_interp.squeeze().cpu().numpy())

    if sphere_interp_file is not None and not os.path.exists(sphere_interp_file):
        shutil.copyfile(sphere_target_file, sphere_interp_file)
    # print(f'interp: >>> {sulc_interp_file}')
    # print(f'interp: >>> {curv_interp_file}')
    # print(f'interp: >>> {sphere_interp_file}')
