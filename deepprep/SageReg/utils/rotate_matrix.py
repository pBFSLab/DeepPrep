import numpy as np
import torch
import nibabel as nib
from featreg.utils.interp import resample_sphere_surface_barycentric


def save_rotate_matrix(rotate_matrix, out_file, xyz=None):
    if xyz is not None:
        np.savez(out_file, data=rotate_matrix, xyz=xyz)
    else:
        np.savez(out_file, data=rotate_matrix)
    print(f'save_rotate_matrix: >>> {out_file}')


def apply_rigid_rotate_matrix(sphere_moving_f, rotate_matrix_rigid_f, sphere_moving_rigid_f):
    xyz_moving, faces_orig = nib.freesurfer.read_geometry(sphere_moving_f)
    xyz_moving = xyz_moving.astype(np.float32)

    rigid_rotate_matrix = np.load(rotate_matrix_rigid_f)['data']

    xyz_moving_rigid = np.dot(rigid_rotate_matrix, np.transpose(xyz_moving))
    xyz_moving_rigid = np.transpose(xyz_moving_rigid)  # 新的顶点坐标_3D
    nib.freesurfer.write_geometry(sphere_moving_rigid_f, xyz_moving_rigid, faces_orig)
    print(f'apply_rotate_matrix: >>> {sphere_moving_rigid_f}')


def apply_norigid_rotate_matrix(sphere_moving_f, rotate_metrix_file, sphere_moved_f, device='cuda', norm=True):
    xyz_moving, faces_orig = nib.freesurfer.read_geometry(sphere_moving_f)
    xyz_moving = xyz_moving.astype(np.float32)

    data_np_load = np.load(rotate_metrix_file)
    euler_angle = data_np_load['data']
    xyz_fixed = data_np_load['xyz']

    euler_angle = torch.from_numpy(euler_angle.astype(np.float32)).to(device)
    xyz_orig_t = torch.from_numpy(xyz_fixed.astype(np.float32)).to(device)
    xyz_target_t = torch.from_numpy(xyz_moving.astype(np.float32)).to(device)
    euler_angle_interp = resample_sphere_surface_barycentric(xyz_orig_t, xyz_target_t, euler_angle)
    xyz_moving = xyz_target_t

    a = euler_angle_interp[:, [0]]
    b = euler_angle_interp[:, [1]]
    g = euler_angle_interp[:, [2]]
    r1 = torch.cat(
        [torch.cos(g) * torch.cos(b), torch.cos(g) * torch.sin(b) * torch.sin(a) - torch.sin(g) * torch.cos(a),
         torch.sin(g) * torch.sin(a) + torch.cos(g) * torch.cos(a) * torch.sin(b)], dim=1)
    r2 = torch.cat(
        [torch.cos(b) * torch.sin(g), torch.cos(g) * torch.cos(a) + torch.sin(g) * torch.sin(b) * torch.sin(a),
         torch.cos(a) * torch.sin(g) * torch.sin(b) - torch.cos(g) * torch.sin(a)], dim=1)
    r3 = torch.cat(
        [-torch.sin(b), torch.cos(b) * torch.sin(a), torch.cos(b) * torch.cos(a)], dim=1)
    moved_x = torch.sum(xyz_moving * r1, dim=1)
    moved_y = torch.sum(xyz_moving * r2, dim=1)
    moved_z = torch.sum(xyz_moving * r3, dim=1)
    sphere_moved = torch.cat((moved_x.unsqueeze(1), moved_y.unsqueeze(1), moved_z.unsqueeze(1)), dim=1)

    if norm:
        sphere_moved = sphere_moved / (torch.norm(sphere_moved, dim=1, keepdim=True).repeat(1, 3))

    nib.freesurfer.write_geometry(sphere_moved_f, sphere_moved.cpu().numpy(), faces_orig)
    print(f'apply_rotate_matrix: >>> {sphere_moved_f}')


def get_en_torch(xyz):
    device = xyz.device
    xyz = xyz.cpu().numpy()
    base_vector = np.array([0, 0, 1])
    en1 = np.cross(base_vector, xyz)
    idx = np.logical_and(xyz[:, 0] == 0, xyz[:, 1] == 0)
    en1[idx] = np.array([1, 0, 0])
    en2 = np.cross(en1, xyz)
    en1 = en1 / np.linalg.norm(en1, axis=1, keepdims=True)
    en2 = en2 / np.linalg.norm(en2, axis=1, keepdims=True)
    en = np.concatenate([en1[:, np.newaxis, :], en2[:, np.newaxis, :]], axis=1)
    en = torch.from_numpy(en).float().to(device)
    return en


def get_en(xyz):
    x_0 = np.argwhere(xyz[:, 0] == 0)
    y_0 = np.argwhere(xyz[:, 1] == 0)
    inter_ind = np.intersect1d(x_0, y_0)
    En_1 = np.cross(np.array([0, 0, 1]), xyz)
    En_1[inter_ind] = np.array([1, 0, 0])
    En_2 = np.cross(xyz, En_1)
    En_1 = En_1 / np.repeat(np.sqrt(np.sum(En_1 ** 2, axis=1))[:, np.newaxis], 3,
                            axis=1)  # normalize to unit orthonormal vector
    En_2 = En_2 / np.repeat(np.sqrt(np.sum(En_2 ** 2, axis=1))[:, np.newaxis], 3,
                            axis=1)  # normalize to unit orthonormal vector
    En = np.transpose(np.concatenate((En_1[np.newaxis, :], En_2[np.newaxis, :]), 0), (1, 2, 0))

    return En


def apply_rotate_matrix(euler_angle, xyz_moving, norm=False, en=None, face=None):
    if euler_angle.shape[1] == 3:

        a = euler_angle[:, [0]]
        b = euler_angle[:, [1]]
        g = euler_angle[:, [2]]
        r1 = torch.cat(
            [torch.cos(g) * torch.cos(b), torch.cos(g) * torch.sin(b) * torch.sin(a) - torch.sin(g) * torch.cos(a),
             torch.sin(g) * torch.sin(a) + torch.cos(g) * torch.cos(a) * torch.sin(b)], dim=1)
        r2 = torch.cat(
            [torch.cos(b) * torch.sin(g), torch.cos(g) * torch.cos(a) + torch.sin(g) * torch.sin(b) * torch.sin(a),
             torch.cos(a) * torch.sin(g) * torch.sin(b) - torch.cos(g) * torch.sin(a)], dim=1)
        r3 = torch.cat(
            [-torch.sin(b), torch.cos(b) * torch.sin(a), torch.cos(b) * torch.cos(a)], dim=1)
        moved_x = torch.sum(xyz_moving * r1, dim=1, keepdim=True)
        moved_y = torch.sum(xyz_moving * r2, dim=1, keepdim=True)
        moved_z = torch.sum(xyz_moving * r3, dim=1, keepdim=True)
        sphere_moved = torch.cat((moved_x, moved_y, moved_z), dim=1)

    elif euler_angle.shape[1] == 2:
        if en is None:
            en = get_en_torch(xyz_moving.detach())
        v_x = torch.sum(euler_angle * en[:, :, 0], dim=1, keepdim=True)
        v_y = torch.sum(euler_angle * en[:, :, 1], dim=1, keepdim=True)
        v_z = torch.sum(euler_angle * en[:, :, 2], dim=1, keepdim=True)
        vs = torch.cat((v_x, v_y, v_z), dim=1)

        sphere_moved = xyz_moving + vs / float(np.power(2, 6))
        for i in range(6):
            sphere_moved = sphere_moved / torch.norm(sphere_moved, dim=1, keepdim=True)
            sphere_moved = resample_sphere_surface_barycentric(xyz_moving, sphere_moved, sphere_moved, face=face)

    # sphere_moved = xyz_moving + euler_angle

    if norm:
        sphere_moved = sphere_moved / torch.norm(sphere_moved, dim=1, keepdim=True)
    return sphere_moved


# def apply_norigid_rotate_matrix(sphere_moving_f, rotate_metrix_file, sphere_moved_f, device='cuda'):
#     """
#     测试diffeomorp使用
#     """
#     xyz_moving, faces_orig = nib.freesurfer.read_geometry(sphere_moving_f)
#     xyz_moving = xyz_moving.astype(np.float32)
#
#     data_np_load = np.load(rotate_metrix_file)
#     euler_angle = data_np_load['data']
#     xyz_fixed = data_np_load['xyz']
#
#     euler_angle = torch.from_numpy(euler_angle.astype(np.float32)).to(device)
#     xyz_orig_t = torch.from_numpy(xyz_fixed.astype(np.float32)).to(device)
#     xyz_target_t = torch.from_numpy(xyz_moving.astype(np.float32)).to(device)
#     euler_angle_interp = resample_sphere_surface_barycentric(xyz_orig_t, xyz_target_t, euler_angle)
#     xyz_moving = xyz_target_t
#
#     xyz_moved = xyz_moving + euler_angle_interp
#     sphere_moved = xyz_moved / (torch.norm(xyz_moved, dim=1, keepdim=True).repeat(1, 3))
#
#     nib.freesurfer.write_geometry(sphere_moved_f, sphere_moved.cpu().numpy(), faces_orig)
#     print(f'apply_rotate_matrix: >>> {sphere_moved_f}')
