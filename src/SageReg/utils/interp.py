import os
import nibabel as nib
import numpy as np
import torch
from pytorch3d.ops.knn import knn_points
import shutil

abspath = os.path.abspath(os.path.dirname(__file__))


def upsample_std_sphere_torch(feature: torch.tensor, norm=False):
    feature_num = len(feature)
    next_level_dict = {
        12: 'fsaverage0',
        42: 'fsaverage1',
        162: 'fsaverage2',
        642: 'fsaverage3',
        2562: 'fsaverage4',
        10242: 'fsaverage5',
        40962: 'fsaverage6',
    }
    next_level = next_level_dict[feature_num]
    upsample_neighbors_file = os.path.join(abspath, 'auxi_data', f'{next_level}_upsample_neighbors.npz')
    upsample_neighbors_load = np.load(upsample_neighbors_file)
    upsample_neighbors = upsample_neighbors_load['upsample_neighbors']
    feature_upsample = (feature[upsample_neighbors[:, 0]] + feature[upsample_neighbors[:, 1]]) / 2
    if norm:
        feature_upsample = feature_upsample / torch.norm(feature_upsample, dim=1, keepdim=True)
    feature_upsample = torch.cat([feature, feature_upsample], dim=0)
    return feature_upsample


def in_triangle(a, b, c, p):
    """
    check p in triangle_abc
    """
    ab = b - a
    ap = p - a
    bc = c - b
    bp = p - b
    ca = a - c
    cp = p - c
    cross_ab_ap = torch.cross(ab, ap)
    cross_bc_bp = torch.cross(bc, bp)
    cross_ca_cp = torch.cross(ca, cp)
    if cross_ab_ap > 0 and cross_bc_bp > 0 and cross_ca_cp > 0:  # in triangle
        return 1
    if cross_ab_ap < 0 and cross_bc_bp < 0 and cross_ca_cp < 0:  # in triangle
        return 1
    if cross_ab_ap * cross_bc_bp * cross_ca_cp == 0:  # on one line
        return 0
    return -1


def find_intersection(p1, p2, p3, x):
    p12 = p2 - p1
    p13 = p3 - p1
    norm_plane = torch.cross(p12, p13)

    p1_dot_n = (p1 * norm_plane).sum(axis=1, keepdims=True)
    x_dot_n = (x * norm_plane).sum(axis=1, keepdims=True)

    p_x = p1_dot_n / (x_dot_n + 1e-10) * x
    return p_x


def in_triangle_up(vertex_all, vertex_p, traingle_index, top3_near_vertex_index, top3_near_vertex, max_triangle_num):
    """
    check p in triangle_abc
    traingle_index : (dim,3,max_triangle_num // 3) 每个点相邻六个面的索引
    vertex_all : (dim,max_triangle_num // 3,3,3) 每个点相邻六个面的坐标
    vertex_p :   (1,top3_dim,3)
    top3_near_vertex_index : (top3_dim,3)

    """

    top3_dim = len(top3_near_vertex_index)
    for i in range(3):
        if i <= 0:
            near_vertex_coordinate = torch.unsqueeze(vertex_all[top3_near_vertex_index[:, i], :, :, :], dim=3)
            near_index_coordinate = torch.unsqueeze(traingle_index[top3_near_vertex_index[:, i], :, :], dim=2)
        else:
            near_vertex_coordinate = torch.cat(
                (near_vertex_coordinate, torch.unsqueeze(vertex_all[top3_near_vertex_index[:, i], :, :, :], dim=3)), dim=3)
            near_index_coordinate = torch.cat(
                (near_index_coordinate, torch.unsqueeze(traingle_index[top3_near_vertex_index[:, i], :, :], dim=2)), dim=2)
    near_index_coordinate = near_index_coordinate.reshape(top3_dim, max_triangle_num, 3)
    a_all = near_vertex_coordinate[:, :, 0, :, :]  # (top3_dim,max_triangle_num // 3,3,3) 三角形其中一个顶点所在六个面的顶点的坐标
    b_all = near_vertex_coordinate[:, :, 1, :, :]
    c_all = near_vertex_coordinate[:, :, 2, :, :]
    p = vertex_p.squeeze(0)
    for i in range(max_triangle_num // 3):
        for j in range(3):

            a = a_all[:, i, j, :]
            b = b_all[:, i, j, :]
            c = c_all[:, i, j, :]
            node = find_intersection(a, b, c, p)
            ab = b - a
            ap = node - a
            ac = c - a

            norm_ab = ab / torch.norm(ab)
            norm_ap = ap / torch.norm(ap)
            norm_ac = ac / torch.norm(ac)
            cross_ab_ap = torch.cross(norm_ab, norm_ap, dim=1)
            cross_ab_ac = torch.cross(norm_ab, norm_ac, dim=1)
            val_1 = torch.sum(torch.mul(cross_ab_ap, cross_ab_ac), dim=1)
            bc = c - b
            bp = node - b
            ba = a - b

            norm_bc = bc / torch.norm(bc)
            norm_bp = bp / torch.norm(bp)
            norm_ba = ba / torch.norm(ba)
            cross_bc_bp = torch.cross(norm_bc, norm_bp, dim=1)
            cross_bc_ba = torch.cross(norm_bc, norm_ba, dim=1)
            val_2 = torch.sum(torch.mul(cross_bc_bp, cross_bc_ba), dim=1)
            ca = a - c
            cp = node - c
            cb = b - c

            norm_ca = ca / torch.norm(ca)
            norm_cp = cp / torch.norm(cp)
            norm_cb = cb / torch.norm(cb)
            cross_ca_cp = torch.cross(norm_ca, norm_cp, dim=1)
            cross_ca_cb = torch.cross(norm_ca, norm_cb, dim=1)
            val_3 = torch.sum(torch.mul(cross_ca_cp, cross_ca_cb), dim=1)

            val_12 = torch.logical_and(val_1 > 0, val_2 > 0)
            val_123 = torch.logical_and(val_12 == True, val_3 > 0)
            if i == 0 and j == 0:
                val_all = torch.unsqueeze(val_123, dim=1)
                node_all = torch.unsqueeze(node, dim=1)
            else:
                val_all = torch.cat((val_all, val_123.unsqueeze(1)), dim=1)
                node_all = torch.cat((node_all, node.unsqueeze(1)), dim=1)

    true_traingle_index = torch.argmax(val_all.double(), dim=1).unsqueeze(1)
    # zero_traingle_index = torch.argwhere(torch.sum(val_all, axis=1) == 0).squeeze()
    zero_traingle_index = torch.sum(val_all, axis=1) == 0
    for i in range(3):
        if i == 0:
            new_top3_near_vertex_index = torch.gather(input=near_index_coordinate[:, :, i], dim=1,
                                                      index=true_traingle_index)
            node_vertex_all = torch.gather(input=node_all[:, :, i], dim=1,
                                                      index=true_traingle_index)
        else:
            new_top3_near_vertex_index = torch.cat((new_top3_near_vertex_index,
                                                    torch.gather(input=near_index_coordinate[:, :, i], dim=1,
                                                                 index=true_traingle_index)), dim=1)
            node_vertex_all = torch.cat((node_vertex_all, torch.gather(input=node_all[:, :, i], dim=1,
                                                                       index=true_traingle_index)), dim=1)

    new_top3_near_vertex_index[zero_traingle_index] = top3_near_vertex_index[zero_traingle_index, :]

    node_vertex_all[zero_traingle_index] = find_intersection(top3_near_vertex[zero_traingle_index, 0],
                                                             top3_near_vertex[zero_traingle_index, 1],
                                                             top3_near_vertex[zero_traingle_index, 2],
                                                             p[zero_traingle_index, :])

    return new_top3_near_vertex_index, node_vertex_all


def find_real_traingle_up(top3_near_vertex_index, p1, face, orig_xyz, device):
    """
    orig_xyz : (dim,3)
    top3_near_vertex_index : (top3_dim,3)
    traingle_index : (dim,max_triangle_num) max_triangle_num=30

    """
    fs_vertex_nums = [2562, 10242, 40962, 163842]
    dim = len(orig_xyz)
    if dim in fs_vertex_nums:
        triangle_file = os.path.join(abspath, 'auxi_data', f'fs_parameters_{dim}.npz')
        fs_parameters = np.load(triangle_file)
        traingle_index = torch.tensor(fs_parameters['traingle_index']).to(device)
        vertex_all = torch.tensor(fs_parameters['vertex_all']).float().to(device)
        max_triangle_num = 18
    else:
        max_triangle_num = 30
        traingle_index = dict([(k, []) for k in range(dim)])  # 获取face上每个顶点所在三角形的索引
        if type(face) is torch.Tensor:
            face = face.cpu().numpy()
        for vertex_index in face:
            for i in range(3):
                traingle_index[vertex_index[i]].extend(vertex_index.tolist())
        for i in range(dim):
            if len(traingle_index[i]) > max_triangle_num:
                traingle_index[i] = traingle_index[i][:max_triangle_num]
            elif len(traingle_index[i]) < max_triangle_num:
                traingle_index[i].extend([i for k in range(max_triangle_num - len(traingle_index[i]))])

        traingle_index = torch.tensor(list(traingle_index.values())).to(device)
        for i in range(max_triangle_num):
            if i == 0:
                vertex_all = torch.unsqueeze(orig_xyz[traingle_index[:, i], :], dim=1)
            else:
                vertex_all = torch.cat((vertex_all, torch.unsqueeze(orig_xyz[traingle_index[:, i], :], dim=1)), dim=1)

    for i in range(3):
        if i == 0:
            top3_near_vertex = torch.unsqueeze(orig_xyz[top3_near_vertex_index[:, i], :], dim=1)
        else:
            top3_near_vertex = torch.cat((top3_near_vertex, torch.unsqueeze(orig_xyz[top3_near_vertex_index[:, i], :], dim=1)), dim=1)
    vertex_all = vertex_all.reshape(dim, max_triangle_num // 3, 3, 3)
    traingle_index_numpy = traingle_index.reshape(dim, max_triangle_num // 3, 3)

    new_top3_near_vertex_index = in_triangle_up(vertex_all, p1, traingle_index_numpy, top3_near_vertex_index, top3_near_vertex, max_triangle_num)
    return new_top3_near_vertex_index


def resample_sphere_surface_barycentric(orig_xyz, target_xyz, orig_value, face=None, device='cuda'):
    """
    Interpolate moving points using fixed points and its feature

    orig_xyz:          N*3, torch cuda tensor, known fixed sphere points
    target_xyz,         N*3, torch cuda tensor, points to be interpolated
    orig_value:         N*3, torch cuda tensor, known feature corresponding to fixed points
    device:             'torch.device('cpu')', or torch.device('cuda:0'), or ,torch.device('cuda:1')

    """
    orig_xyz = orig_xyz.to(device)
    target_xyz = target_xyz.to(device)
    orig_value = orig_value.to(device)
    assert orig_xyz.shape[0] == orig_value.shape[0]

    p1 = target_xyz.unsqueeze(0)
    p2 = orig_xyz.unsqueeze(0)
    result_all = knn_points(p1, p2, K=3)
    top3_near_vertex_index = result_all[1].squeeze()

    if face is not None:
        top3_near_vertex_index, p_intersection = find_real_traingle_up(top3_near_vertex_index, p1, face, orig_xyz, device)

    top3_near_vertex_0 = orig_xyz[top3_near_vertex_index[:, 0], :]
    top3_near_vertex_1 = orig_xyz[top3_near_vertex_index[:, 1], :]
    top3_near_vertex_2 = orig_xyz[top3_near_vertex_index[:, 2], :]

    p_intersection = find_intersection(top3_near_vertex_0, top3_near_vertex_1, top3_near_vertex_2, target_xyz).detach()

    area_bcp = torch.norm(torch.cross(top3_near_vertex_1 - p_intersection, top3_near_vertex_2 - p_intersection),
                          2, dim=1) / 2.0
    area_acp = torch.norm(torch.cross(top3_near_vertex_2 - p_intersection, top3_near_vertex_0 - p_intersection),
                          2, dim=1) / 2.0
    area_abp = torch.norm(torch.cross(top3_near_vertex_0 - p_intersection, top3_near_vertex_1 - p_intersection),
                          2, dim=1) / 2.0

    w = torch.cat((area_bcp.unsqueeze(1), area_acp.unsqueeze(1), area_abp.unsqueeze(1)), 1)
    w[w.sum(1) == 0] = 1
    inter_weight = w / w.sum(1).unsqueeze(1)

    target_value = torch.sum(inter_weight.unsqueeze(2) * orig_value[top3_near_vertex_index], 1)

    # target_value[area_bcp == 0] = orig_value[area_bcp == 0]  # 11-24 如果PABC共线，该目标位置点的值使用重叠点的值

    return target_value


def resample_sphere_surface_nearest(orig_xyz, target_xyz, orig_annot):
    assert orig_xyz.shape[0] == orig_annot.shape[0]

    p1 = target_xyz.unsqueeze(0)
    p2 = orig_xyz.unsqueeze(0)
    result = knn_points(p1, p2, K=1)

    idx_num = result[1].squeeze()
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


if __name__ == '__main__':
    a = torch.from_numpy(np.arange(2562 * 3).reshape(2562, 3))
    upsample_std_sphere_torch(a)
