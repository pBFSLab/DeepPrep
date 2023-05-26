import os
from collections import defaultdict
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


def find_intersection(p1, p2, p3, x, axis=1):
    p12 = p2 - p1
    p13 = p3 - p1
    norm_plane = torch.cross(p12, p13)

    p1_dot_n = (p1 * norm_plane).sum(axis=axis, keepdims=True)
    x_dot_n = (x * norm_plane).sum(axis=axis, keepdims=True)

    p_x = p1_dot_n / (x_dot_n + 1e-10) * x
    return p_x


def in_triangle_up(triangle_index, vertex_target, vertex_orig, faces, topk_near_vertex_index):
    """
    check p in triangle_abc
    triangle_index : (dim,3,max_triangle_num // 3) 每个点相邻六个面的索引
    vertex_all : (dim,max_triangle_num // 3,3,3) 每个点相邻六个面的坐标
    vertex_p :   (1,top3_dim,3)
    top3_near_vertex_index : (top3_dim,3)

    """
    # find every point's near faces' vertex coord
    vertex_target = vertex_target.squeeze()  # target coord
    vertex_orig = vertex_orig.squeeze()  # orig coord
    # one iter one_near_vertex/one_face 节省显存,每次可以遍历一个最邻近点的face
    up_near_vertex_index = topk_near_vertex_index[:, :3]

    for near_num in range(topk_near_vertex_index.shape[1]):
        near_vertex = topk_near_vertex_index[:, near_num]
        near_triangle_index_test = triangle_index[near_vertex]
        near_triangle_vertex_index_test = faces[near_triangle_index_test]

        a = vertex_orig[near_triangle_vertex_index_test][:, :, 0, :]
        b = vertex_orig[near_triangle_vertex_index_test][:, :, 1, :]
        c = vertex_orig[near_triangle_vertex_index_test][:, :, 2, :]
        p = find_intersection(a, b, c, vertex_target.unsqueeze(1), axis=2)

        ab = b - a
        ap = p - a
        ac = c - a

        norm_ab = ab / torch.norm(ab)
        norm_ap = ap / torch.norm(ap)
        norm_ac = ac / torch.norm(ac)
        cross_ab_ap = torch.cross(norm_ab, norm_ap, dim=2)
        cross_ab_ac = torch.cross(norm_ab, norm_ac, dim=2)
        val_1 = torch.sum(torch.mul(cross_ab_ap, cross_ab_ac), dim=2)
        bc = c - b
        bp = p - b
        ba = a - b

        norm_bc = bc / torch.norm(bc)
        norm_bp = bp / torch.norm(bp)
        norm_ba = ba / torch.norm(ba)
        cross_bc_bp = torch.cross(norm_bc, norm_bp, dim=2)
        cross_bc_ba = torch.cross(norm_bc, norm_ba, dim=2)
        val_2 = torch.sum(torch.mul(cross_bc_bp, cross_bc_ba), dim=2)
        ca = a - c
        cp = p - c
        cb = b - c

        norm_ca = ca / torch.norm(ca)
        norm_cp = cp / torch.norm(cp)
        norm_cb = cb / torch.norm(cb)
        cross_ca_cp = torch.cross(norm_ca, norm_cp, dim=2)
        cross_ca_cb = torch.cross(norm_ca, norm_cb, dim=2)
        val_3 = torch.sum(torch.mul(cross_ca_cp, cross_ca_cb), dim=2)

        val_12 = torch.logical_and(val_1 > 0, val_2 > 0)
        in_face = torch.logical_and(val_12, val_3 > 0)

        nun_zero_index = torch.nonzero(in_face.sum(dim=1)).squeeze(1)

        true_index = torch.gather(near_triangle_vertex_index_test[nun_zero_index], dim=1,
                     index=torch.max(in_face[nun_zero_index], dim=1, keepdim=True)[1].repeat(1, 3).unsqueeze(1)).squeeze(1)
        up_near_vertex_index[nun_zero_index] = true_index

    # for near_num in range(topk_near_vertex_index.shape[1]):
    #     for face_num in range(triangle_index.shape[1]):
    #         near_vertex = topk_near_vertex_index[:, near_num]
    #         near_triangle_index = triangle_index[near_vertex][:, face_num]
    #         near_triangle_vertex_index = faces[near_triangle_index]
    #
    #         a = vertex_orig[near_triangle_vertex_index][:, 0, :]
    #         b = vertex_orig[near_triangle_vertex_index][:, 1, :]
    #         c = vertex_orig[near_triangle_vertex_index][:, 2, :]
    #         p = find_intersection(a, b, c, vertex_target)
    #
    #         ab = b - a
    #         ap = p - a
    #         ac = c - a
    #
    #         norm_ab = ab / torch.norm(ab)
    #         norm_ap = ap / torch.norm(ap)
    #         norm_ac = ac / torch.norm(ac)
    #         cross_ab_ap = torch.cross(norm_ab, norm_ap, dim=1)
    #         cross_ab_ac = torch.cross(norm_ab, norm_ac, dim=1)
    #         val_1 = torch.sum(torch.mul(cross_ab_ap, cross_ab_ac), dim=1)
    #         bc = c - b
    #         bp = p - b
    #         ba = a - b
    #
    #         norm_bc = bc / torch.norm(bc)
    #         norm_bp = bp / torch.norm(bp)
    #         norm_ba = ba / torch.norm(ba)
    #         cross_bc_bp = torch.cross(norm_bc, norm_bp, dim=1)
    #         cross_bc_ba = torch.cross(norm_bc, norm_ba, dim=1)
    #         val_2 = torch.sum(torch.mul(cross_bc_bp, cross_bc_ba), dim=1)
    #         ca = a - c
    #         cp = p - c
    #         cb = b - c
    #
    #         norm_ca = ca / torch.norm(ca)
    #         norm_cp = cp / torch.norm(cp)
    #         norm_cb = cb / torch.norm(cb)
    #         cross_ca_cp = torch.cross(norm_ca, norm_cp, dim=1)
    #         cross_ca_cb = torch.cross(norm_ca, norm_cb, dim=1)
    #         val_3 = torch.sum(torch.mul(cross_ca_cp, cross_ca_cb), dim=1)
    #
    #         val_12 = torch.logical_and(val_1 > 0, val_2 > 0)
    #         in_face = torch.logical_and(val_12, val_3 > 0)
    #         # print(in_face.sum())
    #
    #         up_near_vertex_index[in_face] = near_triangle_vertex_index[in_face]

    return up_near_vertex_index


def find_near_triangle(faces, orig_vertex_num, max_triangle_num, save=False):
    fs_vertex_nums = (642, 2562, 10242, 40962, 163842)
    triangle_file = os.path.join(abspath, 'auxi_data', f'fs_barycentric_{orig_vertex_num}.npz')

    if orig_vertex_num in fs_vertex_nums and os.path.exists(triangle_file):
        fs_parameters = np.load(triangle_file)
        triangle_index_metrix = fs_parameters['triangle_index']
    else:
        # find every point's triangle face
        triangle_index = defaultdict(set)
        if type(faces) is torch.Tensor:
            face_np = faces.cpu().numpy()
        else:
            face_np = faces
        for face_index, vertex_index in enumerate(face_np):
            for i in range(3):
                triangle_index[vertex_index[i]].add(face_index)

        # from collections import Counter
        # counter = Counter()
        # for indexes in triangle_index.values():
        #     counter.update([len(indexes)])
        # print(counter)  # 经过统计，每个point最多保留13个face即可

        triangle_index_metrix = np.zeros((orig_vertex_num, max_triangle_num), dtype=int)  # shape = (point_num, triangle_num)
        for i in range(orig_vertex_num):
            tmp = list(triangle_index[i])
            end_index = min(len(tmp), max_triangle_num)
            triangle_index_metrix[i, 0:end_index] = tmp[:end_index]

        if save and not os.path.exists(triangle_file):
            np.savez(triangle_file, triangle_index=triangle_index_metrix)
    return triangle_index_metrix


def find_real_triangle_up(topk_near_vertex_index, xyz_target, xyz_orig, faces, device):
    """
    orig_xyz : (dim,3)
    top3_near_vertex_index : (top3_dim,3)
    triangle_index : (dim,max_triangle_num) max_triangle_num=30

    """
    max_triangle_num = 13
    triangle_index_metrix = find_near_triangle(faces, len(xyz_orig), max_triangle_num)

    triangle_index_metrix = torch.from_numpy(triangle_index_metrix).to(device)
    if type(faces) is np.ndarray:
        faces = torch.from_numpy(faces).to(device)
    new_top3_near_vertex_index = in_triangle_up(triangle_index_metrix, xyz_target, xyz_orig, faces, topk_near_vertex_index)

    return new_top3_near_vertex_index


def resample_sphere_surface_barycentric(orig_xyz, target_xyz, orig_value, orig_face=None, device='cuda'):
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

    k = 5

    p1 = target_xyz.unsqueeze(0)
    p2 = orig_xyz.unsqueeze(0)
    result_all = knn_points(p1, p2, K=k)
    top3_near_vertex_index = result_all[1].squeeze()

    if orig_face is not None:
        top3_near_vertex_index = find_real_triangle_up(top3_near_vertex_index, target_xyz, orig_xyz, orig_face, device)
    else:
        top3_near_vertex_index = top3_near_vertex_index[:, :3]

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
    print(f'interp: >>> {sulc_interp_file}')
    print(f'interp: >>> {curv_interp_file}')
    # print(f'interp: >>> {sphere_interp_file}')


if __name__ == '__main__':
    # a = torch.from_numpy(np.arange(2562 * 3).reshape(2562, 3))
    # upsample_std_sphere_torch(a)
    _, faces_fs = nib.freesurfer.read_geometry('/usr/local/freesurfer/subjects/fsaverage3/surf/lh.sphere')
    num = len(_)
    find_near_triangle(faces_fs, num, max_triangle_num=6, save=True)
