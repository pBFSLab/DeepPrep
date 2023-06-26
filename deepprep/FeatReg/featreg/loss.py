#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2021 12 07

@author: Ning An

Contact: ning.an@neuralgalaxy.com

"""
import numpy as np
import torch
from torch import nn

from featreg.utils.interp import resample_sphere_surface_barycentric


def get_vector_angle(vector_a, vector_b):
    """
    获取两个向量的夹角
    """
    norm_a = torch.norm(vector_a, dim=1)
    norm_b = torch.norm(vector_b, dim=1)
    axb = torch.sum(vector_a * vector_b, dim=1)
    cos_theta = axb / (norm_a * norm_b)
    angle = torch.acos(cos_theta)
    # angle_d = angle * 180 / np.pi
    # return angle_d
    return angle


def get_face_coords(faces, vertex_xyz):
    """
    获取每个face三个顶点的坐标
    """
    a = faces[:, 0]
    b = faces[:, 1]
    c = faces[:, 2]
    xyz_a = vertex_xyz[a]
    xyz_b = vertex_xyz[b]
    xyz_c = vertex_xyz[c]
    return xyz_a, xyz_b, xyz_c


def get_face_edge_length_torch(faces, vertex_xyz):
    """
    获取每个face每条边的边长
    """
    xyz_a, xyz_b, xyz_c = get_face_coords(faces, vertex_xyz)
    ab_dis_temp = xyz_a - xyz_b
    ac_dis_temp = xyz_a - xyz_c
    bc_dis_temp = xyz_b - xyz_c
    edge = torch.zeros(faces.shape)
    edge[:, 0] = torch.norm(ab_dis_temp, dim=1)
    edge[:, 1] = torch.norm(bc_dis_temp, dim=1)
    edge[:, 2] = torch.norm(ac_dis_temp, dim=1)
    return edge


def get_face_area_torch(faces, vertex_xyz):
    """
    获取每个face的面积
    """
    xyz_a, xyz_b, xyz_c = get_face_coords(faces, vertex_xyz)
    vector_b_a = xyz_a - xyz_b
    vector_b_c = xyz_c - xyz_b
    triangle_area = torch.norm(torch.cross(vector_b_a, vector_b_c), 2, dim=1) / 2.0
    return triangle_area


def get_face_angle_torch(faces, vertex_xyz):
    """
    计算每个face三个夹角的角度
    """
    xyz_a, xyz_b, xyz_c = get_face_coords(faces, vertex_xyz)
    angle_a = get_vector_angle(xyz_b - xyz_a, xyz_c - xyz_a)
    angle_b = get_vector_angle(xyz_a - xyz_b, xyz_c - xyz_b)
    angle_c = get_vector_angle(xyz_a - xyz_c, xyz_b - xyz_c)
    angle = torch.cat([angle_a.unsqueeze(1), angle_b.unsqueeze(1), angle_c.unsqueeze(1)], dim=1)
    # print(angle[0], angle[0].sum())
    # print(angle[1], angle[0].sum())
    return angle


def get_distance_vertex_neighbor(vertex_xyz, neighbor):
    """
    计算每个顶点和它在拓扑相邻点的距离
    neighbor shape 为 2D
    """
    vertex_xyz_neighbor = vertex_xyz[neighbor]
    vertex_xyz_neighbor = vertex_xyz_neighbor.transpose(1, 2)
    delta = vertex_xyz.unsqueeze(2) - vertex_xyz_neighbor

    dist = torch.norm(delta, dim=1)
    return dist


# def compute_repulsion_loss(vertex_xyz, k=50):
#     p1, p2 = vertex_xyz, vertex_xyz,
#     p1 = p1.unsqueeze(0)
#     p2 = p2.unsqueeze(0)
#     result_all = knn_points(p2, p1, K=k)
#     dist, index = result_all[:2]
#     dist[dist > 1e-4] = 0
#     dist = dist.squeeze()
#     dist[:, :10] = 0
#     loss_pre = torch.sum(dist, dim=1)
#     loss = torch.sum(dist[:, 18:])
#     # top3_near_vertex_index = result.reshape(-1, 3)
#     # top3_near_vertex_0 = orig_xyz[top3_near_vertex_index[:, 0], :]
#     # top3_near_vertex_1 = orig_xyz[top3_near_vertex_index[:, 1], :]
#     # top3_near_vertex_2 = orig_xyz[top3_near_vertex_index[:, 2], :]
#     return loss


def compute_mssim(orig, target, neighbor,
                  l_range=None):
    k1 = 0.01
    k2 = 0.03
    if l_range is None:
        l_range = target.max() - target.min()
    c1 = (k1 * l_range) ** 2
    c2 = (k2 * l_range) ** 2
    # c3 = c2 / 2

    orig = orig[neighbor]
    target = target[neighbor]
    mean_x = torch.mean(orig, dim=1)
    std_x = torch.std(orig, dim=1)
    mean_y = torch.mean(target, dim=1)
    std_y = torch.std(target, dim=1)
    cov_xy = torch.sum((orig - mean_x.unsqueeze(1)) * (target - mean_y.unsqueeze(1)), dim=1) / (neighbor.shape[1] - 1)

    nu = (2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)
    de = (mean_x.pow(2) + mean_y.pow(2) + c1) * (std_x.pow(2) + std_y.pow(2) + c2)
    ssim = nu / de
    return ssim


def compute_mssim_loss(orig, target, neighbor):
    ssim = compute_mssim(orig, target, neighbor, None)
    loss_ssim = torch.mean(1 - ssim)
    return loss_ssim


def compute_repulsion_loss(vertex_xyz, dist_ring23_init, neighbor):
    """
    ring could be 1 2 3 23
    """
    dist = get_distance_vertex_neighbor(vertex_xyz, neighbor)
    delta = dist_ring23_init / dist - 1
    delta[delta < 0] = 0  # 进行限制，只推开到初始的距离即可
    loss = torch.mean(delta)
    return loss


def compute_edge_length_loss(edge_init, sphere_faces, sphere_xyz):
    """
    计算边长差异 edge_loss
    """
    edge_length = get_face_edge_length_torch(sphere_faces, sphere_xyz)
    # print(torch.mean(edge_length), torch.min(edge_length), torch.max(edge_length))
    mse_loss = torch.mean((edge_init - edge_length) ** 2)
    return mse_loss


def compute_triangle_area_loss(triangle_area_init, faces, vertex_xyz):
    """
    计算面积差异 area_loss
    """
    triangle_area = get_face_area_torch(faces, vertex_xyz)
    # print(T_area_init, T_area)
    triangle_area_neg_count = (triangle_area < 0).sum()
    if triangle_area_neg_count > 0:
        print(f"neg angle count : {triangle_area_neg_count}")
    index = triangle_area < triangle_area_init  # 只计算
    loss = triangle_area_init - triangle_area
    loss = torch.mean(loss[index])
    return loss


def compute_angle_loss(moved_xyz, face_angle_orig, faces_moved):
    """
    计算moved和moving的angle loss
    """
    angle_moved = get_face_angle_torch(faces_moved, moved_xyz)
    delta = angle_moved - face_angle_orig
    # loss = torch.mean(torch.abs(delta)[delta < 0])  # 仅计算角度变小的角度loss
    loss = torch.mean(torch.abs(delta))  # 仅计算角度变小的角度loss
    return loss


def compute_distance_loss(moved_xyz, dist_orig, neighbor):
    """
    计算相邻点距离差异 dist_loss
    """
    moved_dist = get_distance_vertex_neighbor(moved_xyz, neighbor)
    loss = torch.abs(moved_dist - dist_orig)
    loss = torch.mean(loss)
    return loss


def compute_sim_loss(predict, target, weight=None):
    target[target == 0] = 1e-8
    corr = ((predict - predict.mean(dim=0, keepdim=True)) * (target - target.mean(dim=0, keepdim=True))
            ).mean(dim=0, keepdim=True) / (predict.std(dim=0, keepdim=True) * target.std(dim=0, keepdim=True))
    if weight is None:
        loss_corr = (1 - corr).mean()
    else:
        loss_corr = ((1 - corr) * weight).sum()
    # loss_corr = 1 - ((predict - predict.mean()) * (target - target.mean())).mean() / predict.std() / target.std()
    if weight is None:
        loss_l2 = torch.mean((predict - target) ** 2)
        loss_l1 = torch.mean(torch.abs(predict - target))
        # loss_mape = torch.mean(torch.abs((predict - target) / target))
        loss_mape = torch.zeros(1, dtype=torch.float32, device='cuda')
    else:
        # L2
        loss_l2 = torch.mean((predict - target) ** 2, dim=0, keepdim=True)
        loss_l2 = (loss_l2 * weight).sum()

        # L1
        loss_l1 = torch.mean(torch.abs(predict - target), dim=0, keepdim=True)
        loss_l1 = (loss_l1 * weight).sum()

        # MAPE
        # loss_mape = torch.mean(torch.abs((predict - target) / target), dim=0, keepdim=True)
        # loss_mape = (loss_mape * weight).sum()

        # SMAPE
        # loss_smape = (torch.abs(predict - target) * 2) / (torch.abs(predict) + torch.abs(target))
        # loss_smape = torch.mean(loss_smape, dim=0, keepdim=True)
        # loss_mape = (loss_smape * weight).sum()

        loss_mape = torch.zeros(1, dtype=torch.float32, device='cuda')
    return loss_corr, loss_l2, loss_l1, loss_mape


def compute_phi_smooth_loss(phi_3d_orig, neigh_orders, n_vertex, grad_filter, device='cuda'):
    if grad_filter is None:
        grad_filter = torch.ones((7, 1), dtype=torch.float32, device=device)
        grad_filter[6] = -6
    loss_smooth = \
        torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:, [0]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
        torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:, [1]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
        torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:, [2]][neigh_orders].view(n_vertex, 7), grad_filter))
    loss_smooth = torch.mean(loss_smooth)
    return loss_smooth


def compute_phi_consistency_loss(phi_3d_0_to_1, phi_3d_1_orig, phi_3d_1_to_2, phi_3d_2_orig,
                                 phi_3d_0_to_2, merge_index):
    loss_phi_consistency = torch.mean(torch.abs(phi_3d_0_to_1[merge_index[7]] - phi_3d_1_orig[merge_index[7]])) + \
                           torch.mean(torch.abs(phi_3d_1_to_2[merge_index[8]] - phi_3d_2_orig[merge_index[8]])) + \
                           torch.mean(torch.abs(phi_3d_0_to_2[merge_index[9]] - phi_3d_2_orig[merge_index[9]]))
    return loss_phi_consistency


def compute_phi_loss(phi_3d_0_to_1, phi_3d_1_orig, phi_3d_1_to_2, phi_3d_2_orig,
                     phi_3d_0_to_2, phi_3d_orig, merge_index, neigh_orders, n_vertex,
                     grad_filter):
    loss_phi_consistency = torch.mean(torch.abs(phi_3d_0_to_1[merge_index[7]] - phi_3d_1_orig[merge_index[7]])) + \
                           torch.mean(torch.abs(phi_3d_1_to_2[merge_index[8]] - phi_3d_2_orig[merge_index[8]])) + \
                           torch.mean(torch.abs(phi_3d_0_to_2[merge_index[9]] - phi_3d_2_orig[merge_index[9]]))
    loss_smooth = \
        torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:, [0]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
        torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:, [1]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
        torch.abs(torch.mm(phi_3d_orig[0:n_vertex][:, [2]][neigh_orders].view(n_vertex, 7), grad_filter))
    loss_smooth = torch.mean(loss_smooth)
    return loss_phi_consistency, loss_smooth


def cal_loss_norigid(fixed_xyz, moved_xyz, fixed, moving, config, phis=None, sulc_curv_weight=None):
    neigh_orders = config['neigh_orders']
    n_vertex = config['n_vertex']
    phi_3d_orig = phis[0]

    # resample
    moved = resample_sphere_surface_barycentric(moved_xyz, fixed_xyz, moving)
    # calculate sim loss
    loss_corr, loss_l2, loss_l1, loss_mape = compute_sim_loss(moved, fixed, weight=sulc_curv_weight)
    # calculate phi loss
    loss_phi_smooth = compute_phi_smooth_loss(phi_3d_orig, neigh_orders, n_vertex, config['grad_filter'])

    loss = \
        config['weight_l2'] * loss_l2 + \
        config['weight_l1'] * loss_l1 + \
        config['weight_corr'] * loss_corr
    if config['weight_smooth'] != 0:
        loss += config['weight_smooth'] * loss_phi_smooth

    return loss, loss_corr.item(), loss_l2.item(), loss_l1.item(), loss_phi_smooth.item()


def cal_loss_rigid(fixed_xyz, moved_xyz, fixed, moving,
                   weight_corr=0, weight_l2=0, weight_l1=16, weight_data=None):

    # resample
    moved = resample_sphere_surface_barycentric(moved_xyz, fixed_xyz, moving)

    # calculate sim loss
    loss_corr, loss_l2, loss_l1, _ = compute_sim_loss(moved, fixed, weight=weight_data)

    loss = weight_corr * loss_corr + weight_l2 * loss_l2 + weight_l1 * loss_l1

    return loss, loss_corr.item(), loss_l2.item(), loss_l1.item()


class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False, device='cuda'):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.shape[0]
        self.nf = faces.shape[0]
        self.average = average
        laplacian = torch.zeros(self.nv, self.nv, dtype=torch.float32)

        # faces = faces.int().numpy()
        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            ii = laplacian[i, i].clone()
            laplacian[i, :] /= ii

        self.laplacian = laplacian.to(device)
        # self.laplacian = torch.from_numpy(laplacian).to(device)

    def forward(self, x):
        if x.shape[1] == 163842:
            x = x[:, :40962, :]
            # x = x[:, :10242, :]
        # batch_size = x.size(0)
        batch_size = 1
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(len(x.shape))[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x


class FlattenLoss(nn.Module):
    def __init__(self, faces, average=False, device='cuda'):
        super(FlattenLoss, self).__init__()
        self.nf = faces.shape[0]
        self.average = average

        # faces = faces.numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            for face in faces:
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.v0s = torch.from_numpy(v0s).to(device)
        self.v1s = torch.from_numpy(v1s).to(device)
        self.v2s = torch.from_numpy(v2s).to(device)
        self.v3s = torch.from_numpy(v3s).to(device)

    def execute(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps)).unsqueeze(-1)
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps)).unsqueeze(-1)
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(len(cos.shape))[1:])
        loss = (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss
