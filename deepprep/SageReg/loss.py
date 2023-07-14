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
from torch_scatter import scatter_mean

from utils.interp import resample_sphere_surface_barycentric
from utils.negative_area_triangle import negative_area
from utils.auxi_data import get_distance_by_points_num


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


def compute_sim_loss_np(predict, target, weight=None):
    corr_top = ((predict - predict.mean(axis=0, keepdims=True)) * (target - target.mean(axis=0, keepdims=True))).mean(axis=0, keepdims=True)
    corr_bottom = (predict.std(axis=0, keepdims=True) * target.std(axis=0, keepdims=True))
    corr = corr_top / corr_bottom
    if weight is None:
        loss_corr = (1 - corr).mean()
    else:
        loss_corr = ((1 - corr) * weight).sum()
    if weight is None:
        loss_l2 = np.mean((predict - target) ** 2)
        loss_l1 = np.mean(np.abs(predict - target))
    else:
        # L2
        loss_l2 = np.mean((predict - target) ** 2, axis=0, keepdims=True)
        loss_l2 = (loss_l2 * weight).sum()

        # L1
        loss_l1 = np.mean(np.abs(predict - target), axis=0, keepdims=True)
        loss_l1 = (loss_l1 * weight).sum()

    return loss_corr, loss_l2, loss_l1


def compute_sim_loss(predict, target, weight=None):

    if weight is None:
        corr_top = ((predict - predict.mean(dim=0, keepdim=True)) * (target - target.mean(dim=0, keepdim=True))).mean(
            dim=0, keepdim=True)
        corr_bottom = (predict.std(dim=0, keepdim=True) * target.std(dim=0, keepdim=True))
        corr = corr_top / corr_bottom
        loss_corr = (1 - corr).mean()
        loss_l2 = torch.mean((predict - target) ** 2)
        loss_l1 = torch.mean(torch.abs(predict - target))
    else:
        # # loss_corr = ((1 - corr) * weight).sum()
        # # L2
        # loss_l2 = torch.mean((predict - target) ** 2, dim=0, keepdim=True)
        # loss_l2 = (loss_l2 * weight).sum()
        #
        # # L1
        # loss_l1 = torch.mean(torch.abs(predict - target), dim=0, keepdim=True)
        # loss_l1 = (loss_l1 * weight).sum()

        corr_top = ((predict - predict.mean(dim=0, keepdim=True)) * (target - target.mean(dim=0, keepdim=True)))
        corr_bottom = (predict.std(dim=0, keepdim=True) * target.std(dim=0, keepdim=True))
        corr = corr_top / corr_bottom

        loss_corr = 1 - corr.mean()

        # L2
        loss_l2 = (predict - target) ** 2
        w = torch.sum(loss_l2) / torch.sum(loss_l2 / weight)
        loss_l2 = (loss_l2 * w / weight).mean()

        # L1
        loss_l1 = torch.abs(predict - target)
        loss_l1 = (loss_l1 * weight).mean()

    return loss_corr, loss_l2, loss_l1


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


def cal_loss_sim(fixed_xyz, moved_xyz, fixed, moving, config, sulc_curv_weight=None):
    # resample
    moved = resample_sphere_surface_barycentric(moved_xyz, fixed_xyz, moving, device='cuda')
    # calculate sim loss
    loss_corr, loss_l2, loss_l1 = compute_sim_loss(moved, fixed, weight=sulc_curv_weight)
    loss = \
        config['weight_l2'] * loss_l2 + \
        config['weight_l1'] * loss_l1 + \
        config['weight_corr'] * loss_corr
    return loss, loss_corr.item(), loss_l2.item(), loss_l1.item()


def cal_loss_norigid(fixed_xyz, moved_xyz, fixed, moving, config, phis=None, sulc_curv_weight=None):
    neigh_orders = config['neigh_orders']
    n_vertex = config['n_vertex']

    # resample
    moved = resample_sphere_surface_barycentric(moved_xyz, fixed_xyz, moving, device='cuda')
    # calculate sim loss
    loss_corr, loss_l2, loss_l1 = compute_sim_loss(moved, fixed, weight=sulc_curv_weight)
    # calculate phi loss
    loss_phi_smooth = compute_phi_smooth_loss(phis, neigh_orders, n_vertex, config['grad_filter'])

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
    loss_corr, loss_l2, loss_l1 = compute_sim_loss(moved, fixed, weight=weight_data)

    loss = weight_corr * loss_corr + weight_l2 * loss_l2 + weight_l1 * loss_l1

    return loss, loss_corr.item(), loss_l2.item(), loss_l1.item()


class LaplacianSmoothingLoss(nn.Module):
    def __init__(self, faces, xyz=None, rate=False, device='cuda'):
        super(LaplacianSmoothingLoss, self).__init__()
        self.nf = faces.shape[0]
        self.rate = rate

        from utils.smooth import get_edge_index
        edge_index = get_edge_index(faces.cpu().numpy(), device=device)
        self.row, self.col = edge_index

        if xyz is not None:
            self.nv = xyz.shape[0]
            self.distance = get_distance_by_points_num(self.nv)
            xyz = xyz * 100
            xyz_mean = scatter_mean(xyz[self.col], self.row, dim=0, dim_size=xyz.size(0))
            xyz_mean = xyz_mean / torch.norm(xyz_mean, dim=1, keepdim=True) * 100
            if self.rate:
                self.xyz_dif = torch.norm(xyz - xyz_mean, dim=1, keepdim=True)
            else:
                self.xyz_dif = (xyz - xyz_mean).abs()
        else:
            self.xyz_dif = None

    def forward(self, x, mean=True, norm=True):
        x = x * 100

        x_mean = scatter_mean(x[self.col], self.row, dim=0, dim_size=x.size(0))

        if norm:
            x_mean = x_mean / torch.norm(x_mean, dim=1, keepdim=True) * 100
            x_dif = (x - x_mean).abs()
            if self.rate:
                x_dif = torch.norm(x - x_mean, dim=1, keepdim=True)
                dif = (x_dif - self.xyz_dif).abs() / (self.distance / 2)
                # print(dif.max().item(), dif.min().item())
            else:
                dif = (x_dif - self.xyz_dif).abs()
        else:
            dif = (x - x_mean).abs()
        if mean:
            return dif.mean()
        else:
            return dif.sum()


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, true, pred, weight):
        loss_corr, loss_l2, loss_l1 = compute_sim_loss(true, pred, weight=weight)

        return loss_corr, loss_l2, loss_l1


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, true, pred):

        top = 2 * (true * pred).sum(dim=0)
        bottom = torch.clamp((true + pred).sum(dim=0), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1 - dice


class NegativeAreaLoss(nn.Module):
    def __init__(self, faces, xyz, device='cuda'):
        super(NegativeAreaLoss, self).__init__()
        self.nf = faces.shape[0]
        self.faces = faces.to(device)

        self.init_area = negative_area(self.faces, xyz * 100)

    def forward(self, x, mean=True):

        area = negative_area(self.faces, x * 100)
        zero = torch.zeros(1).to(x.device)
        dif = torch.max(-area, zero)

        if mean:
            me = dif.mean()
            return me
        else:
            return dif.sum()


class AreaLoss(nn.Module):
    def __init__(self, faces, xyz, device='cuda'):
        super(AreaLoss, self).__init__()
        self.nf = faces.shape[0]
        self.faces = faces.to(device)

        self.init_area = negative_area(self.faces, xyz).abs()

    def forward(self, x, mean=True):

        area = negative_area(self.faces, x).abs()
        dif = 1 - (area / self.init_area)
        dif = dif.abs()

        if mean:
            return dif.mean()
        else:
            return dif.sum()


class AngleLoss(nn.Module):
    def __init__(self, faces, xyz, device='cuda'):
        super(AngleLoss, self).__init__()
        self.nf = faces.shape[0]
        self.faces = faces.to(device)

        self.init_cos_theta, self.init_angle = self.angle(xyz)

    @staticmethod
    def vector_angle(vector_a, vector_b):
        norm_a = torch.norm(vector_a, dim=1)
        norm_b = torch.norm(vector_b, dim=1)
        axb = torch.sum(vector_a * vector_b, dim=1)
        cos_theta = axb / (norm_a * norm_b)
        angle = torch.acos(cos_theta)
        angle_d = angle * 180 / np.pi
        return cos_theta, angle_d

    def angle(self, xyz):
        """
        获取两个向量的夹角
        """
        xyz_a, xyz_b, xyz_c = get_face_coords(self.faces, xyz)
        cos_theta_a, angle_a = self.vector_angle(xyz_b - xyz_a, xyz_c - xyz_a)
        cos_theta_b, angle_b = self.vector_angle(xyz_a - xyz_b, xyz_c - xyz_b)
        cos_theta_c, angle_c = self.vector_angle(xyz_a - xyz_c, xyz_b - xyz_c)
        angle = torch.cat([angle_a.unsqueeze(1), angle_b.unsqueeze(1), angle_c.unsqueeze(1)], dim=1)
        cos_theta = torch.cat([cos_theta_a.unsqueeze(1), cos_theta_b.unsqueeze(1), cos_theta_c.unsqueeze(1)], dim=1)

        return cos_theta, angle

    def forward(self, x, mean=True):
        # fsaverage6 angle range = (0.3013, 0.5935)  (53.59, 72.46)

        cos_theta, angle = self.angle(x)

        dif = 1 - (cos_theta / self.init_cos_theta)
        dif = dif.abs()

        if mean:
            return dif.mean()
        else:
            return dif.sum()


if __name__ == '__main__':
    import nibabel as nib
    fsaverage6_sphere = '/usr/local/freesurfer/subjects/fsaverage6/surf/lh.sphere'
    msm_sphere_reg = '/mnt/ngshare/SurfReg/Data_TrainResult/testmultilevel/NAMIC_SD_MSM/sub32/surf/lh.MSM2018.sphere.reg'
    sd_sphere_reg = '/mnt/ngshare/SurfReg/Data_TrainResult/testmultilevel/NAMIC_SD_MSM/sub32/surf/lh.SDdefault.sphere.reg'
    device = 'cuda'

    xyz_fixed, faces_fixed = nib.freesurfer.read_geometry(fsaverage6_sphere)
    xyz_fixed = torch.from_numpy(xyz_fixed.astype(float)).to(device) / 100
    faces_fixed = torch.from_numpy(faces_fixed.astype(int)).to(device)
    fs6_angle_loss = AngleLoss(faces_fixed, xyz_fixed, device=device)

    xyz_fixed, faces_fixed = nib.freesurfer.read_geometry(msm_sphere_reg)
    xyz_fixed = torch.from_numpy(xyz_fixed.astype(float)).to(device) / 100
    faces_fixed = torch.from_numpy(faces_fixed.astype(int)).to(device)
    msm_angle_loss = AngleLoss(faces_fixed, xyz_fixed, device=device)

    xyz_fixed, faces_fixed = nib.freesurfer.read_geometry(sd_sphere_reg)
    xyz_fixed = torch.from_numpy(xyz_fixed.astype(float)).to(device) / 100
    faces_fixed = torch.from_numpy(faces_fixed.astype(int)).to(device)
    sd_angle_loss = AngleLoss(faces_fixed, xyz_fixed, device=device)

    print()