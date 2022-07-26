import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import onering_conv_layer, pool_layer, upconv_layer
from ring_neighbor import Get_neighs_order, Get_upconv_index, get_z_weight


class down_block(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2

    """

    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first=False):
        super(down_block, self).__init__()

        #        Batch norm version
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders, 'mean'),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        # batch norm version
        x = self.block(x)

        return x


class up_block(nn.Module):
    """Define the upsamping block in spherica unet
    upconv => (conv => BN => ReLU) * 2

    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders

    """

    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()

        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)

        # batch norm version
        self.double_conv = nn.Sequential(
            conv_layer(in_ch, out_ch, neigh_orders),
            nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            conv_layer(out_ch, out_ch, neigh_orders),
            nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1)
        x = self.double_conv(x)

        return x


class SphricalUnet(nn.Module):
    """Define the Spherical UNet structure

    """

    def __init__(self, in_ch, out_ch, level=7, n_res=5, rotated=0):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
            level (int) - - input surface's icosahedron level. default: 7, for 40962 vertices
                            2:42, 3:162, 4:642, 5:2562, 6:10242, 7:40962, 8:163842
            n_res (int) - - the total resolution levels of u-net, default: 5
            rotated (int) - -  the sphere is original (0), rotated 90 degrees along y axis (0), or
                               90 degrees along x axis (1)
        """
        super(SphricalUnet, self).__init__()

        assert (level - n_res) >= 1, "number of resolution levels in unet should be at least 1 smaller than iput level"
        assert n_res >= 2, "number of resolution levels should be larger than 2"
        assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"

        neigh_orders = Get_neighs_order(rotated)
        neigh_orders = neigh_orders[8 - level:8 - level + n_res]
        upconv_indices = Get_upconv_index(rotated)
        upconv_indices = upconv_indices[16 - 2 * level:16 - 2 * level + (n_res - 1) * 2]

        chs = [in_ch]
        for i in range(n_res):
            chs.append(2 ** i * 8 * 8)

        conv_layer = onering_conv_layer

        self.down = nn.ModuleList([])
        for i in range(n_res):
            if i == 0:
                self.down.append(down_block(conv_layer, chs[i], chs[i + 1], neigh_orders[i], None, True))
            else:
                self.down.append(down_block(conv_layer, chs[i], chs[i + 1], neigh_orders[i], neigh_orders[i - 1]))

        self.up = nn.ModuleList([])
        for i in range(n_res - 1):
            self.up.append(up_block(conv_layer, chs[n_res - i], chs[n_res - 1 - i],
                                    neigh_orders[n_res - 2 - i], upconv_indices[(n_res - 2 - i) * 2],
                                    upconv_indices[(n_res - 2 - i) * 2 + 1]))

        self.outc = nn.Linear(chs[1], out_ch)

        self.n_res = n_res

    def forward(self, x):
        xs = [x]
        for i in range(self.n_res):
            xs.append(self.down[i](xs[i]))

        x = xs[-1]
        for i in range(self.n_res - 1):
            x = self.up[i](x, xs[self.n_res - 1 - i])

        x = self.outc(x)  # N * 2
        return x


class SphericalNet(nn.Module):
    """Define the Spherical UNet structure

    """

    def __init__(self, in_ch, out_ch=3, level=7, n_res=8, rotated=0):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
            level (int) - - input surface's icosahedron level. default: 7, for 40962 vertices
                            1：12 2:42, 3:162, 4:642, 5:2562, 6:10242, 7:40962, 8:163842
            n_res (int) - - the total resolution levels of u-net, default: 5
            rotated (int) - -  the sphere is original (0), rotated 90 degrees along y axis (0), or
                               90 degrees along x axis (1)
        """
        super(SphericalNet, self).__init__()

        assert n_res >= 2, "number of resolution levels should be larger than 2"
        assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"

        neigh_orders = Get_neighs_order(rotated)[8 - n_res:]
        chs = [in_ch]
        for i in range(n_res):
            chs.append(2 ** i * 8)

        conv_layer = onering_conv_layer

        self.down = nn.ModuleList([])
        for i in range(n_res):
            if i == 0:
                self.down.append(down_block(conv_layer, chs[i], chs[i + 1], neigh_orders[i], None, True))
            else:
                self.down.append(down_block(conv_layer, chs[i], chs[i + 1], neigh_orders[i], neigh_orders[i - 1]))

        self.outr = nn.Linear(12, out_ch)

        self.n_res = n_res

    def forward(self, x):
        xs = [x]
        for i in range(self.n_res):
            xs.append(self.down[i](xs[i]))

        # rigid rotate matrix
        r = xs[-1]
        # for i in range(8 - self.n_res):
        #     r = self.rigid[i](r)
        r = r.unsqueeze(0)
        r = F.adaptive_avg_pool1d(r, 1)
        r = self.outr(r.squeeze(2))
        return r


class SNetRMRR(nn.Module):
    """
    SphericalNetRotateMatrixRigidRegistration
    """

    def __init__(self, in_ch=2, out_ch=3, level=8, n_res=8):
        """ Initialize the model.

        Parameters:
        """
        super(SNetRMRR, self).__init__()
        self.rotate_type = "Euler_Angle"  # 4元组 or 欧拉角  Quaternion or Euler_Angle

        in_ch = in_ch
        out_ch = out_ch
        level = level
        n_res = n_res

        self.net = SphericalNet(in_ch, out_ch, level, n_res, 0)

    def rotate_matrix(self, z1, y2, x3):
        """
        first x3, then y2, lastly z1
        """
        return torch.cat(
            [torch.cat([torch.cos(z1) * torch.cos(y2),
                        torch.cos(z1) * torch.sin(y2) * torch.sin(x3) - torch.sin(z1) * torch.cos(x3),
                        torch.sin(z1) * torch.sin(x3) + torch.cos(z1) * torch.cos(x3) * torch.sin(y2)]),

             torch.cat([torch.cos(y2) * torch.sin(z1),
                        torch.cos(z1) * torch.cos(x3) + torch.sin(z1) * torch.sin(y2) * torch.sin(x3),
                        torch.cos(x3) * torch.sin(z1) * torch.sin(y2) - torch.cos(z1) * torch.sin(x3)]),

             torch.cat([-torch.sin(y2),
                        torch.cos(y2) * torch.sin(x3),
                        torch.cos(y2) * torch.cos(x3)])], dim=1)

    @staticmethod
    def rigid_rotate_matrix(moving_xyz, rigid_rotate_matrix):
        """
        应用旋转矩阵
        """
        moving_xyz_rigid = torch.mm(rigid_rotate_matrix, torch.transpose(moving_xyz, 0, 1))
        moved_xyz = torch.transpose(moving_xyz_rigid, 0, 1)  # 新的顶点坐标_3D
        return moved_xyz

    def forward(self, x, moving_xyz, val=False):
        # registraion starts
        rigid_euler_angle = self.net(x)

        # rigid
        rigid_rotate_matrix = self.rotate_matrix(rigid_euler_angle[:, [0]],
                                                 rigid_euler_angle[:, [1]],
                                                 rigid_euler_angle[:, [2]])
        moving_xyz_rigid = torch.mm(rigid_rotate_matrix, torch.transpose(moving_xyz, 0, 1))
        moved_xyz = torch.transpose(moving_xyz_rigid, 0, 1)  # 新的顶点坐标_3D

        return moved_xyz, rigid_rotate_matrix


class S3UnetRotate(nn.Module):
    """
    3个Spherical Unet组合
    """

    def __init__(self, in_ch=2, out_ch=3, level=8, n_res=6, rotate_type='Euler_Angle',
                 merge_index=None, device='cuda'):
        """ Initialize the model.

        Parameters:
        """
        super(S3UnetRotate, self).__init__()
        self.rotate_type = rotate_type  # 4元组 or 欧拉角  Quaternion or Euler_Angle
        self.level = level  # 第几个等级

        if self.rotate_type == "Quaternion":
            out_ch = 4
        else:
            out_ch = 3

        self.unet0 = SphricalUnet(in_ch, out_ch, level, n_res, 0)
        self.unet1 = SphricalUnet(in_ch, out_ch, level, n_res, 1)
        self.unet2 = SphricalUnet(in_ch, out_ch, level, n_res, 2)

        self.merge_index = merge_index
        self.device = device

    def rotate_matrix(self, phi_3d_xyz):
        """
        first x3, then y2, lastly z1
        # torch.cat(
        #     [[torch.cos(g) * torch.cos(b), torch.cos(g) * torch.sin(b) * torch.sin(a) - torch.sin(g) * torch.cos(a),
        #       torch.sin(g) * torch.sin(a) + torch.cos(g) * torch.cos(a) * torch.sin(b)],
        #
        #      [torch.cos(b) * torch.sin(g), torch.cos(g) * torch.cos(a) + torch.sin(g) * torch.sin(b) * torch.sin(a),
        #       torch.cos(a) * torch.sin(g) * torch.sin(b) - torch.cos(g) * torch.sin(a)],
        #
        #      [-torch.sin(b), torch.cos(b) * torch.sin(a), torch.cos(b) * torch.cos(a)]])
        """
        a = phi_3d_xyz[:, [0]]
        b = phi_3d_xyz[:, [1]]
        g = phi_3d_xyz[:, [2]]
        r1 = torch.cat(
            [torch.cos(g) * torch.cos(b), torch.cos(g) * torch.sin(b) * torch.sin(a) - torch.sin(g) * torch.cos(a),
             torch.sin(g) * torch.sin(a) + torch.cos(g) * torch.cos(a) * torch.sin(b)], dim=1)
        r2 = torch.cat(
            [torch.cos(b) * torch.sin(g), torch.cos(g) * torch.cos(a) + torch.sin(g) * torch.sin(b) * torch.sin(a),
             torch.cos(a) * torch.sin(g) * torch.sin(b) - torch.cos(g) * torch.sin(a)], dim=1)
        r3 = torch.cat(
            [-torch.sin(b), torch.cos(b) * torch.sin(a), torch.cos(b) * torch.cos(a)], dim=1)

        # rotate_metrix = torch.cat(
        #     [r1, r2, r3], dim=2)
        return r1, r2, r3

    def forward(self, x, moving_xyz, val=False):
        rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, z_weight_0, z_weight_1, z_weight_2, \
        index_01, index_12, index_02, index_0_0, index_1_0, index_2_0, index_double_02, \
        index_double_12, index_double_01, index_triple_computed = self.merge_index

        # registraion starts
        # tangent vector field phi
        phi_3d_0_orig = self.unet0(x) / 50.0
        phi_3d_1_orig = self.unet1(x) / 50.0
        phi_3d_2_orig = self.unet2(x) / 50.0

        """ deformation consistency  """
        phi_3d_0_to_1 = torch.mm(rot_mat_01, torch.transpose(phi_3d_0_orig, 0, 1))
        phi_3d_0_to_1 = torch.transpose(phi_3d_0_to_1, 0, 1)
        phi_3d_1_to_2 = torch.mm(rot_mat_12, torch.transpose(phi_3d_1_orig, 0, 1))
        phi_3d_1_to_2 = torch.transpose(phi_3d_1_to_2, 0, 1)
        phi_3d_0_to_2 = torch.mm(rot_mat_02, torch.transpose(phi_3d_0_orig, 0, 1))
        phi_3d_0_to_2 = torch.transpose(phi_3d_0_to_2, 0, 1)

        """ first merge """
        phi_3d = torch.zeros(len(moving_xyz), 3).to(self.device)
        phi_3d[index_double_02] = (phi_3d_0_to_2[index_double_02] + phi_3d_2_orig[index_double_02]) / 2.0
        phi_3d[index_double_12] = (phi_3d_1_to_2[index_double_12] + phi_3d_2_orig[index_double_12]) / 2.0
        tmp = (phi_3d_0_to_1[index_double_01] + phi_3d_1_orig[index_double_01]) / 2.0
        phi_3d[index_double_01] = torch.transpose(torch.mm(rot_mat_12, torch.transpose(tmp, 0, 1)), 0, 1)
        phi_3d[index_triple_computed] = (phi_3d_1_to_2[index_triple_computed] + phi_3d_2_orig[index_triple_computed] +
                                         phi_3d_0_to_2[index_triple_computed]) / 3.0
        phi_3d_orig = torch.transpose(torch.mm(rot_mat_20, torch.transpose(phi_3d, 0, 1)), 0, 1)
        # print(torch.norm(phi_3d_orig,dim=1).max().item())

        # get moved_xyz by rotate metrix
        r1, r2, r3 = self.rotate_matrix(phi_3d_orig)
        moved_x = torch.sum(moving_xyz * r1, dim=1)
        moved_y = torch.sum(moving_xyz * r2, dim=1)
        moved_z = torch.sum(moving_xyz * r3, dim=1)
        moving_warp_phi_3d = torch.cat((moved_x.unsqueeze(1), moved_y.unsqueeze(1), moved_z.unsqueeze(1)), dim=1)

        if val:
            return moving_warp_phi_3d, phi_3d_orig
        else:
            return moving_warp_phi_3d, phi_3d_orig, phi_3d_0_to_1, phi_3d_1_orig, phi_3d_1_to_2, \
                   phi_3d_2_orig, phi_3d_0_to_2


class SUnetRotate(nn.Module):
    """
    1个Spherical Unet
    """

    def __init__(self, in_ch=2, out_ch=3, level=8, n_res=6, rotate_type='Euler_Angle'):
        """ Initialize the model.

        Parameters:
        """
        super(SUnetRotate, self).__init__()
        self.rotate_type = rotate_type  # 4元组 or 欧拉角  Quaternion or Euler_Angle

        if self.rotate_type == "Quaternion":
            out_ch = 4
        else:
            out_ch = 3
        self.level = level
        self.n_res = n_res

        self.unet = SphricalUnet(in_ch, out_ch, level, n_res, 0)

    def rotate_matrix(self, phi_3d_xyz):
        """
        first x3, then y2, lastly z1
        # torch.cat(
        #     [[torch.cos(g) * torch.cos(b), torch.cos(g) * torch.sin(b) * torch.sin(a) - torch.sin(g) * torch.cos(a),
        #       torch.sin(g) * torch.sin(a) + torch.cos(g) * torch.cos(a) * torch.sin(b)],
        #
        #      [torch.cos(b) * torch.sin(g), torch.cos(g) * torch.cos(a) + torch.sin(g) * torch.sin(b) * torch.sin(a),
        #       torch.cos(a) * torch.sin(g) * torch.sin(b) - torch.cos(g) * torch.sin(a)],
        #
        #      [-torch.sin(b), torch.cos(b) * torch.sin(a), torch.cos(b) * torch.cos(a)]])
        """
        a = phi_3d_xyz[:, [0]]
        b = phi_3d_xyz[:, [1]]
        g = phi_3d_xyz[:, [2]]
        r1 = torch.cat(
            [torch.cos(g) * torch.cos(b), torch.cos(g) * torch.sin(b) * torch.sin(a) - torch.sin(g) * torch.cos(a),
             torch.sin(g) * torch.sin(a) + torch.cos(g) * torch.cos(a) * torch.sin(b)], dim=1)
        r2 = torch.cat(
            [torch.cos(b) * torch.sin(g), torch.cos(g) * torch.cos(a) + torch.sin(g) * torch.sin(b) * torch.sin(a),
             torch.cos(a) * torch.sin(g) * torch.sin(b) - torch.cos(g) * torch.sin(a)], dim=1)
        r3 = torch.cat(
            [-torch.sin(b), torch.cos(b) * torch.sin(a), torch.cos(b) * torch.cos(a)], dim=1)

        # rotate_metrix = torch.cat(
        #     [r1, r2, r3], dim=2)
        return r1, r2, r3

    def forward(self, x, moving_xyz, val=False):
        # registraion starts
        # tangent vector field phi
        phi_3d_orig = self.unet(x)
        # phi_3d_orig, align_rotate_matrix = self.unet(x)

        # get moved_xyz by rotate metrix
        r1, r2, r3 = self.rotate_matrix(phi_3d_orig)
        moved_x = torch.sum(moving_xyz * r1, dim=1)
        moved_y = torch.sum(moving_xyz * r2, dim=1)
        moved_z = torch.sum(moving_xyz * r3, dim=1)
        moving_warp_phi_3d = torch.cat((moved_x.unsqueeze(1), moved_y.unsqueeze(1), moved_z.unsqueeze(1)), dim=1)

        # ### test the scaling and squaring layers
        # import math
        # from layer import diffeomorp
        # phi_3d_orig_diffe = phi_3d_orig
        # phi_3d = phi_3d_orig_diffe / math.pow(2, 6)
        # moving_warp_phi_3d = diffeomorp(moving_xyz, phi_3d,
        #                                 num_composition=6,
        #                                 )

        if val:
            return moving_warp_phi_3d, phi_3d_orig
        else:
            return moving_warp_phi_3d, phi_3d_orig, None, None, None, None, None
