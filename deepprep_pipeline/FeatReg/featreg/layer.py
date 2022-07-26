import torch
import numpy as np
import torch.nn as nn
from featreg.utils.interp import resample_sphere_surface_barycentric


def diffeomorp(moving_xyz, phi_3d, num_composition=6):
    warped_vertices = moving_xyz + phi_3d
    warped_vertices = warped_vertices / (torch.norm(warped_vertices, dim=1, keepdim=True).repeat(1, 3))
    # compute exp
    for i in range(num_composition):
        warped_vertices = resample_sphere_surface_barycentric(moving_xyz, warped_vertices, warped_vertices)
        warped_vertices = warped_vertices / (torch.norm(warped_vertices, dim=1, keepdim=True).repeat(1, 3))

    return warped_vertices


class repa_conv_layer(nn.Module):
    """Define the convolutional layer on icosahedron discretized sphere using
    rectagular filter in tangent plane

    Parameters:
            in_feats (int) - - input features/channels
            out_feats (int) - - output features/channels

    Input:
        N x in_feats, tensor
    Return:
        N x out_feats, tensor
    """

    def __init__(self, in_feats, out_feats, neigh_indices, neigh_weights):
        super(repa_conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_indices = neigh_indices.reshape(-1) - 1
        self.weight = nn.Linear(25 * in_feats, out_feats)
        self.nodes_number = neigh_indices.shape[0]

        neigh_weights = np.reshape(np.tile(neigh_weights, self.in_feats),
                                   (neigh_weights.shape[0], neigh_weights.shape[1], 3, -1)).astype(np.float32)
        self.neigh_weights = torch.from_numpy(neigh_weights).cuda()

    def forward(self, x):
        mat = x[self.neigh_indices]
        mat = mat.view(self.nodes_number, 25, 3, -1)
        assert (mat.size() == torch.Size([self.nodes_number, 25, 3, self.in_feats]))

        assert (mat.size() == self.neigh_weights.size())

        x = torch.mul(mat, self.neigh_weights)
        x = torch.sum(x, 2).view(self.nodes_number, -1)
        assert (x.size() == torch.Size([self.nodes_number, 25 * self.in_feats]))

        out = self.weight(x)
        return out


class onering_conv_layer(nn.Module):
    """The convolutional layer on icosahedron discretized sphere using
    1-ring filter

    Parameters:
            in_feats (int) - - input features/channels
            out_feats (int) - - output features/channels

    Input:
        N x in_feats tensor
    Return:
        N x out_feats tensor
    """

    def __init__(self, in_feats, out_feats, neigh_orders, neigh_indices=None, neigh_weights=None):
        super(onering_conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders

        self.weight = nn.Linear(7 * in_feats, out_feats)

    def forward(self, x):
        mat = x[self.neigh_orders].view(len(x), 7 * self.in_feats)

        out_features = self.weight(mat)
        return out_features


class tworing_conv_layer(nn.Module):
    """The convolutional layer on icosahedron discretized sphere using
    2-ring filter

    Parameters:
            in_feats (int) - - input features/channels
            out_feats (int) - - output features/channels

    Input:
        N x in_feats tensor
    Return:
        N x out_feats tensor
    """

    def __init__(self, in_feats, out_feats, neigh_orders):
        super(tworing_conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders

        self.weight = nn.Linear(19 * in_feats, out_feats)

    def forward(self, x):
        mat = x[self.neigh_orders].view(len(x), 19 * self.in_feats)

        out_features = self.weight(mat)
        return out_features


class pool_layer(nn.Module):
    """
    The pooling layer on icosahedron discretized sphere using 1-ring filter

    Input:
        N x D tensor
    Return:
        ((N+6)/4) x D tensor

    """

    def __init__(self, neigh_orders, pooling_type='mean'):
        super(pool_layer, self).__init__()

        self.neigh_orders = neigh_orders
        self.pooling_type = pooling_type

    def forward(self, x):

        num_nodes = int((x.size()[0] + 6) / 4)
        feat_num = x.size()[1]
        x = x[self.neigh_orders[0:num_nodes * 7]].view(num_nodes, feat_num, 7)
        if self.pooling_type == "mean":
            x = torch.mean(x, 2)
        if self.pooling_type == "max":
            x = torch.max(x, 2)
            # assert(x[0].size() == torch.Size([num_nodes, feat_num]))
            # return x[0], x[1]
            return x[0]

        assert (x.size() == torch.Size([num_nodes, feat_num]))

        return x


class upconv_layer(nn.Module):
    """
    The transposed convolution layer on icosahedron discretized sphere using 1-ring filter

    Input:
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x out_feats, tensor

    """

    def __init__(self, in_feats, out_feats, upconv_top_index, upconv_down_index):
        super(upconv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.upconv_top_index = upconv_top_index
        self.upconv_down_index = upconv_down_index
        self.weight = nn.Linear(in_feats, 7 * out_feats)

    def forward(self, x):
        raw_nodes = x.size()[0]
        new_nodes = int(raw_nodes * 4 - 6)
        x = self.weight(x)
        x = x.view(len(x) * 7, self.out_feats)
        x1 = x[self.upconv_top_index]
        assert (x1.size() == torch.Size([raw_nodes, self.out_feats]))
        x2 = x[self.upconv_down_index].view(-1, self.out_feats, 2)
        x = torch.cat((x1, torch.mean(x2, 2)), 0)
        assert (x.size() == torch.Size([new_nodes, self.out_feats]))
        return x


class upsample_interpolation(nn.Module):
    """
    The upsampling layer on icosahedron discretized sphere using interpolation

    Input:
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x in_feats, tensor

    """

    def __init__(self, upsample_neighs_order):
        super(upsample_interpolation, self).__init__()

        self.upsample_neighs_order = upsample_neighs_order

    def forward(self, x):
        num_nodes = x.size()[0] * 4 - 6
        feat_num = x.size()[1]
        x1 = x[self.upsample_neighs_order].view(num_nodes - x.size()[0], feat_num, 2)
        x1 = torch.mean(x1, 2)
        x = torch.cat((x, x1), 0)

        return x


class upsample_fixindex(nn.Module):
    """
    The upsampling layer on icosahedron discretized sphere using fixed indices 0,
    padding new vertices with 0

    Input:
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x in_feats, tensor

    """

    def __init__(self, upsample_neighs_order):
        super(upsample_fixindex, self).__init__()

        self.upsample_neighs_order = upsample_neighs_order

    def forward(self, x):
        num_nodes = x.size()[0] * 4 - 6
        feat_num = x.size()[1]
        x1 = torch.zeros(num_nodes - x.size()[0], feat_num).cuda()
        x = torch.cat((x, x1), 0)

        return x


class upsample_maxindex(nn.Module):
    """
    The upsampling layer on icosahedron discretized sphere using max indices.

    Input:
        N x in_feats, tensor
    Return:
        ((Nx4)-6) x in_feats, tensor

    """

    def __init__(self, num_nodes, neigh_orders):
        super(upsample_maxindex, self).__init__()

        self.num_nodes = num_nodes
        self.neigh_orders = neigh_orders

    def forward(self, x, max_index):
        raw_nodes, feat_num = x.size()
        assert (max_index.size() == x.size())
        x = x.view(-1)

        y = torch.zeros(self.num_nodes, feat_num).to(torch.device("cuda"))
        column_ref = torch.zeros(raw_nodes, feat_num)
        for i in range(raw_nodes):
            column_ref[i, :] = i * 7 + max_index[i, :]
        column_index = self.neigh_orders[column_ref.view(-1).long()]
        column_index = torch.from_numpy(column_index).long()
        row_index = np.floor(np.linspace(0.0, float(feat_num), num=raw_nodes * feat_num))
        row_index[-1] = row_index[-1] - 1
        row_index = torch.from_numpy(row_index).long()
        y[column_index, row_index] = x

        return y