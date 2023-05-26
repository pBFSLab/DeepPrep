import os
import glob
import numpy as np
import scipy.io as sio
import torch

from featreg.utils.vtk import read_vtk
from featreg.utils.utils import abspath


def get_neighbor_ring(n_vertex=163842, ring=1):
    ring1_file_path = os.path.join(abspath, f'neigh_indices/neighbor_{n_vertex}_rotated_0_ring1.npz')
    ring2_file_path = os.path.join(abspath, f'neigh_indices/neighbor_{n_vertex}_rotated_0_ring2.npz')
    ring3_file_path = os.path.join(abspath, f'neigh_indices/neighbor_{n_vertex}_rotated_0_ring3.npz')
    if ring == 1:
        npzfile = np.load(ring1_file_path)
        return npzfile['data']
    elif ring == 2:
        npzfile = np.load(ring2_file_path, allow_pickle=True)
        return npzfile['data']
    elif ring == 3:
        npzfile = np.load(ring3_file_path)
        return npzfile['data']
    elif ring == 12:
        ring_1 = get_neighbor_ring(n_vertex=n_vertex, ring=1)
        ring_2 = get_neighbor_ring(n_vertex=n_vertex, ring=2)
        return np.concatenate((ring_1, ring_2), axis=1)
    elif ring == 23:
        ring_2 = get_neighbor_ring(n_vertex=n_vertex, ring=2)
        ring_3 = get_neighbor_ring(n_vertex=n_vertex, ring=3)
        return np.concatenate((ring_2, ring_3), axis=1)
    elif ring == 123:
        ring_1 = get_neighbor_ring(n_vertex=n_vertex, ring=1)
        ring_2 = get_neighbor_ring(n_vertex=n_vertex, ring=2)
        ring_3 = get_neighbor_ring(n_vertex=n_vertex, ring=3)
        return np.concatenate((ring_1, ring_2, ring_3), axis=1)
    else:
        raise KeyError('ring is error')


def Get_neighs_order(rotated=0):
    neigh_orders_163842 = get_neighs_order(
        abspath + '/neigh_indices/adj_mat_order_163842_rotated_' + str(rotated) + '.mat')
    neigh_orders_40962 = get_neighs_order(
        abspath + '/neigh_indices/adj_mat_order_40962_rotated_' + str(rotated) + '.mat')
    neigh_orders_10242 = get_neighs_order(
        abspath + '/neigh_indices/adj_mat_order_10242_rotated_' + str(rotated) + '.mat')
    neigh_orders_2562 = get_neighs_order(abspath + '/neigh_indices/adj_mat_order_2562_rotated_' + str(rotated) + '.mat')
    neigh_orders_642 = get_neighs_order(abspath + '/neigh_indices/adj_mat_order_642_rotated_' + str(rotated) + '.mat')
    neigh_orders_162 = get_neighs_order(abspath + '/neigh_indices/adj_mat_order_162_rotated_' + str(rotated) + '.mat')
    neigh_orders_42 = get_neighs_order(abspath + '/neigh_indices/adj_mat_order_42_rotated_' + str(rotated) + '.mat')
    neigh_orders_12 = get_neighs_order(abspath + '/neigh_indices/adj_mat_order_12_rotated_' + str(rotated) + '.mat')

    return neigh_orders_163842, neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12


def get_neighs_order(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:, 0:6] = adj_mat_order - 1
    neigh_orders[:, 6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)

    return neigh_orders


def Get_upconv_index(rotated=0):
    upconv_top_index_163842, upconv_down_index_163842 = get_upconv_index(
        abspath + '/neigh_indices/adj_mat_order_163842_rotated_' + str(rotated) + '.mat')
    upconv_top_index_40962, upconv_down_index_40962 = get_upconv_index(
        abspath + '/neigh_indices/adj_mat_order_40962_rotated_' + str(rotated) + '.mat')
    upconv_top_index_10242, upconv_down_index_10242 = get_upconv_index(
        abspath + '/neigh_indices/adj_mat_order_10242_rotated_' + str(rotated) + '.mat')
    upconv_top_index_2562, upconv_down_index_2562 = get_upconv_index(
        abspath + '/neigh_indices/adj_mat_order_2562_rotated_' + str(rotated) + '.mat')
    upconv_top_index_642, upconv_down_index_642 = get_upconv_index(
        abspath + '/neigh_indices/adj_mat_order_642_rotated_' + str(rotated) + '.mat')
    upconv_top_index_162, upconv_down_index_162 = get_upconv_index(
        abspath + '/neigh_indices/adj_mat_order_162_rotated_' + str(rotated) + '.mat')

    return upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242, upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162


def get_upconv_index(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    adj_mat_order = adj_mat_order - 1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order) + 6) / 4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes - next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert (len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i - next_nodes) * 2 + j] = parent_nodes[j] * 7 + index

    return upconv_top_index, upconv_down_index


def get_upsample_order(n_vertex):
    n_last = int((n_vertex + 6) / 4)
    neigh_orders = get_neighs_order(abspath + '/neigh_indices/adj_mat_order_' + str(n_vertex) + '_rotated_0.mat')
    neigh_orders = neigh_orders.reshape(n_vertex, 7)
    neigh_orders = neigh_orders[n_last:, :]
    row, col = (neigh_orders < n_last).nonzero()
    assert len(row) == (n_vertex - n_last) * 2, "len(row) == (n_vertex - n_last)*2, error!"

    u, indices, counts = np.unique(row, return_index=True, return_counts=True)
    assert len(u) == n_vertex - n_last, "len(u) == n_vertex - n_last, error"
    assert u.min() == 0 and u.max() == n_vertex - n_last - 1, "u.min() == 0 and u.max() == n_vertex-n_last-1, error"
    assert (indices == np.asarray(list(range(
        n_vertex - n_last))) * 2).sum() == n_vertex - n_last, "(indices == np.asarray(list(range(n_vertex - n_last))) * 2).sum() == n_vertex - n_last, error"
    assert (counts == 2).sum() == n_vertex - n_last, "(counts == 2).sum() == n_vertex - n_last, error"

    upsample_neighs_order = neigh_orders[row, col]

    return upsample_neighs_order


def get_orthonormal_vectors(n_ver, rotated=0):
    """
    get the orthonormal vectors

    n_vec: int, number of vertices, 42,162,642,2562,10242,...
    rotated: 0: original, 1: rotate 90 degrees along y axis, 2: then rotate 90 degrees along z axis
    return orthonormal matrix, shape: n_vec * 3 * 2
    """
    assert type(n_ver) is int, "n_ver, the number of vertices should be int type"
    assert n_ver in [42, 162, 642, 2562, 10242, 40962,
                     163842], "n_ver, the number of vertices should be the one of [42,162,642,2562,10242,40962,163842]"
    assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"

    template = read_vtk(abspath + '/neigh_indices/sphere_' + str(n_ver) + '_rotated_' + str(rotated) + '.vtk')
    vertices = template['vertices'].astype(np.float64)

    x_0 = np.argwhere(vertices[:, 0] == 0)
    y_0 = np.argwhere(vertices[:, 1] == 0)
    inter_ind = np.intersect1d(x_0, y_0)

    En_1 = np.cross(np.array([0, 0, 1]), vertices)
    En_1[inter_ind] = np.array([1, 0, 0])
    En_2 = np.cross(vertices, En_1)

    En_1 = En_1 / np.repeat(np.sqrt(np.sum(En_1 ** 2, axis=1))[:, np.newaxis], 3,
                            axis=1)  # normalize to unit orthonormal vector
    En_2 = En_2 / np.repeat(np.sqrt(np.sum(En_2 ** 2, axis=1))[:, np.newaxis], 3,
                            axis=1)  # normalize to unit orthonormal vector
    En = np.transpose(np.concatenate((En_1[np.newaxis, :], En_2[np.newaxis, :]), 0), (1, 2, 0))

    return En


def get_patch_indices(n_vertex):
    """
    return all the patch indices and weights
    """
    indices_files = sorted(glob.glob(abspath + '/neigh_indices/*_indices.npy'))
    weights_files = sorted(glob.glob(abspath + '/neigh_indices/*_weights.npy'))

    assert len(indices_files) == len(weights_files), "indices files should have the same number as weights number"
    assert len(indices_files) == 163842, "Indices should have dimension 163842 "

    indices = [x.split('/')[-1].split('_')[0] for x in indices_files]
    weights = [x.split('/')[-1].split('_')[0] for x in weights_files]
    assert indices == weights, "indices are not consistent with weights!"

    indices = [int(x) for x in indices]
    weights = [int(x) for x in weights]
    assert indices == weights, "indices are not consistent with weights!"

    indices = np.zeros((n_vertex, 4225, 3)).astype(np.int32)
    weights = np.zeros((n_vertex, 4225, 3))

    for i in range(n_vertex):
        indices_file = abspath + '/neigh_indices/' + str(i) + '_indices.npy'
        weights_file = abspath + '/neigh_indices/' + str(i) + '_weights.npy'
        indices[i, :, :] = np.load(indices_file)
        weights[i, :, :] = np.load(weights_file)

    return indices, weights


def get_z_weight(n_vertex, rotated=0):
    sphere = read_vtk(abspath + '/neigh_indices/sphere_' + str(n_vertex) + '_rotated_' + str(rotated) + '.vtk')
    fixed_xyz = sphere['vertices'] / 100.0
    z_weight = np.abs(fixed_xyz[:, 2])
    index_1 = (z_weight <= 1 / np.sqrt(2)).nonzero()[0]
    index_2 = (z_weight > 1 / np.sqrt(2)).nonzero()[0]
    assert len(index_1) + len(index_2) == n_vertex, "error"
    z_weight[index_1] = 1.0
    z_weight[index_2] = z_weight[index_2] * (-1. / (1. - 1. / np.sqrt(2))) + 1. / (1. - 1. / np.sqrt(2))

    return z_weight


def get_vertex_dis(n_vertex):
    vertex_dis_dic = {42: 54.6,
                      162: 27.5,
                      642: 13.8,
                      2562: 6.9,
                      10242: 3.4,
                      40962: 1.7,
                      163842: 0.8}
    return vertex_dis_dic[n_vertex]

def getOverlapIndex(n_vertex, device):
    """
    Compute the overlap indices' index for the 3 deforamtion field
    """
    z_weight_0 = get_z_weight(n_vertex, 0)
    z_weight_0 = torch.from_numpy(z_weight_0.astype(np.float32)).to(device)
    index_0_0 = (z_weight_0 == 1).nonzero()
    index_0_1 = (z_weight_0 < 1).nonzero()
    assert len(index_0_0) + len(index_0_1) == n_vertex, "error!"
    z_weight_1 = get_z_weight(n_vertex, 1)
    z_weight_1 = torch.from_numpy(z_weight_1.astype(np.float32)).to(device)
    index_1_0 = (z_weight_1 == 1).nonzero()
    index_1_1 = (z_weight_1 < 1).nonzero()
    assert len(index_1_0) + len(index_1_1) == n_vertex, "error!"
    z_weight_2 = get_z_weight(n_vertex, 2)
    z_weight_2 = torch.from_numpy(z_weight_2.astype(np.float32)).to(device)
    index_2_0 = (z_weight_2 == 1).nonzero()
    index_2_1 = (z_weight_2 < 1).nonzero()
    assert len(index_2_0) + len(index_2_1) == n_vertex, "error!"

    index_01 = np.intersect1d(index_0_0.detach().cpu().numpy(), index_1_0.detach().cpu().numpy())
    index_02 = np.intersect1d(index_0_0.detach().cpu().numpy(), index_2_0.detach().cpu().numpy())
    index_12 = np.intersect1d(index_1_0.detach().cpu().numpy(), index_2_0.detach().cpu().numpy())
    index_01 = torch.from_numpy(index_01).to(device)
    index_02 = torch.from_numpy(index_02).to(device)
    index_12 = torch.from_numpy(index_12).to(device)
    rot_mat_01 = torch.tensor([[np.cos(np.pi/2), 0, np.sin(np.pi/2)],
                                [0., 1., 0.],
                                [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]], dtype=torch.float).to(device)
    rot_mat_12 = torch.tensor([[1., 0., 0.],
                                [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                                [0, np.sin(np.pi/2), np.cos(np.pi/2)]], dtype=torch.float).to(device)
    rot_mat_02 = torch.mm(rot_mat_12, rot_mat_01)
    rot_mat_20 = torch.inverse(rot_mat_02)

    tmp = torch.cat((index_0_0, index_1_0, index_2_0))
    tmp, indices = torch.sort(tmp.squeeze())
    output, counts = torch.unique_consecutive(tmp, return_counts=True)
    assert len(output) == n_vertex, "len(output) = n_vertex, error"
    assert output[0] == 0, "output[0] = 0, error"
    assert output[-1] == n_vertex-1, "output[-1] = n_vertex-1, error"
    assert counts.max() == 3, "counts.max() == 3, error"
    assert counts.min() == 2, "counts.min() == 3, error"
    index_triple_computed = (counts == 3).nonzero().squeeze()
    tmp = np.intersect1d(index_02.cpu().numpy(), index_triple_computed.cpu().numpy())
    assert (tmp == index_triple_computed.cpu().numpy()).all(), "(tmp == index_triple_computed.cpu().numpy()).all(), error"
    index_double_02 = torch.from_numpy(np.setdiff1d(index_02.cpu().numpy(), index_triple_computed.cpu().numpy())).to(device)
    tmp = np.intersect1d(index_12.cpu().numpy(), index_triple_computed.cpu().numpy())
    assert (tmp == index_triple_computed.cpu().numpy()).all(), "(tmp == index_triple_computed.cpu().numpy()).all(), error"
    index_double_12 = torch.from_numpy(np.setdiff1d(index_12.cpu().numpy(), index_triple_computed.cpu().numpy())).to(device)
    tmp = np.intersect1d(index_01.cpu().numpy(), index_triple_computed.cpu().numpy())
    assert (tmp == index_triple_computed.cpu().numpy()).all(), "(tmp == index_triple_computed.cpu().numpy()).all(), error"
    index_double_01 = torch.from_numpy(np.setdiff1d(index_01.cpu().numpy(), index_triple_computed.cpu().numpy())).to(device)
    assert len(index_double_01) + len(index_double_12) + len(index_double_02) + len(index_triple_computed) == n_vertex, "double computed and three computed error"

    return rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, z_weight_0, z_weight_1, z_weight_2, index_01, index_12, index_02, index_0_0, index_1_0, index_2_0, index_double_02, index_double_12, index_double_01, index_triple_computed
