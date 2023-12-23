import os
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_max

abspath = os.path.abspath(os.path.dirname(__file__))


def xyz_to_lon_lat(xyz):
    """
    Convert x, y, z coordinates to lon, lat in degrees.
    x: x coordinate
    y: y coordinate
    z: z coordinate
    return: lon, lat in degrees
    """
    import numpy as np
    xyz = xyz.cpu().numpy()
    x, y, z = xyz[:, [0]], xyz[:, [1]], xyz[:, [2]]
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    theta_phi = np.concatenate([theta, phi], axis=1)
    # return np.degrees(theta_phi)
    return theta_phi


def get_coordinates_feature(xyz):
    xyz = xyz / torch.norm(xyz, dim=1, keepdim=True)
    theta_phi = xyz_to_lon_lat(xyz)
    theta_phi[:, 0] /= (theta_phi[:, 0].max() - theta_phi[:, 0].min())
    theta_phi[:, 1] /= (theta_phi[:, 1].max() - theta_phi[:, 1].min())
    theta_phi += 1
    theta_phi = torch.from_numpy(theta_phi)
    return theta_phi


def get_network_index(ico_level, pe=None, device='cuda'):
    from nibabel.freesurfer import read_geometry
    sphere_file = os.path.join(abspath, 'auxi_data', f'{ico_level}.sphere')
    xyz, faces = read_geometry(sphere_file)

    x = np.expand_dims(faces[:, 0], 1)
    y = np.expand_dims(faces[:, 1], 1)
    z = np.expand_dims(faces[:, 2], 1)

    a = np.concatenate([x, y], axis=1)
    b = np.concatenate([y, x], axis=1)
    c = np.concatenate([x, z], axis=1)
    d = np.concatenate([z, x], axis=1)
    e = np.concatenate([y, z], axis=1)
    f = np.concatenate([z, y], axis=1)

    edge_index = np.concatenate([a, b, c, d, e, f]).astype(int)
    edge_index = np.unique(edge_index, axis=0).astype(int)
    edge_index = edge_index[np.argsort(edge_index[:, 0])]
    edge_index = torch.from_numpy(edge_index).to(device)
    edge_index = edge_index.t().contiguous()

    xyz = torch.from_numpy(xyz).float().to(device)
    edge_xyz = xyz[edge_index]

    if pe is not None:
        edge_feature = pe(edge_xyz.mean(dim=0))
    else:
        edge_feature = get_coordinates_feature(edge_xyz.mean(dim=0)).to(device)
    return edge_index, edge_xyz, edge_feature


def get_pooling_index(ico_level, device='cuda'):
    from nibabel.freesurfer import read_geometry
    sphere_file = os.path.join(abspath, 'auxi_data', f'{ico_level}.sphere')
    xyz, faces = read_geometry(sphere_file)

    unpooling_num = {
        'fsaverage1': 12,
        'fsaverage2': 42,
        'fsaverage3': 162,
        'fsaverage4': 642,
        'fsaverage5': 2562,
        'fsaverage6': 10242,
        'fsaverage7': 40962,
    }

    x = np.expand_dims(faces[:, 0], 1)
    y = np.expand_dims(faces[:, 1], 1)
    z = np.expand_dims(faces[:, 2], 1)

    a = np.concatenate([x, y], axis=1)
    b = np.concatenate([y, x], axis=1)
    c = np.concatenate([x, z], axis=1)
    d = np.concatenate([z, x], axis=1)
    e = np.concatenate([y, z], axis=1)
    f = np.concatenate([z, y], axis=1)

    edge_index = np.concatenate([a, b, c, d, e, f]).astype(int)
    edge_index = np.unique(edge_index, axis=0).astype(int)
    edge_index = edge_index[np.argsort(edge_index[:, 0])]

    # only keep the low_level index
    num = np.where(edge_index[:, 0] == unpooling_num[ico_level])
    edge_index = edge_index[:np.min(num)]

    # add self node edge
    self = np.arange(0, unpooling_num[ico_level]).reshape(-1, 1)
    self = np.concatenate([self, self], axis=1)
    edge_index = np.concatenate([edge_index, self], axis=0)
    edge_index = edge_index[np.argsort(edge_index[:, 0])]

    edge_index = torch.from_numpy(edge_index).to(device)
    edge_index = edge_index.t().contiguous()
    return edge_index


def get_unpooling_index(ico_level, device='cuda'):
    upsample_neighbors_file = os.path.join(abspath, 'auxi_data', f'{ico_level}_upsample_neighbors.npz')
    upsample_neighbors_load = np.load(upsample_neighbors_file)
    upsample_index = upsample_neighbors_load['upsample_neighbors']
    upsample_index = torch.from_numpy(upsample_index).to(device)
    return upsample_index


def get_matrix(index):
    index = index.cpu().numpy()
    num = len(np.unique(index[0]))
    matrix = [[] for _ in range(num)]
    for row, col in index.T:
        # row, col = indexs[0], indexs[1]
        matrix[row].append(col)
    for i, row in enumerate(matrix):
        if len(row) < 7:
            row.extend([i] * (7-len(row)))
    matrix = torch.from_numpy(np.array(matrix))
    return matrix


class IcosahedronPooling(nn.Module):
    def __init__(self, ico_level, pooling_type='mean'):
        super(IcosahedronPooling, self).__init__()
        self.pooling_index = get_pooling_index(ico_level, device='cpu')
        self.pooling_type = pooling_type
        # self.pooling_index_matrix = get_matrix(self.pooling_index)

    def forward(self, x):
        index = self.pooling_index.to(x.device)

        if self.pooling_type == 'mean':
            x = scatter_mean(x[index[1]], index[0], dim=0)
            # x = x[self.pooling_index_matrix].mean(dim=1)
        elif self.pooling_type == 'max':
            x = scatter_max(x[index[1]], index[0], dim=0)[0]
            # x = x[self.pooling_index_matrix].max(dim=1)
        return x


class IcosahedronUnPooling(nn.Module):
    def __init__(self, ico_level, channel_in=None, channel_out=None):
        super(IcosahedronUnPooling, self).__init__()
        self.unpooling_index = get_unpooling_index(ico_level, device='cpu')
        if channel_in and channel_out:
            self.linear = nn.Linear(channel_in, channel_out)
        else:
            self.linear = None

    def forward(self, x):
        index = self.unpooling_index.to(x.device)

        x_up = (x[index[:, 0]] + x[index[:, 1]]) / 2
        x = torch.cat([x, x_up], dim=0)
        if self.linear is not None:
            x = self.linear(x)
        return x


if __name__ == '__main__':
    # get_network_index('fsaverage3', device='cuda')
    # ico_pooling = IcosahedronPooling('fsaverage4', pooling_type='max')
    # test = torch.ones((2562, 2), dtype=torch.float, device='cuda')
    # ico_pooling(test)
    # ico_unpooling = IcosahedronUnPooling('fsaverage2')

    for i in range(1, 7):
        get_network_index(f'fsaverage{i}')
        print()