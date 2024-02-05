import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GMMConv
from torch import nn

from .utils.rotate_matrix import apply_rotate_matrix, get_en_torch
from .utils.pooling import IcosahedronPooling, IcosahedronUnPooling, get_network_index


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
    phi = np.arctan2(y, x)  # [−π,π]
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)  # [0,π]
    theta_phi = np.concatenate([theta, phi], axis=1)
    # return np.degrees(theta_phi)
    return theta_phi


def get_coordinates_feature(xyz):
    xyz = xyz / torch.norm(xyz, dim=1, keepdim=True)
    theta_phi = xyz_to_lon_lat(xyz)
    theta_phi[:, 0] = theta_phi[:, 0] / np.pi
    theta_phi[:, 1] = theta_phi[:, 1] / (2 * np.pi) + 0.5
    theta_phi = torch.from_numpy(theta_phi)
    return theta_phi


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class ResEncodingBlock(nn.Module):

    def __init__(
            self,
            base_layer,
            in_channel,
            out_channel,
            num_heads=1,
            use_residual=False,
            **kwargs
    ):
        super(ResEncodingBlock, self).__init__()

        self.use_residual = use_residual

        self.same_channel = None
        if self.use_residual:
            if in_channel != out_channel:
                self.same_channel = nn.Linear(in_channel, out_channel)
            #     # self.same_channel = base_layer(in_channel, out_channel, **kwargs)
            self.conv1 = base_layer(out_channel, out_channel // num_heads, num_heads, **kwargs)
            # self.norm1 = nn.LayerNorm(out_channel)
            self.relu1 = nn.LeakyReLU()
            self.conv2 = base_layer(out_channel, out_channel // num_heads, num_heads, **kwargs)
            # self.norm2 = nn.LayerNorm(out_channel)
        else:
            self.conv1 = base_layer(in_channel, out_channel // num_heads, num_heads, **kwargs)
            # self.norm1 = nn.LayerNorm(out_channel)
            self.relu1 = nn.LeakyReLU()
            self.conv2 = base_layer(out_channel, out_channel // num_heads, num_heads, **kwargs)
            # self.norm2 = nn.LayerNorm(out_channel)

    def forward(self, x, edge_index, **kwargs):
        if self.same_channel is not None:
            x = self.same_channel(x)
            # x = self.same_channel(x, edge_index)

        identity = x
        x = self.conv1(x, edge_index, **kwargs)
        # x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x, edge_index, **kwargs)
        # x = self.norm2(x)

        if self.use_residual:
            x += identity

        return x


class GatUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, dropout,
                 use_position_decoding, use_residual, ico_level, input_dropout=0.0, euler_scale=None,
                 rigid=False):
        super(GatUNet, self).__init__()

        self.input_dropout = input_dropout
        self.euler_scale = euler_scale
        self.rigid = rigid

        base_layer = GATv2Conv
        kwargs = {
            'bias': True
        }
        if base_layer is GATv2Conv:
            kwargs['share_weights'] = True
            kwargs['edge_dim'] = 27
        if dropout > 0:
            kwargs['dropout'] = dropout
        if base_layer is GMMConv:
            kwargs['edge_dim'] = 27

        self.theta_phi = None
        self.en = None

        self.use_position_decoding = use_position_decoding
        self.use_residual = use_residual
        self.edge_e = Embedding(3, 4)

        ico_down_levels = {
            'fsaverage3': [3, 3, 2, 1, 0],
            'fsaverage4': [4, 4, 3, 2, 1, 0],
            'fsaverage5': [5, 5, 4, 3, 2, 1, 0],
            'fsaverage6': [6, 6, 5, 4, 3, 2, 1, 0],
        }
        ico_down_levels = ico_down_levels[ico_level]

        ico_down_channels = {
            'fsaverage3': [in_channels, 32, 64, 128, 256, 512],
            'fsaverage4': [in_channels, 32, 64, 128, 256, 512, 768],
            'fsaverage5': [in_channels, 32, 64, 128, 256, 512, 768, 1024],
            'fsaverage6': [in_channels, 32, 64, 128, 256, 512, 768, 1024, 1024],
        }
        ico_down_channels = ico_down_channels[ico_level]

        self.encoding_layers = torch.nn.ModuleList()
        self.pooling_layers = torch.nn.ModuleList()
        self.encoding_edge_index = []
        for i, ico_level in enumerate(ico_down_levels):  # [6, 6, 5, 4, 3, 2]
            if i < len(ico_down_levels) - 1:
                level_in = ico_down_levels[i]
                level_out = ico_down_levels[i+1]
            else:
                level_in = level_out = ico_down_levels[-1]
            channel_in = ico_down_channels[i]
            channel_out = ico_down_channels[i+1]
            self.encoding_layers.append(ResEncodingBlock(base_layer, channel_in, channel_out,
                                                         num_heads=num_heads, use_residual=use_residual,
                                                         **kwargs))
            self.encoding_edge_index.append(get_network_index(f'fsaverage{ico_level}', self.edge_e))
            if level_in > level_out:
                self.pooling_layers.append(IcosahedronPooling(f'fsaverage{ico_level}', pooling_type='mean'))
            else:
                self.pooling_layers.append(torch.nn.Identity())

        self.bottom_layer = ResEncodingBlock(base_layer, ico_down_channels[-1], ico_down_channels[-1],
                                             num_heads=num_heads, use_residual=use_residual,
                                             **kwargs)
        self.bottom_edge_index = get_network_index(f'fsaverage{ico_level}', self.edge_e)

        self.decoding_layers = torch.nn.ModuleList()
        self.decoding_edge_index = []
        self.unpooling_layers = torch.nn.ModuleList()
        for i, ico_level in enumerate(ico_down_levels[::-1]):  # [6, 6, 5, 4, 3, 2]
            if i < len(ico_down_levels) - 1:
                level_in = ico_down_levels[-(i + 1)]
                level_out = ico_down_levels[-(i + 2)]

                channel_in = ico_down_channels[-(i + 1)]
                channel_out = ico_down_channels[-(i + 2)]
                self.decoding_edge_index.append(get_network_index(f'fsaverage{ico_level}', self.edge_e))
                self.decoding_layers.append(ResEncodingBlock(base_layer, (channel_in + channel_out), channel_out,
                                                             num_heads=num_heads, use_residual=use_residual,
                                                             **kwargs))
            else:
                level_in = level_out = ico_down_levels[0]
                channel_in = ico_down_channels[0]
                channel_out = ico_down_channels[1]
                self.decoding_edge_index.append(get_network_index(f'fsaverage{ico_level}', self.edge_e))
                self.decoding_layers.append(ResEncodingBlock(base_layer, (channel_in + channel_out), channel_out,
                                                             num_heads=num_heads, use_residual=use_residual,
                                                             **kwargs))
            if level_in < level_out:
                self.unpooling_layers.append(IcosahedronUnPooling(f'fsaverage{level_in}'))
            else:
                self.unpooling_layers.append(torch.nn.Identity())

        self.output_encoding = ResEncodingBlock(base_layer, ico_down_channels[1], out_channels,
                                                num_heads=1, use_residual=use_residual,
                                                **kwargs)

        self.activate_layer = F.leaky_relu

        self.pe = Embedding(2, 4)
        print(f"learnable_params: {sum(p.numel() for p in list(self.parameters()) if p.requires_grad)}")

    def forward(self, x, xyz_moving, face=None):
        if self.theta_phi is None:
            self.theta_phi = get_coordinates_feature(xyz_moving).to(x.device)
            self.p_e = self.pe(self.theta_phi)
        if self.en is None:
            self.en = get_en_torch(xyz_moving)

        if self.encoding_layers[0].conv1.in_channels == 20:  # in_channels==20 means using PE
            no_pe = False
            x = torch.cat([x, self.p_e], dim=1)
        else:
            no_pe = True

        if 1 > self.input_dropout > 0:
            x = F.dropout(x, self.input_dropout, training=self.training)

        downs = [x]
        for i, encoding_layer in enumerate(self.encoding_layers):
            edge_index, edge_xyz, edge_feature = self.encoding_edge_index[i]
            if no_pe:
                edge_feature = None
            x = encoding_layer(x, edge_index, edge_attr=edge_feature)
            x = self.activate_layer(x)
            downs.append(x)
            x = self.pooling_layers[i](x)

        edge_index, edge_xyz, edge_feature = self.bottom_edge_index
        if no_pe:
            edge_feature = None
        x = self.bottom_layer(x, edge_index, edge_attr=edge_feature)
        x = self.activate_layer(x)

        for i, decoding_layer in enumerate(self.decoding_layers):
            x = self.unpooling_layers[i](x)
            x = torch.cat([x, downs[-(i + 2)]], dim=1)
            edge_index, edge_xyz, edge_feature = self.decoding_edge_index[i]
            if no_pe:
                edge_feature = None
            x = decoding_layer(x, edge_index, edge_attr=edge_feature)
            x = self.activate_layer(x)

        edge_index, edge_xyz, edge_feature = self.decoding_edge_index[0]
        if no_pe:
            edge_feature = None
        euler_angle = self.output_encoding(x, edge_index, edge_attr=edge_feature)
        if self.rigid:
            euler_angle = torch.mean(euler_angle, dim=0, keepdim=True)
        xyz_moved = apply_rotate_matrix(euler_angle, xyz_moving, norm=True, en=self.en, face=face)
        return xyz_moved, euler_angle


if __name__ == '__main__':
    model_params = dict(
        in_channels=2,
        out_channels=3,
        num_heads=8,
        dropout=0,
        use_position_decoding=False,
        use_residual=True,
        ico_level='fsaverage6'
    )
    model = GatUNet(**model_params).to('cuda')
    data_x = torch.ones((40962, 2), dtype=torch.float).to('cuda')
    data_xyz = torch.ones((40962, 3), dtype=torch.float).to('cuda')
    model(data_x, data_xyz)
    print(model)
