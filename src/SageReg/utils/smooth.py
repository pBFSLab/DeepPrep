import numpy as np
import torch
import nibabel as nib
from torch_scatter import scatter_mean


def get_edge_index(faces, device='cuda'):
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
    return edge_index


def smooth(surf_file, morph_file, smooth_file, times=1, device='cuda'):
    xyz_surf, faces_surf = nib.freesurfer.read_geometry(surf_file)
    edge_index = get_edge_index(faces_surf).to(device)
    row, col = edge_index

    morph_data = nib.freesurfer.read_morph_data(morph_file).astype(np.float32)
    morph_data = np.expand_dims(morph_data, 1)
    morph_data = torch.from_numpy(morph_data.astype(np.float32)).to(device)

    for _ in range(times):
        morph_data = scatter_mean(morph_data[col], row, dim=0, dim_size=morph_data.size(0))
    nib.freesurfer.write_morph_data(smooth_file, morph_data.cpu())
    print(f'smooth >>> {smooth_file}')


def main():
    times = 5
    surf_file = '/mnt/ngshare/SurfReg/Data_TrainResult/test/NAMIC/sub01/surf/lh.rigid.interp_fs.sphere'
    morph_file = '/mnt/ngshare/SurfReg/Data_TrainResult/test/NAMIC/sub01/surf/lh.rigid.interp_fs.sulc'
    smooth_file = f'/mnt/ngshare/SurfReg/Data_TrainResult/test/NAMIC/sub01/surf/lh.rigid.interp_fs.smooth_{times}.sulc'
    smooth(surf_file, morph_file, smooth_file, times=times)


if __name__ == '__main__':
    main()
