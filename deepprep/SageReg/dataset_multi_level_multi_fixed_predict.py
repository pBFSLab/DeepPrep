import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from nibabel.freesurfer import read_morph_data, read_geometry, read_annot


def normalize(data, norm_method='SD', mean=None, std=None, mi=None, ma=None):
    """
    data: 163842 * 1, numpy array
    """
    if norm_method == 'SD':
        data = data - np.median(data)
        data = data / np.std(data)

        index = np.where(data < -3)[0]
        data[index] = -3 - (1 - np.exp(3 - np.abs(data[index])))
        index = np.where(data > 3)[0]
        data[index] = 3 + (1 - np.exp(3 - np.abs(data[index])))

    elif norm_method == 'MinMax':
        data = (data - data.min()) / data.max()
    elif norm_method == 'zscore':
        data = (data - data.mean()) / data.std()

        index = np.where(data < -3)[0]
        data[index] = -3 - (1 - np.exp(3 - np.abs(data[index])))
        index = np.where(data > 3)[0]
        data[index] = 3 + (1 - np.exp(3 - np.abs(data[index])))
    elif norm_method == 'PriorGaussian':
        assert mean is not None and std is not None, "PriorGaussian needs prior mean and std"
        data = (data - mean) / std
    elif norm_method == 'PriorMinMax':
        assert mi is not None and ma is not None, "PriorMinMax needs prior min and max"
        data = (data - mi) / (ma - mi) * 2. - 1.
    else:
        raise NotImplementedError('e')

    return data


def data_random_rotate(sulc, curv, xyz):
    from utils.rotate_matrix import apply_rotate_matrix
    from utils.interp import resample_sphere_surface_barycentric
    euler = np.random.random(3) * 0.01
    euler_t = torch.from_numpy(euler.reshape(1, -1)).float().to('cuda')
    sulc_t = torch.from_numpy(sulc).float().to('cuda')
    curv_t = torch.from_numpy(curv).float().to('cuda')
    xyz_t = torch.from_numpy(xyz).float().to('cuda')
    xyz_r = apply_rotate_matrix(euler_t, xyz_t, norm=True)
    sulc_r = resample_sphere_surface_barycentric(xyz_r, xyz_t, sulc_t.unsqueeze(1))
    curv_r = resample_sphere_surface_barycentric(xyz_r, xyz_t, curv_t.unsqueeze(1))
    return sulc_r.squeeze().cpu().numpy(), curv_r.squeeze().cpu().numpy()



class SphericalDataset(Dataset):
    def __init__(self,  dir_fixed=None, dir_result=None,
                 lrh='lh', feature='sulc', norm_type='SD',
                 ico_levels=None,
                 seg=False, is_train=True, is_da=False, is_rigid=False):
        self.dir_fixed = dir_fixed
        self.dir_result = dir_result
        self.lrh = lrh
        self.feature = feature
        self.norm_type = norm_type

        self.sulc_fixed = self.curv_fixed = None

        self.ico_levels = ico_levels

        self.fixed = None

        self.seg = seg
        self.is_train = is_train
        self.is_rigid = is_rigid
        self.is_da = is_da

    def get_fixed(self):
        normalize_type = self.norm_type

        if self.fixed is None:
            fixed = list()
            for ico_level in self.ico_levels:
                dir_fixed = self.dir_fixed
                sulc_fixed = read_morph_data(os.path.join(dir_fixed, ico_level, 'surf', f'{self.lrh}.sulc')).astype(np.float32)
                curv_fixed = read_morph_data(os.path.join(dir_fixed, ico_level, 'surf', f'{self.lrh}.curv')).astype(np.float32)
                xyz_fixed, faces_fixed = read_geometry(os.path.join(dir_fixed, ico_level, 'surf', f'{self.lrh}.sphere'))
                xyz_fixed = xyz_fixed.astype(np.float32) / 100
                faces_fixed = faces_fixed.astype(int)

                if normalize_type == 'PriorMinMax':
                    sulc_fixed *= 10
                sulc_data_min = -12
                sulc_data_max = 14
                sulc_fixed = normalize(sulc_fixed, normalize_type, mi=sulc_data_min, ma=sulc_data_max)
                curv_data_min = -1.3
                curv_data_max = 1.0
                curv_fixed = normalize(curv_fixed, normalize_type, mi=curv_data_min, ma=curv_data_max)

                seg_fixed, seg_color_fixed, seg_name_fixed = read_annot(os.path.join(dir_fixed, ico_level, 'label', f'{self.lrh}.aparc.annot'))
                seg_fixed = seg_fixed.astype(int)
                fixed.append([sulc_fixed, curv_fixed, xyz_fixed, faces_fixed, seg_fixed])
            self.fixed = fixed
        return self.fixed

    def get_moving(self, ico_level):
        """
        刚性配准后的数据
        """
        sub_dir_result = os.path.join(self.dir_result, 'surf')
        if self.is_rigid:
            data_type = 'orig'
        else:
            data_type = 'rigid'

        sulc_moving_interp = os.path.join(sub_dir_result, f'{self.lrh}.{data_type}.interp_{ico_level}.sulc')
        curv_moving_interp = os.path.join(sub_dir_result, f'{self.lrh}.{data_type}.interp_{ico_level}.curv')
        sphere_moving_file = os.path.join(sub_dir_result, f'{self.lrh}.{data_type}.interp_{ico_level}.sphere')

        sulc_moving = read_morph_data(sulc_moving_interp).astype(np.float32)
        curv_moving = read_morph_data(curv_moving_interp).astype(np.float32)

        xyz_moving, faces_moving = read_geometry(sphere_moving_file)
        xyz_moving = xyz_moving.astype(np.float32) / 100
        faces_moving = faces_moving.astype(int)

        # TODO 数据增强
        if self.is_da:
            sulc_moving, curv_moving = data_random_rotate(sulc_moving, curv_moving, xyz_moving)

        # 归一化
        normalize_type = self.norm_type
        sulc_data_min = -12
        sulc_data_max = 14
        sulc_moving = normalize(sulc_moving, normalize_type, mi=sulc_data_min, ma=sulc_data_max)  # PriorMinMax
        curv_data_min = -1.3
        curv_data_max = 1.0
        curv_moving = normalize(curv_moving, normalize_type, mi=curv_data_min, ma=curv_data_max)

        if self.seg is True:
            sub_dir_label = os.path.join(self.dir_result, 'label')
            annot_file = os.path.join(sub_dir_label, f'{self.lrh}.gt.{data_type}.interp_{ico_level}.aparc.annot')
            seg_moving, seg_color_moving, seg_name_moving = read_annot(annot_file)
            seg_moving = seg_moving.astype(int)
            seg_moving[seg_moving == -1] = 0  # any -1 should be 0
        else:
            seg_moving = np.zeros_like(sulc_moving, dtype=int)

        return sulc_moving, curv_moving, xyz_moving, faces_moving, seg_moving

    def __getitem__(self, index):

        fixed = self.get_fixed()
        movings = list()

        for ico_level in self.ico_levels:
            sulc_moving, curv_moving, xyz_moving, faces_moving, seg_moving = self.get_moving(ico_level)
            movings.append([sulc_moving, curv_moving, xyz_moving, faces_moving, seg_moving])


        return movings, fixed

    def __len__(self):
        return 1




