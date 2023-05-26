import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from nibabel.freesurfer import read_morph_data, read_geometry


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


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

        data = data / np.std(data)
        index = np.where(data < -3)[0]
        data[index] = -3 - (1 - np.exp(3 - np.abs(data[index])))
        index = np.where(data > 3)[0]
        data[index] = 3 + (1 - np.exp(3 - np.abs(data[index])))

    elif norm_method == 'MinMax':
        data = (data - data.min()) / data.max()
    elif norm_method == 'Gaussian':
        data = (data - data.mean()) / data.std()
    elif norm_method == 'PriorGaussian':
        assert mean is not None and std is not None, "PriorGaussian needs prior mean and std"
        data = (data - mean) / std
    elif norm_method == 'PriorMinMax':
        assert mi is not None and ma is not None, "PriorMinMax needs prior min and max"
        data = (data - mi) / (ma - mi) * 2. - 1.
    else:
        raise NotImplementedError('e')

    return data


def data_inverse_norm(data):
    index = np.where(data < -3)[0]
    data[index] = -2 - (1 - np.exp(2 - np.abs(data[index])))
    index = np.where(data > 3)[0]
    data[index] = 2 + (1 - np.exp(2 - np.abs(data[index])))
    return data


class SphericalDataset(Dataset):
    def __init__(self, sublist=None, dir_fixed=None, dir_result=None,
                 lrh='lh', feature='sulc', norm_type='SD'):
        self.dir_fixed = dir_fixed
        self.dir_result = dir_result
        self.lrh = lrh
        self.feature = feature
        self.norm_type = norm_type
        if isinstance(sublist, str):  # str, split_txt_file
            with open(sublist, "r") as f:
                self.sub_dirs = f.readlines()
        else:  # list
            self.sub_dirs = sublist

        self.sulc_fixed = self.curv_fixed = None

    def get_fixed(self):
        normalize_type = self.norm_type

        if self.sulc_fixed is None:
            sub_dir_fixed = self.dir_fixed
            sulc_fixed = read_morph_data(os.path.join(sub_dir_fixed, 'surf', f'{self.lrh}.sulc')).astype(np.float32)
            curv_fixed = read_morph_data(os.path.join(sub_dir_fixed, 'surf', f'{self.lrh}.curv')).astype(np.float32)
            if normalize_type == 'PriorMinMax':
                sulc_fixed *= 10
            sulc_data_min = -12
            sulc_data_max = 14
            sulc_fixed = normalize(sulc_fixed, normalize_type, mi=sulc_data_min, ma=sulc_data_max)
            curv_data_min = -1.3
            curv_data_max = 1.0
            curv_fixed = normalize(curv_fixed, normalize_type, mi=curv_data_min, ma=curv_data_max)
            self.sulc_fixed, self.curv_fixed = sulc_fixed, curv_fixed
        else:
            sulc_fixed, curv_fixed = self.sulc_fixed, self.curv_fixed
        return sulc_fixed, curv_fixed

    def get_moving(self, sub_id):
        """
        刚性配准后的数据
        """
        sub_dir_result = os.path.join(self.dir_result, sub_id, 'surf')

        sulc_moving_interp = os.path.join(sub_dir_result, f'{self.lrh}.rigid.interp_fs.sulc')
        curv_moving_interp = os.path.join(sub_dir_result, f'{self.lrh}.rigid.interp_fs.curv')

        sphere_moving_file = os.path.join(sub_dir_result, f'{self.lrh}.rigid.interp_fs.sphere')

        sulc_moving = read_morph_data(sulc_moving_interp).astype(np.float32)
        curv_moving = read_morph_data(curv_moving_interp).astype(np.float32)

        # 归一化
        normalize_type = self.norm_type
        sulc_data_min = -12
        sulc_data_max = 14
        sulc_moving = normalize(sulc_moving, normalize_type, mi=sulc_data_min, ma=sulc_data_max)  # PriorMinMax
        curv_data_min = -1.3
        curv_data_max = 1.0
        curv_moving = normalize(curv_moving, normalize_type, mi=curv_data_min, ma=curv_data_max)

        xyz_moving, faces_moving = read_geometry(sphere_moving_file)
        xyz_moving = xyz_moving.astype(np.float32) / 100
        faces_moving = faces_moving.astype(np.float32)

        return sulc_moving, curv_moving, xyz_moving, faces_moving, sphere_moving_file

    def __getitem__(self, index):
        sub_id = self.sub_dirs[index].strip()
        sulc_moving, curv_moving, xyz_moving, faces_moving, sphere_moving_file = self.get_moving(sub_id)
        sulc_fixed, curv_fixed = self.get_fixed()

        if self.feature == 'sulc':
            moving = sulc_moving.reshape(-1, 1)
            fixed = sulc_fixed.reshape(-1, 1)
        elif self.feature == 'curv':
            moving = curv_moving.reshape(-1, 1)
            fixed = curv_fixed.reshape(-1, 1)
        elif self.feature == 'sucu':
            moving = np.dstack((sulc_moving, curv_moving)).squeeze()
            fixed = np.dstack((sulc_fixed, curv_fixed)).squeeze()
        elif self.feature == 'sucu_inverse':
            sulc_moving_inverse = 1. / sulc_moving
            curv_moving_inverse = 1. / curv_moving
            sulc_fixed_inverse = 1. / sulc_fixed
            curv_fixed_inverse = 1. / curv_fixed

            sulc_moving_inverse = data_inverse_norm(sulc_moving_inverse)
            curv_moving_inverse = data_inverse_norm(curv_moving_inverse)
            sulc_fixed_inverse = data_inverse_norm(sulc_fixed_inverse)
            curv_fixed_inverse = data_inverse_norm(curv_fixed_inverse)

            moving = np.dstack((sulc_moving, curv_moving, sulc_moving_inverse, curv_moving_inverse)).squeeze()
            fixed = np.dstack((sulc_fixed, curv_fixed, sulc_fixed_inverse, curv_fixed_inverse)).squeeze()
        else:
            raise NotImplementedError('NotImplementedError')

        return moving, fixed, xyz_moving, sub_id, sphere_moving_file

    def __len__(self):
        return len(self.sub_dirs)


def get_inputdata(sulc_moving_f, curv_moving_f=None, sphere_moving_f=None,
                  sulc_fixed_f=None, curv_fixed_f=None, sphere_fixed_f=None,
                  normalize_type="SD", device='cuda'):
    # 加载数据
    sulc_moving = read_morph_data(sulc_moving_f).astype(np.float32)
    curv_moving = read_morph_data(curv_moving_f).astype(np.float32)
    xyz_moving, faces_moving = read_geometry(sphere_moving_f)
    xyz_moving = xyz_moving.astype(np.float32)

    sulc_fixed = read_morph_data(sulc_fixed_f).astype(np.float32)
    curv_fixed = read_morph_data(curv_fixed_f).astype(np.float32)
    xyz_fixed, faces_fixed = read_geometry(sphere_fixed_f)
    xyz_fixed = xyz_fixed.astype(np.float32)

    # 数据归一化
    sulc_data_min = -12
    sulc_data_max = 14
    sulc_moving = normalize(sulc_moving, normalize_type, mi=sulc_data_min, ma=sulc_data_max)
    curv_data_min = -1.3
    curv_data_max = 1.0
    curv_moving = normalize(curv_moving, normalize_type, mi=curv_data_min, ma=curv_data_max)

    sulc_data_min = -12
    sulc_data_max = 14
    sulc_fixed = normalize(sulc_fixed, normalize_type, mi=sulc_data_min, ma=sulc_data_max)
    if normalize_type == 'PriorMinMax':
        sulc_fixed *= 10
    curv_data_min = -1.3
    curv_data_max = 1.0
    curv_fixed = normalize(curv_fixed, normalize_type, mi=curv_data_min, ma=curv_data_max)

    sulc_moving = torch.from_numpy(sulc_moving).to(device)
    curv_moving = torch.from_numpy(curv_moving).to(device)
    xyz_moving = torch.from_numpy(xyz_moving).to(device)
    sulc_fixed = torch.from_numpy(sulc_fixed).to(device)
    curv_fixed = torch.from_numpy(curv_fixed).to(device)
    xyz_fixed = torch.from_numpy(xyz_fixed).to(device)

    return sulc_moving, curv_moving, xyz_moving, sulc_fixed, curv_fixed, xyz_fixed, faces_fixed


def get_dataset_item(dir_moved, dir_fixed, sub_id, hemisphere, feature, device='cuda'):
    if not os.path.isdir(dir_moved):
        os.makedirs(dir_moved)

    # fixed文件
    sulc_fixed_file = os.path.join(dir_fixed, 'surf', f'{hemisphere}.sulc')
    curv_fixed_file = os.path.join(dir_fixed, 'surf', f'{hemisphere}.curv')
    sphere_fixed_file = os.path.join(dir_fixed, 'surf', f'{hemisphere}.sphere')
    # 插值到标准20面体的结果
    sulc_moving_interp_file = os.path.join(dir_moved, sub_id, 'surf', f'{hemisphere}.orig.interp_fs.sulc')
    curv_moving_interp_file = os.path.join(dir_moved, sub_id, 'surf', f'{hemisphere}.orig.interp_fs.curv')
    sphere_moving_interp_file = os.path.join(dir_moved, sub_id, 'surf', f'{hemisphere}.orig.interp_fs.sphere')

    # ############################# 开始配准 ###########################
    sulc_moving, curv_moving, xyz_moving, sulc_fixed, curv_fixed, xyz_fixed, faces_fixed = get_inputdata(
        sulc_moving_interp_file, curv_moving_interp_file, sphere_moving_interp_file,
        sulc_fixed_file, curv_fixed_file, sphere_fixed_file, device=device)

    # 数据归一化
    if feature == 'sulc':
        data_input = torch.cat((sulc_moving.unsqueeze(1), sulc_fixed.unsqueeze(1)), 1)
        moving_data = sulc_moving.unsqueeze(1)
        fixed_data = sulc_fixed.unsqueeze(1)
    elif feature == 'sucu':
        data_input = torch.cat((sulc_moving.unsqueeze(1), curv_moving.unsqueeze(1),
                                sulc_fixed.unsqueeze(1), curv_fixed.unsqueeze(1)), 1)
        moving_data = torch.cat((sulc_moving.unsqueeze(1), curv_moving.unsqueeze(1)), 1)
        fixed_data = torch.cat((sulc_fixed.unsqueeze(1), curv_fixed.unsqueeze(1)), 1)
    else:
        raise KeyError("feature is error")
    xyz_moving /= 100
    xyz_fixed /= 100
    return data_input, moving_data, fixed_data, xyz_moving, xyz_fixed


if __name__ == '__main__':
    _sublist_txt = '/home/anning/projects/Data_SurfReg/train_list.txt'
    _dir_moving = '/home/anning/projects/Data_SurfReg/FreeSurfer_904_16W'
    _dir_fixed = '/home/anning/projects/Data_SurfReg/fsaverage'
    dataset_train = SphericalDataset(_sublist_txt, _dir_moving, _dir_fixed)
    _moving, _fixed, _sub_dir = dataset_train[0]
    print(_moving, _fixed)
    print(np.min(_moving[:, 0]), np.max(_moving[:, 0]))
    print(np.min(_moving[:, 1]), np.max(_moving[:, 1]))
    print(np.min(_fixed[:, 0]), np.max(_fixed[:, 0]))
    print(np.min(_fixed[:, 1]), np.max(_fixed[:, 1]))
    print(_moving.shape, _fixed.shape, _sub_dir)
