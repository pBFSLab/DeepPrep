import os
from nibabel.freesurfer import read_geometry
import torch

abspath = os.path.abspath(os.path.dirname(__file__))
auxi_data_path = os.path.join(abspath, 'auxi_data')


ico_level_by_points_num = {
    12: 0,
    42: 1,
    162: 2,
    642: 3,
    2562: 4,
    10242: 5,
    40962: 6,
    163842: 7,
}

points_num_by_ico_level = {
    0: 12,
    1: 42,
    2: 162,
    3: 642,
    4: 2562,
    5: 10242,
    6: 40962,
    7: 163842,
}

distance_by_ico_level = {
    0: 54.6490 * 2,
    1: 54.6490,
    2: 27.5809,
    3: 13.8173,
    4: 6.9110,
    5: 3.4491,
    6: 1.7178,
    7: 1.7178 / 2,
}


def fs_to_num(fsaverage):
    return int(fsaverage[-1])


def get_geometry_by_ico_level(ico_level):
    sphere_file = os.path.join(auxi_data_path, f'{ico_level}.sphere')
    xyz, faces = read_geometry(sphere_file)
    return xyz, faces


def get_points_num_by_ico_level(ico_level: str):
    return points_num_by_ico_level[fs_to_num(ico_level)]


def get_distance_by_points_num(num: int):
    ico_level = ico_level_by_points_num[num]
    distance = distance_by_ico_level[ico_level]
    return distance


def get_geometry_all_level_torch():
    xyzs = {}
    faces = {}
    for i in range(0, 7):
        ico_level = f'fsaverage{i}'
        xyz_fixed, face_fixed = get_geometry_by_ico_level(ico_level)
        xyzs[ico_level] = torch.from_numpy(xyz_fixed).float() / 100
        faces[ico_level] = torch.from_numpy(face_fixed.astype(int))
    return xyzs, faces
