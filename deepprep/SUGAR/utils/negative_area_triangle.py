import os
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch_scatter import scatter_mean
import time
from functools import wraps
"""
负面积三角形相关代码

1. 判断负面积三角形
2. 优化负面积三角形
"""


def timing_func(function):
    @wraps(function)
    def timer(*args, **kwargs):
        tic = time.time()
        result = function(*args, **kwargs)
        toc = time.time()
        print('[Finished func: {func_name} in {time:.4f}s]'.format(func_name=function.__name__, time=toc - tic))
        return result

    return timer


def negative_area(faces, xyz):
    n = faces[:, 0]
    n0 = faces[:, 2]
    n1 = faces[:, 1]

    v0 = xyz[n] - xyz[n0]
    v1 = xyz[n1] - xyz[n]

    d1 = -v1[:, 1] * v0[:, 2] + v0[:, 1] * v1[:, 2]
    d2 = v1[:, 0] * v0[:, 2] - v0[:, 0] * v1[:, 2]
    d3 = -v1[:, 0] * v0[:, 1] + v0[:, 0] * v1[:, 1]

    dot = xyz[n][:, 0] * d1 + xyz[n][:, 1] * d2 + xyz[n][:, 2] * d3

    area = torch.sqrt(d1 * d1 + d2 * d2 + d3 * d3) / 2

    area = torch.where(dot < 0, area * -1, area)
    # area[dot < 0] *= -1

    return area


def count_negative_area(faces, xyz):
    area = negative_area(faces, xyz)
    index = area < 0
    count = index.sum()  # 面积为负的面
    print(f'negative area count : {count}')
    return count


@timing_func
def remove_negative_area(faces, xyz, device='cuda'):
    """
    基于laplacian smoothing的原理
    https://en.wikipedia.org/wiki/Laplacian_smoothing
    """

    # xyz_dt = (-6 * xyz + scatter_add(xyz[col], row, dim=0)) / 6
    area = negative_area(faces, xyz)
    index = area < 0
    count = index.sum()  # 面积为负的面
    # print(f'negative area count : {count}')

    remove_times = 0
    dt_weight_init = 1  # 初始值

    x = np.expand_dims(faces.cpu()[:, 0], 1)
    y = np.expand_dims(faces.cpu()[:, 1], 1)
    z = np.expand_dims(faces.cpu()[:, 2], 1)

    a = np.concatenate([x, y], axis=1)
    b = np.concatenate([y, x], axis=1)
    c = np.concatenate([x, z], axis=1)
    d = np.concatenate([z, x], axis=1)
    e = np.concatenate([y, z], axis=1)
    f = np.concatenate([z, y], axis=1)

    edge_index = np.concatenate([a, b, c, d, e, f])
    edge_index = torch.from_numpy(edge_index).to(device)
    edge_index = edge_index.t().contiguous()

    row, col = edge_index

    while count > 0:
        dt_weight = dt_weight_init - count % 10 * 0.01  # 按比例减小

        xyz_dt = scatter_mean(xyz[col], row, dim=0) - xyz
        neg_faces = faces[index]
        index = neg_faces.flatten()
        xyz[index] = xyz[index] + xyz_dt[index] * dt_weight
        xyz = xyz / torch.norm(xyz, dim=1, keepdim=True) * 100

        area = negative_area(faces, xyz)
        index = area < 0
        count = index.sum()  # 面积为负的面
        # print(f'negative area count : {count}')
        remove_times += 1

        if remove_times >= 1000:
            break

    return xyz, count, remove_times


def single_remove_negative_area(sphere, sphere_removed, device='cuda'):
    xyz_sphere, faces_sphere = nib.freesurfer.read_geometry(str(sphere))
    xyz_sphere = torch.from_numpy(xyz_sphere.astype(np.float32)).to(device)
    faces_sphere = torch.from_numpy(faces_sphere.astype(int)).to(device)
    area = negative_area(faces_sphere, xyz_sphere)
    index = area < 0
    count_orig = index.sum()  # 面积为负的面
    # print(f'negative area count : {count_orig}')
    times = count_final = 0
    if count_orig > 0:
        xyz_sphere_removed, count_final, times = remove_negative_area(faces_sphere, xyz_sphere, device)
        # print(f'negative area: {count_orig}   {count_final}  {times}')
        if sphere_removed == sphere:
            os.system(f'mv {sphere} {sphere}.bak')
        nib.freesurfer.write_geometry(sphere_removed, xyz_sphere_removed.cpu().numpy(), faces_sphere.cpu().numpy())
        if os.path.exists(f'{sphere}.bak'):
            os.system(f'rm {sphere}.bak')
        # print(f'remove negative area triangle: >>> {sphere_removed}')
    else:
        print(f'negative area: {count_orig}   {count_final}  {times}')
