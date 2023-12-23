import nibabel as nib
import os
from utils.utils import abspath
import torch
from pathlib import Path
import numpy as np
from utils.interp_fine import resample_sphere_surface_barycentric
from multiprocessing import Pool

def multipool(cmd, sub, Multi_Num=2):
    cmd_pool = []
    for i in range(len(sub)):
        cmd_pool.append([sub[i]])

    pool = Pool(Multi_Num)
    pool.starmap(cmd, cmd_pool)
    pool.close()
    pool.join()


def get_sucu(sub):
    dir_path = f'/run/user/1001/gvfs/sftp:host=30.30.30.204,user=anning/mnt/ngshare/SurfReg/Data_Processing/904'

    ori_sphere_path = Path(dir_path) / sub / 'surf' / 'rh.sphere.reg'
    fixed_xyz, faces_to_native = nib.freesurfer.read_geometry(ori_sphere_path)

    fixed_sulc_path = f'{dir_path}/{sub}/surf/rh.sulc'
    fixed_sulc = nib.freesurfer.read_morph_data(fixed_sulc_path)

    fixed_curv_path = f'{dir_path}/{sub}/surf/rh.curv'
    fixed_curv = nib.freesurfer.read_morph_data(fixed_curv_path)

    fs_sphere_path = f'/usr/local/freesurfer720/subjects/fsaverage6/surf/rh.sphere.reg'
    moving_xyz, faces_to_fixed = nib.freesurfer.read_geometry(fs_sphere_path)


    new_sucu_path = f'/mnt/ngshare/FreeSurfer_904_sucu/{sub}/surf'
    Path(new_sucu_path).mkdir(exist_ok=True, parents=True)

    fixed_xyz = torch.from_numpy(fixed_xyz)
    moving_xyz = torch.from_numpy(moving_xyz)
    fixed_sulc = torch.as_tensor(np.expand_dims(fixed_sulc.astype(np.float64), 1))
    fixed_curv = torch.as_tensor(np.expand_dims(fixed_curv.astype(np.float64), 1))
    device = torch.device('cuda')

    res_sulc = resample_sphere_surface_barycentric(fixed_xyz, moving_xyz, fixed_sulc,
                                                           device=device)
    res_curv = resample_sphere_surface_barycentric(fixed_xyz, moving_xyz, fixed_curv,
                                                           device=device)
    nib.freesurfer.write_morph_data(f'{new_sucu_path}/rh.40962.sulc', res_sulc.cpu().numpy())
    nib.freesurfer.write_morph_data(f'{new_sucu_path}/rh.40962.curv', res_curv.cpu().numpy())
    print('done')

def make_weight():
    # dir_path = '/run/user/1001/gvfs/sftp:host=30.30.30.52,user=zhenyu/mnt/ngshare/Workspace/result/all_reg_HCP_sucu/all_data'
    dir_path = f'/mnt/ngshare/FreeSurfer_904_sucu'
    sub_list = os.listdir(dir_path)

    auxi_data_path = '/home/lincong/workspace/NGSurfReg/FeatReg/featreg/utils/auxi_data'

    datas = None
    datas_mean = []
    datas_std = []
    for sub_name in sub_list:
        sulc_file = os.path.join(dir_path, sub_name, 'surf', 'rh.40962.sulc')
        sulc_data = nib.freesurfer.read_morph_data(sulc_file)

        # print(sub_name, '_min: ', sulc_data.min(), ' max: ', sulc_data.max(), ' mean: ', sulc_data.mean(), ' std: ', sulc_data.std())
        datas = sulc_data[:, np.newaxis] if datas is None else np.concatenate((datas, sulc_data[:, np.newaxis]), axis=1)
    # for i in range(nd(np.nanstd(datas[i, :]))

    point_mean = datas.mean(axis=1).reshape(40962, 1)
    point_std = datas.std(axis=1).reshape(40962, 1)
    z_score = abs((datas - point_mean) / point_std)
    res = np.where(z_score > 3)
    datas[res] = np.nan
    datas_mean.append(np.nanmean(datas, axis=1))
    datas_std.append(np.nanstd(datas, axis=1))


    datas_mean = np.array(datas_mean).reshape(40962)
    datas_std = np.array(datas_std).reshape(40962)
    print()
    nib.freesurfer.write_morph_data(os.path.join(auxi_data_path, 'rh.904_weight.mean.sulc'), datas_mean)
    nib.freesurfer.write_morph_data(os.path.join(auxi_data_path, 'rh.904_weight.std.sulc'), datas_std)


def get_weight(feature, hemi):
    if feature == 'sulc':
        weight_data_path = os.path.join(abspath, 'auxi_data', f'{hemi}.904_weight.std.sulc')
    elif feature == 'curv':
        weight_data_path = os.path.join(abspath, 'auxi_data', f'{hemi}.904_weight.std.curv')

    weight = nib.freesurfer.read_morph_data(weight_data_path)
    return weight

if __name__ == '__main__':
    make_weight()
