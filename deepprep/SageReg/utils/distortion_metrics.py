import numpy as np
import nibabel as nib
import os
import torch
from utils.interp import resample_sphere_surface_barycentric


def set_environ():
    # FreeSurfer
    value = os.environ.get('FREESURFER_HOME')
    if value is None:
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer710'
        os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer710/subjects'
        os.environ['PATH'] = '/usr/local/freesurfer710/bin:' + os.environ['PATH']


def mris_convert(nii_surf, gii_surf):
    cmd = f'mris_convert {nii_surf} {gii_surf}'
    os.system(cmd)


def mri_convert(gii_file, mgz_file):
    cmd = f'mri_convert {gii_file} {mgz_file}'
    os.system(cmd)


def wb_surface_distortion(command, origial_sphere, deformed_sphere, metric_shape_out):
    if command == '-local-affine-method':
        cmd = f'wb_command -surface-distortion {origial_sphere} {deformed_sphere} {metric_shape_out} {command} -log2'
    else:
        cmd = f'wb_command -surface-distortion {origial_sphere} {deformed_sphere} {metric_shape_out} {command}'
    os.system(cmd)


def metric_data_interp_fs6(moved_xyz, fixed_xyz, metric_data):
    metric_data_t = torch.from_numpy(metric_data)
    moved_data_interp_fs6 = resample_sphere_surface_barycentric(
        torch.from_numpy(moved_xyz), torch.from_numpy(fixed_xyz),
        metric_data_t, device='cuda')

    return moved_data_interp_fs6.cpu().numpy()


def distortion_metrics(config, origial_sphere, deformed_sphere, tmp_dir):
    origial_sphere_gii = os.path.join(str(tmp_dir), 'lh.origial_sphere.surf.gii')
    deformed_sphere_gii = os.path.join(str(tmp_dir), 'lh.deformed_sphere.surf.gii')
    mris_convert(origial_sphere, origial_sphere_gii)
    mris_convert(deformed_sphere, deformed_sphere_gii)

    areal_shape_distortion_shape_gii = os.path.join(str(tmp_dir), 'lh.areal_shape_distortion.shape.gii')
    edge_distortion_shape_gii = os.path.join(str(tmp_dir), 'lh.edge_distortion.shape.gii')
    wb_surface_distortion('-local-affine-method', origial_sphere_gii, deformed_sphere_gii, areal_shape_distortion_shape_gii)
    wb_surface_distortion('-edge-method', origial_sphere_gii, deformed_sphere_gii, edge_distortion_shape_gii)


def distortion_metrics_interp(config, origial_sphere, deformed_sphere, tmp_dir):
    areal_shape_distortion_shape_gii = os.path.join(str(tmp_dir), 'lh.areal_shape_distortion.shape.gii')
    edge_distortion_shape_gii = os.path.join(str(tmp_dir), 'lh.edge_distortion.shape.gii')
    areal_metric_shape_data = nib.load(areal_shape_distortion_shape_gii).darrays[0].data
    shape_metric_shape_data = nib.load(areal_shape_distortion_shape_gii).darrays[1].data
    edge_metric_shape_data = nib.load(edge_distortion_shape_gii).darrays[0].data

    # ####################### cal distortion in native space ##########

    areal_distortion_value = np.mean(abs(areal_metric_shape_data))
    shape_distortion_value = np.mean(shape_metric_shape_data)
    edge_distortion_value = np.mean(edge_metric_shape_data)
    # ################ interp to fsaverage6 ###########################
    # fsaverage_dir = config["dir_fixed"]
    # interp_sphere_reg_file = origial_sphere.replace('sphere', 'sphere')
    # moved_xyz, _ = nib.freesurfer.read_geometry(interp_sphere_reg_file)
    # fixed_xyz, _ = nib.freesurfer.read_geometry(f'{fsaverage_dir}/fsaverage6/surf/lh.sphere')
    #
    # combine_data = np.concatenate([areal_metric_shape_data[:, np.newaxis],
    #                                shape_metric_shape_data[:, np.newaxis],
    #                                edge_metric_shape_data[:, np.newaxis]], axis=1)
    # combine_data_fs6 = metric_data_interp_fs6(moved_xyz, fixed_xyz, combine_data)
    #
    # areal_metric_shape_data_fs6 = combine_data_fs6[:, 0]
    # shape_metric_shape_data_fs6 = combine_data_fs6[:, 1]
    # edge_metric_shape_data_fs6 = combine_data_fs6[:, 2]
    #
    # areal_distortion_value = np.mean(abs(areal_metric_shape_data_fs6))
    # shape_distortion_value = np.mean(shape_metric_shape_data_fs6)
    # edge_distortion_value = np.mean(edge_metric_shape_data_fs6)

    # ################ remove the medial wall in fsaverage6 ###########################
    # label, _, _ = nib.freesurfer.read_annot(f'{fsaverage_dir}/fsaverage6/label/lh.aparc.annot')
    #
    # areal_metric_shape_data_de0 = areal_metric_shape_data_fs6[label != 0]
    # shape_metric_shape_data_de0 = shape_metric_shape_data_fs6[label != 0]
    # edge_metric_shape_data_de0 = edge_metric_shape_data_fs6[label != 0]
    #
    # areal_distortion_value = np.mean(abs(areal_metric_shape_data_de0))
    # shape_distortion_value = np.mean(shape_metric_shape_data_de0)
    # edge_distortion_value = np.mean(edge_metric_shape_data_de0)

    return areal_distortion_value, shape_distortion_value, edge_distortion_value


def distortion_mean():

    recon_dir = '/mnt/ngshare/SurfReg/Data_Processing/NAMIC'
    # dir_path = '/mnt/ngshare/SurfReg/Data_TrainResult/pretraineada2/NAMIC'
    # dir_path = '/mnt/ngshare/SurfReg/Data_TrainResult/predict/angle_0_1_0_4_reg_0_2_30_1_1.5_0.0003_weight_da_zscore_rdfixed/NAMIC'
    # dir_path = '/mnt/ngshare/SurfReg/Data_TrainResult/predict/new_904_sim_angle_6_0_0_2_reg_0_2_30_1_1.5_0.0003_da/NAMIC'
    # dir_path = '/run/user/1000/gvfs/sftp:host=30.30.30.136,user=lincong/mnt/ngshare/SurfReg/Data_TrainResult/pretraineada222/NAMIC'
    # dir_path = '/run/user/1000/gvfs/sftp:host=30.30.30.136,user=lincong/mnt/ngshare/SurfReg/Data_TrainResult/predict/6_0_0_2_0_1.5_35_1_1/NAMIC'
    # dir_path = '/run/user/1000/gvfs/sftp:host=30.30.30.136,user=lincong/mnt/ngshare/SurfReg/Data_TrainResult/predict/0_1_0_4_0_2_35_1_1/NAMIC'
    dir_path = '/mnt/ngshare/SurfReg/Data_TrainResult/predict_best/NAMIC'
    fsaverage_dir = '/usr/local/freesurfer/subjects'
    sub_list = os.listdir(dir_path)
    areal_datas = []
    shape_datas = []
    edge_datas = []

    areal_abs_means = []
    shape_means = []
    edge_means = []
    for sub_name in sub_list:
        if not os.path.isdir(os.path.join(dir_path, sub_name)):
            continue
        areal_file = os.path.join(dir_path, sub_name, 'surf', sub_name, 'lh.areal_shape_distortion.shape.gii')
        edge_file = os.path.join(dir_path, sub_name, 'surf', sub_name, 'lh.edge_distortion.shape.gii')
        areal_metric_shape_data = nib.load(areal_file).darrays[0].data
        shape_metric_shape_data = nib.load(areal_file).darrays[1].data
        edge_metric_shape_data = nib.load(edge_file).darrays[0].data

        # ####################### cal distortion in native space
        areal_abs_means.append(np.mean(np.abs(areal_metric_shape_data)))
        shape_means.append(np.mean(shape_metric_shape_data))
        edge_means.append(np.mean(edge_metric_shape_data))

        interp_sphere_reg_file = os.path.join(recon_dir, sub_name, 'surf', 'lh.sphere')
        moved_xyz, _ = nib.freesurfer.read_geometry(interp_sphere_reg_file)
        fixed_xyz, _ = nib.freesurfer.read_geometry(f'{fsaverage_dir}/fsaverage6/surf/lh.sphere')

        combine_data = np.concatenate([areal_metric_shape_data[:, np.newaxis],
                                       shape_metric_shape_data[:, np.newaxis],
                                       edge_metric_shape_data[:, np.newaxis]], axis=1)
        combine_data_fs6 = metric_data_interp_fs6(moved_xyz, fixed_xyz, combine_data)

        areal_metric_shape_data_fs6 = combine_data_fs6[:, 0]
        shape_metric_shape_data_fs6 = combine_data_fs6[:, 1]
        edge_metric_shape_data_fs6 = combine_data_fs6[:, 2]

        areal_datas.append(np.abs(areal_metric_shape_data_fs6))
        shape_datas.append(shape_metric_shape_data_fs6)
        edge_datas.append(edge_metric_shape_data_fs6)

        # print(np.abs(areal_metric_shape_data_fs6).mean(), shape_metric_shape_data_fs6.mean(),
        #       edge_metric_shape_data_fs6.mean())

        # label, _, _ = nib.freesurfer.read_annot(f'{fsaverage_dir}/fsaverage6/label/lh.aparc.annot')
        #
        # areal_metric_shape_data_de0 = areal_metric_shape_data_fs6[label != 0]
        # shape_metric_shape_data_de0 = shape_metric_shape_data_fs6[label != 0]
        # edge_metric_shape_data_de0 = edge_metric_shape_data_fs6[label != 0]

    print('areal distortion mean', np.mean(areal_abs_means))
    print('shape distortion mean', np.mean(shape_datas))
    print('edge  distortion mean', np.mean(edge_datas))

    out_file = os.path.join(dir_path, 'lh.areal_distortion_mean.sulc')
    nib.freesurfer.write_morph_data(out_file, np.mean(areal_datas, axis=0))
    cmd = f'mri_convert {out_file} {out_file.replace(".sulc", ".shape.gii")}'
    os.system(cmd)

    out_file = os.path.join(dir_path, 'lh.shape_distortion_mean.sulc')
    nib.freesurfer.write_morph_data(out_file, np.mean(shape_datas, axis=0))
    cmd = f'mri_convert {out_file} {out_file.replace(".sulc", ".shape.gii")}'
    os.system(cmd)

    out_file = os.path.join(dir_path, 'lh.edge_distortion_mean.sulc')
    nib.freesurfer.write_morph_data(out_file, np.mean(edge_datas, axis=0))
    cmd = f'mri_convert {out_file} {out_file.replace(".sulc", ".shape.gii")}'
    os.system(cmd)


if __name__ == '__main__':
    set_environ()
    distortion_mean()


# if __name__ == '__main__':
#     set_environ()
#     tmp_dir = Path('')
#     origial_sphere = ''
#     deformed_sphere = ''
#     tmp_dir.mkdir(exist_ok=True)
#     areal_distortion_value, shape_distortion_value, edge_distortion_value = distortion_metrics(origial_sphere,
#                                                                                                deformed_sphere)
#     shutil.rmtree(str(tmp_dir))