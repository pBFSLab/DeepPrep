import os
import torch
from sphere_registrate_norigid_multi_level_multi_fixed_predict import train_val
from utils.auxi_data import get_geometry_all_level_torch
from weight import get_weight
from pathlib import Path
import argparse
from utils.negative_area_triangle import single_remove_negative_area


def parse_args():
    parser = argparse.ArgumentParser()
    # os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/SurfReg/NAMIC_tmp'
    parser.add_argument('--hemi', help="which hemisphere")
    parser.add_argument('--sid', required=True, help='Subject ID for directory inside $SUBJECTS_DIR to be created')
    # parser.add_argument('--sid', default='sub01', help='Subject ID for directory inside $SUBJECTS_DIR to be created')
    parser.add_argument('--fsd', default=os.environ.get('FREESURFER_HOME'),
                        help='Output directory $FREESURFER_HOME (pass via environment or here)')
    parser.add_argument('--sd', default=os.environ.get('SUBJECTS_DIR'),
                        help='Output directory $SUBJECTS_DIR (pass via environment or here)')
    parser.add_argument('--model_path', required=True, help='The path of model')
    parser.add_argument('--device', default='cuda', help='Use number of cuda or cpu')

    args = parser.parse_args()
    if args.sd is None:
        raise ValueError('Subjects dir need to set via $SUBJECTS_DIR environment or --sd parameter')
    else:
        os.environ['SUBJECTS_DIR'] = args.sd
    subj_dir = os.path.join(args.sd, args.sid)
    if not os.path.exists(subj_dir):
        raise ValueError(f'{subj_dir} is not exists, please check.')
    args_dict = vars(args)

    if args.hemi is None:
        args_dict['hemi'] = ['lh', 'rh']
    else:
        args_dict['hemi'] = [args.hemi]

    if args.fsd is None:
        args_dict['fsd'] = '/usr/local/freesurfer'

    return argparse.Namespace(**args_dict)


if __name__ == '__main__':
    # abspath = os.path.abspath(os.path.dirname(__file__))
    args = parse_args()
    config = dict()
    # ========================== Predict Config ============================= #

    config["dir_fixed"] = os.path.join(args.fsd, 'subjects')  # FreeSurfer fsaverage6 目录
    config["dir_predict_recon"] = os.path.join(args.sd, args.sid)  # native recon dir
    config["dir_predict_rigid"] = os.path.join(args.sd, args.sid, 'tmp')  # rigid predict result dir
    config["dir_predict_result"] = os.path.join(args.sd, args.sid) # norigid predict result dir

    # ========================== Default Config ============================= #

    xyzs, faces = get_geometry_all_level_torch()
    config['xyz'] = xyzs
    config['face'] = faces
    ico_level = 'fsaverage6'

    config["device"] = args.device
    if config['device'] != 'cpu':
        config['device'] = 'cuda'
    config['validation'] = True
    config['rd_sample'] = False
    config['is_da'] = True
    config["ico_level"] = ico_level
    config["model_name"] = "GatUNet"
    sulc_curv_weight = [[1, 0], [1, 0], [1, 0], [1, 0]]
    config["n_vertex"] = 40962  # 当前细化等级的顶点数量163842 40962
    config["normalize_type"] = 'zscore'  # 计算与相邻顶点的push距离
    config["feature"] = ['sulc', 'sulc', 'sulc', 'sulc']

    # ############### rigid predict #########################

    ico_levels = ['fsaverage6']
    ico_index = ico_levels.index(ico_level)
    config['ico_levels'] = ico_levels
    config['ico_index'] = ico_index

    config['is_rigid'] = True

    rigid_model_result_dir = [f'{args.model_path}/fsaverage6']
    for hemi in args.hemi:
        config['sim_weight'] = torch.from_numpy(get_weight('sulc', hemi).astype(float)).float()
        config["hemisphere"] = hemi
        model_files = []
        for i in range(len(ico_levels)):
            model_files.append(os.path.join(rigid_model_result_dir[i],
                                            f'{config["hemisphere"]}_Rigid_904_{ico_levels[i]}.model'))
        config["model_files"] = model_files
        train_val(config=config)
    # exit()
# ############### norigid predict #########################

    ico_levels = ['fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6']
    ico_index = ico_levels.index(ico_level)
    config['ico_levels'] = ico_levels
    config['ico_index'] = ico_index
    config['is_rigid'] = False
    norigid_model_result_dir = [
        f'{args.model_path}/fsaverage3',
        f'{args.model_path}/fsaverage4',
        f'{args.model_path}/fsaverage5',
        f'{args.model_path}/fsaverage6'
    ]
    for hemi in args.hemi:
        config["hemisphere"] = hemi
        config['sim_weight'] = torch.from_numpy(get_weight('sulc', hemi).astype(float)).float()
        model_files = []
        for i in range(len(ico_levels)):
            model_files.append(os.path.join(norigid_model_result_dir[i],
                                            f'{config["hemisphere"]}_NoRigid_904_{ico_levels[i]}.model'))
        config["model_files"] = model_files
        train_val(config=config)
        # remove negative area triangles
        sphere_moved_native_neg_file = os.path.join(args.sd, args.sid, 'surf', f'{hemi}.sphere.reg')
        sphere_moved_native_file = os.path.join(args.sd, args.sid, 'surf', f'{hemi}.sphere.reg')
        single_remove_negative_area(sphere_moved_native_neg_file, sphere_moved_native_file, device=config['device'])
