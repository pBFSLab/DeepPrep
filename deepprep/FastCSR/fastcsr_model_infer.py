import os
import glob
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.preprocessing.preprocessing import GenericPreprocessor
import SimpleITK as sitk
import ants


class LevelsetPredictor(object):
    def __init__(self, hemi, device, model_path: Path = Path('model')):
        self.hemi = hemi
        self.device = device
        if self.hemi == 'lh':
            self.model_file = model_path / 'lh_model.pth'
        elif self.hemi == 'rh':
            self.model_file = model_path / 'rh_model.pth'
        else:
            raise Exception('hemi parameter error')

        params = torch.load(self.model_file, map_location=device)
        self.plans = params['plans']
        self.network = self.create_network()
        self.network.load_state_dict(params['state_dict'])
        self.network.eval()

    # 模型结构创建
    def create_network(self):
        base_num_features = self.plans['base_num_features']
        num_input_channels = self.plans['num_modalities']
        conv_per_stage = self.plans['conv_per_stage']
        network_plans = self.plans['plans_per_stage'][0]
        net_conv_kernel_sizes = network_plans['conv_kernel_sizes']
        net_num_pool_op_kernel_sizes = network_plans['pool_op_kernel_sizes']
        net_numpool = len(net_num_pool_op_kernel_sizes)
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        final_nonlin = lambda x: x
        output_channels = 1
        network = Generic_UNet(num_input_channels, base_num_features, output_channels, net_numpool,
                               conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                               dropout_op_kwargs,
                               net_nonlin, net_nonlin_kwargs, False, False, final_nonlin, InitWeights_He(1e-2),
                               net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
        network.to(device)
        return network

    # 模型前传
    def infer(self, input_files, levelset_file):
        # preprocess
        normalization_schemes = self.plans['normalization_schemes']
        use_mask_for_norm = self.plans['use_mask_for_norm']
        transpose_forward = self.plans['transpose_forward']
        intensity_properties = self.plans['dataset_properties']['intensityproperties']
        current_spacing = self.plans['plans_per_stage'][0]['current_spacing']
        preprocessor = GenericPreprocessor(normalization_schemes, use_mask_for_norm, transpose_forward,
                                           intensity_properties)
        data, seg, properties = preprocessor.preprocess_test_case(input_files, current_spacing)

        all_in_gpu = False
        pad_border_mode = 'constant'
        pad_kwargs = {'constant_values': 0}
        patch_size = self.plans['plans_per_stage'][0]['patch_size']
        mirror_axes = (0, 1, 2)
        # model infer
        pred = self.network.predict_3D(data, do_mirroring=True, mirror_axes=mirror_axes,
                                       use_sliding_window=True, step_size=0.5,
                                       patch_size=patch_size, regions_class_order=None,
                                       use_gaussian=True, pad_border_mode=pad_border_mode,
                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=True,
                                       mixed_precision=True)[1]

        shape_original_before_cropping = properties['original_size_of_raw_data']
        bbox = properties['crop_bbox']
        levelset_np = np.zeros(shape_original_before_cropping)
        levelset_np[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = pred[0]
        levelset_itk = sitk.GetImageFromArray(levelset_np.astype(np.float32))
        levelset_itk.SetSpacing(properties['itk_spacing'])
        levelset_itk.SetOrigin(properties['itk_origin'])
        levelset_itk.SetDirection(properties['itk_direction'])
        sitk.WriteImage(levelset_itk, levelset_file)

    # 调用入口
    def process(self, input_files, fastcsr_subjects_dir, subj, hemi):
        # levelset为模型预测结果
        levelset_file = fastcsr_subjects_dir / subj / 'mri' / f'{hemi}_levelset.nii.gz'
        self.infer(input_files, str(levelset_file))


# 依赖文件：
#     freesurfer_subjects_dir/$subj/mri/orig.mgz
#     freesurfer_subjects_dir/$subj/mri/filled.mgz
# 输出文件：
#     $outputpath/${subj}_0000.nii.gz'
#     $outputpath/${subj}_0001.nii.gz'
def convert_data(inputpath: Path, outputpath: Path, subj):
    # orig
    in_orig_file = inputpath / subj / 'mri' / 'orig.mgz'
    orig = ants.image_read(str(in_orig_file))
    out_orig_file = outputpath / f'{subj}_0000.nii.gz'
    ants.image_write(orig, str(out_orig_file))

    # filled
    in_filled_file = inputpath / subj / 'mri' / 'filled.mgz'
    filled = ants.image_read(str(in_filled_file))
    out_filled_file = outputpath / f'{subj}_0001.nii.gz'
    ants.image_write(filled, str(out_filled_file))


# 依赖文件：
#     freesurfer_subjects_dir/$subj/mri/orig.mgz
#     freesurfer_subjects_dir/$subj/mri/filled.mgz
#     freesurfer_subjects_dir/$subj/mri/brainmask.mgz
#     freesurfer_subjects_dir/$subj/mri/aseg.presurf.mgz
# 输出文件：
#     freesurfer_subjects_dir/$subj/mri/?h_levelset.nii.gz
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fastcsr_subjects_dir', required=True)
    parser.add_argument('--subj', required=True)
    parser.add_argument('--hemi', required=True, choices=['lh', 'rh'])
    parser.add_argument('--suffix', default='orig.nofix', choices=['orig.nofix', 'orig'])
    args = parser.parse_args()
    fastcsr_subjects_dir = Path(args.fastcsr_subjects_dir)
    subj = args.subj
    hemi = args.hemi

    # 模型输入文件存储临时目录
    input_path = fastcsr_subjects_dir / subj / 'tmp' / f'{hemi}_input'
    os.makedirs(input_path, exist_ok=True)
    # 准备深度学习模型输入
    convert_data(fastcsr_subjects_dir, input_path, subj)

    # 深度学习模型运行设备
    if torch.cuda.is_available():
        device = torch.device(type='cuda', index=0)
    else:
        device = torch.device('cpu')
    # 模型初始化
    model_path, _ = os.path.split(os.path.abspath(__file__))
    model_path = Path(model_path) / 'model'
    levelset_model = LevelsetPredictor(hemi=hemi, device=device, model_path=model_path)
    # 输入文件
    input_files = sorted(glob.glob(str(input_path / '*.nii.gz')))
    # FactCSR处理过程
    levelset_model.process(input_files, fastcsr_subjects_dir, subj, hemi)
    shutil.rmtree(input_path)
