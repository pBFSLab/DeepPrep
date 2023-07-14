# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : NingAnMe <ninganme0317@gmail.com>
# @Author : Cong Lin       @Email : lincong <lincong8722@gmail.com>
# @Author : Youjia Zhang   @Email : youjia <ireneyou33@gmail.com>
# @Author : Zhenyu Sun     @Email : Kid-sunzhenyu <sun25939789@gmail.com>

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, File, TraitedSpec, Directory, Str
import sh
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path
import bids
import os
import tensorflow as tf
import ants
import voxelmorph as vxm


class VxmRegistraionInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(desc="subject id", mandatory=True)
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    data_path = Directory(exists=True, desc='data path', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    model_file = File(exists=True, desc="atlas_type/model.h5", mandatory=True)
    vxm_model_path = Directory(exists=True, desc="model/voxelmorph", mandatory=True)
    atlas_type = Str(desc="atlas_type", mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    gpuid = Str(exists=True, desc='gpuid set', mandatory=True)


class VxmRegistraionOutputSpec(TraitedSpec):
    subjects_dir = Directory(exists=True, desc='subjects_dir')


class VxmRegistraion(BaseInterface):
    input_spec = VxmRegistraionInputSpec
    output_spec = VxmRegistraionOutputSpec

    time = 20 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 1  # 最大cpu占用：个
    gpu = 2703  # 最大gpu占用：MB

    def __init__(self):
        super(VxmRegistraion, self).__init__()

    def check_output(self, output_dir: Path):
        subject_id = self.inputs.subject_id
        atlas_type = self.inputs.atlas_type
        VxmRegistraion_output_files = [f'{subject_id}_norm_space-vxm{atlas_type}.nii.gz',  # {subject_id}_norm_space-vxm{atlas_type}.nii.gz
                                       f'{subject_id}_norm_space-{atlas_type}.nii.gz',  # {subject_id}_norm_space-{atlas_type}.nii.gz
                                       f'{subject_id}_from_fsnative_to_vxm{atlas_type}_vxm_nonrigid.nii.gz',  # {subject_id}_from_fsnative_to_vxm{atlas_type}_vxm_nonrigid.nii.gz
                                       f'{subject_id}_norm_affine_space-vxm{atlas_type}.npz',  # {subject_id}_norm_affine_space-vxm{atlas_type}.npz
                                       f'{subject_id}_from_fsnative_to_vxm{atlas_type}_ants_affine.mat'  # {subject_id}_from_fsnative_to_vxm{atlas_type}_ants_affine.mat
                                       ]
        for filename in VxmRegistraion_output_files:
            if not (output_dir / filename).exists():
                return FileExistsError(output_dir / filename)

    def _run_interface(self, runtime):
        subject_id = self.inputs.subject_id
        deepprep_subj_path = Path(self.inputs.derivative_deepprep_path) / subject_id
        atlas_type = self.inputs.atlas_type

        func_dir = Path(deepprep_subj_path) / 'func'
        anat_dir = Path(deepprep_subj_path) / 'anat'
        func_dir.mkdir(parents=True, exist_ok=True)
        anat_dir.mkdir(parents=True, exist_ok=True)

        norm = Path(self.inputs.subjects_dir) / subject_id / 'mri' / 'norm.mgz'

        trf_fsnative2vxmatlask_affine_path = anat_dir / f'{subject_id}_from_fsnative_to_vxm{atlas_type}_ants_affine.mat'  # fromaffine trf from native T1 to vxm_MNI152
        vxm_warp = anat_dir / f'{subject_id}_from_fsnative_to_vxm{atlas_type}_vxm_nonrigid.nii.gz'  # from_fsnative_to_vxm{atlas_type}_norigid.nii.gz norigid warp from native T1 to vxm_MNI152
        trf_vxmatlas2atlask_rigid_path = anat_dir / f'{subject_id}_from_vxm{atlas_type}_to_{atlas_type}_ants_rigid.mat'

        vxm_input_npz = anat_dir / f'{subject_id}_norm_affine_space-vxm{atlas_type}.npz'  # from_fsnative_to_vxm{atlas_type}_affined.npz
        vxm_warped_path = anat_dir / f'{subject_id}_norm_space-vxm{atlas_type}.nii.gz'
        warped_path = anat_dir / f'{subject_id}_norm_space-{atlas_type}.nii.gz'

        # atlas and model
        vxm_model_path = Path(self.inputs.vxm_model_path)
        model_file = self.inputs.model_file  # vxm_model_path / atlas_type / f'model.h5'
        atlas_path = vxm_model_path / atlas_type / f'{atlas_type}_brain.nii.gz'  # MNI152空间模板
        vxm_atlas_path = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.nii.gz'  # vxm_MNI152空间模板
        vxm_atlas_npz_path = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.npz'
        vxm2atlas_trf = vxm_model_path / atlas_type / f'{atlas_type}_vxm2atlas.mat'  # from vxm_MNI152_nraoi to MNI152

        # ####################### ants affine transform norm to vxm_atlas
        norm = ants.image_read(str(norm))
        vxm_atlas = ants.image_read(str(vxm_atlas_path))
        tx = ants.registration(fixed=vxm_atlas, moving=norm, type_of_transform='Affine')  # 将

        # save moved
        affined = tx['warpedmovout']  # vxm的输入，应用deformation_field，输出moved
        vol = affined.numpy() / 255.0  # vxm模型输入，输入模型用来计算deformation_field
        np.savez_compressed(vxm_input_npz, vol=vol)  # vxm输入，
        # save transforms matrix
        fwdtransforms_file = Path(tx['fwdtransforms'][0])
        shutil.copy(fwdtransforms_file, trf_fsnative2vxmatlask_affine_path)

        # ####################### voxelmorph norigid
        # tensorflow device handling
        if 'cuda' in self.inputs.gpuid:
            if len(self.inputs.gpuid) == 4:
                deepprep_device = '0'
            else:
                deepprep_device = self.inputs.gpuid.split(":")[1]
        else:
            deepprep_device = -1
        device, nb_devices = vxm.tf.utils.setup_device(deepprep_device)

        # load moving and fixed images
        moving_divide_255 = vxm.py.utils.load_volfile(str(vxm_input_npz), add_batch_axis=True,
                                                      add_feat_axis=True)
        fixed, fixed_affine = vxm.py.utils.load_volfile(str(vxm_atlas_npz_path), add_batch_axis=True,
                                                        add_feat_axis=True,
                                                        ret_affine=True)
        vxm_atlas_nib = nib.load(str(vxm_atlas_path))
        fixed_affine = vxm_atlas_nib.affine.copy()
        inshape = moving_divide_255.shape[1:-1]
        nb_feats = moving_divide_255.shape[-1]

        with tf.device(device):
            # load model and predict
            # TODO now not using SemiSupervisedModel,maybe accuracy have been influenced
            # warp = vxm.networks.VxmDenseSemiSupervisedSeg.load(args.model).register(moving, fixed)
            warp = vxm.networks.VxmDense.load(model_file).register(moving_divide_255, fixed)
            moving = affined.numpy()[np.newaxis, ..., np.newaxis]
            moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])  # if combine transform，need to know how to trans vxm_trf to ants_trf

        # save warp from norm_affine to vxmatlas
        vxm.py.utils.save_volfile(warp.squeeze(), str(vxm_warp), fixed_affine)

        # save moved image
        vxm.py.utils.save_volfile(moved.squeeze(), str(vxm_warped_path), fixed_affine)

        # from vxmatlas to atlas
        atlas = ants.image_read(str(atlas_path))
        vxm_warped = ants.image_read(str(vxm_warped_path))
        warped = ants.apply_transforms(fixed=atlas, moving=vxm_warped, transformlist=[str(vxm2atlas_trf)])
        ants.image_write(warped, str(warped_path))
        # copy trf from vxmatlas to atlas
        shutil.copy(vxm2atlas_trf, trf_vxmatlas2atlask_rigid_path)

        self.check_output(anat_dir)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subjects_dir"] = self.inputs.subjects_dir
        return outputs

    def create_sub_node(self, settings):
        if settings.BOLD_ONLY:
            from deepprep.interface.create_node_bold import create_VxmRegNormMNI152_node
            node = create_VxmRegNormMNI152_node(self.inputs.subject_id,
                                                self.inputs.task,
                                                self.inputs.atlas_type,
                                                self.inputs.preprocess_method,
                                                settings)
            return node
        return []


class VxmRegNormMNI152InputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    data_path = Directory(exists=True, desc='data_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    vxm_model_path = Directory(exists=True, desc='model/voxelmorph', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)
    resource_dir = Directory(exists=True, desc='resource', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    gpuid = Str(exists=True, desc='gpuid set', mandatory=True)
    batch_size = Str(exists=True, desc='batch size for interpret', mandatory=True)


class VxmRegNormMNI152OutputSpec(TraitedSpec):
    subject_id = Str(exists=True, desc='subject')
    data_path = Directory(exists=True, desc='data_path')


class VxmRegNormMNI152(BaseInterface):
    input_spec = VxmRegNormMNI152InputSpec
    output_spec = VxmRegNormMNI152OutputSpec

    time = 217 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(VxmRegNormMNI152, self).__init__()

    def check_output(self, output_bolds: list):
        for bold in output_bolds:
            if not bold.exists():
                raise FileExistsError(bold)

    def register_dat_to_fslmat(self, bold_mc_file, norm_fsnative_2mm_file, register_dat_file, fslmat_file):
        sh.tkregister2('--mov', bold_mc_file,
                       '--targ', norm_fsnative_2mm_file,
                       '--reg', register_dat_file,
                       '--fslregout', fslmat_file,
                       '--noedit')
        (register_dat_file.parent / register_dat_file.name.replace('.dat', '.dat~')).unlink(missing_ok=True)

    def register_dat_to_trf(self, bold_mc_file: Path, norm_fsnative_2mm_file, register_dat_file, ants_trf_file):
        import SimpleITK as sitk

        tmp_dir = bold_mc_file.parent / bold_mc_file.name.replace('.nii.gz', '')
        tmp_dir.mkdir(exist_ok=True)

        fsltrf_file = tmp_dir / bold_mc_file.name.replace('.nii.gz', f'_from_mc_to_norm_2mm_fsl_rigid.fsl')
        self.register_dat_to_fslmat(bold_mc_file, norm_fsnative_2mm_file, register_dat_file, fsltrf_file)

        c3d_affine_tool = Path(self.inputs.resource_dir) / 'c3d_affine_tool'
        template_file = bold_mc_file.parent / bold_mc_file.name.replace('_mc.nii.gz', '_boldref.nii.gz')
        tfm_file = tmp_dir / bold_mc_file.name.replace('.nii.gz', f'_from_mc_to_norm2mm_itk_rigid.tfm')
        cmd = f'{c3d_affine_tool} -ref {norm_fsnative_2mm_file} -src {template_file} {fsltrf_file} -fsl2ras -oitk {tfm_file}'
        os.system(cmd)

        trf_sitk = sitk.ReadTransform(str(tfm_file))
        trf = ants.new_ants_transform()
        trf.set_parameters(trf_sitk.GetParameters())
        trf.set_fixed_parameters(trf_sitk.GetFixedParameters())
        ants.write_transform(trf, ants_trf_file)

        shutil.rmtree(tmp_dir)

    def bold_mc_to_fsnative2mm_ants(self, bold_mc_file: Path, norm_fsnative2mm_file, register_dat_file,
                                    bold_fsnative2mm_file: str, func_dir: Path, output_bolds, verbose=False):
        """
        bold_mc_file : moving
        norm_fsnative_file : norm.mgz
        register_dat_file : bbregister.register.dat
        """

        # 将bbregister dat文件转换为ants trf mat文件
        ants_rigid_trf_file = func_dir / bold_mc_file.name.replace('.nii.gz', '_from_mc_to_fsnative_ants_rigid.mat')
        self.register_dat_to_trf(bold_mc_file, norm_fsnative2mm_file, register_dat_file, ants_rigid_trf_file)

        bold_img = ants.image_read(str(bold_mc_file))
        fixed = ants.image_read(str(norm_fsnative2mm_file))
        affined_bold_img = ants.apply_transforms(fixed=fixed, moving=bold_img, transformlist=[str(ants_rigid_trf_file)], imagetype=3)

        if verbose:
            affine_info = nib.load(norm_fsnative2mm_file).affine
            header_info = nib.load(bold_mc_file).header
            affined_nib_img = nib.Nifti1Image(affined_bold_img.numpy().astype(int), affine=affine_info, header=header_info)
            nib.save(affined_nib_img, bold_fsnative2mm_file)
            output_bolds.append(bold_fsnative2mm_file)

        return affined_bold_img

    def vxm_warp_bold_2mm(self, bold_fsnative2mm, bold_fsnative2mm_file,
                          trt_ants_affine_file, trt_vxm_norigid_file, warped_file, output_bolds, verbose=True):
        import voxelmorph as vxm

        vxm_model_path = Path(self.inputs.vxm_model_path)
        atlas_type = self.inputs.atlas_type

        vxm_atlas_file = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.nii.gz'
        MNI152_2mm_file = vxm_model_path / atlas_type / f'{atlas_type}_brain.nii.gz'
        MNI152_2mm = ants.image_read(str(MNI152_2mm_file))
        vxm_atlas = ants.image_read(str(vxm_atlas_file))
        if isinstance(bold_fsnative2mm, str):
            bold_img = ants.image_read(bold_fsnative2mm)
        else:
            bold_img = bold_fsnative2mm
        n_frame = bold_img.shape[3]
        bold_origin = bold_img.origin
        bold_spacing = bold_img.spacing
        bold_direction = bold_img.direction.copy()

        # tensorflow device handling
        if 'cuda' in self.inputs.gpuid:
            if len(self.inputs.gpuid) == 4:
                deepprep_device = '0'
            else:
                deepprep_device = self.inputs.gpuid.split(":")[1]
        else:
            deepprep_device = -1
        device, nb_devices = vxm.tf.utils.setup_device(deepprep_device)

        fwdtrf_MNI152_2mm = [str(trt_ants_affine_file)]
        trf_file = vxm_model_path / atlas_type / f'{atlas_type}_vxm2atlas.mat'
        fwdtrf_atlas2MNI152_2mm = [str(trf_file)]
        deform, deform_affine = vxm.py.utils.load_volfile(str(trt_vxm_norigid_file), add_batch_axis=True, ret_affine=True)

        # affine to MNI152 croped
        affined_np = ants.apply_transforms(vxm_atlas, bold_img, fwdtrf_MNI152_2mm, imagetype=3).numpy()

        # voxelmorph warp
        warped_np = np.zeros(shape=(*vxm_atlas.shape, n_frame), dtype=np.float32)
        with tf.device(device):
            transform = vxm.networks.Transform(vxm_atlas.shape, interp_method='linear', nb_feats=1)
            tf_dataset = tf.data.Dataset.from_tensor_slices(np.transpose(affined_np, (3, 0, 1, 2)))
            del affined_np
            batch_size = int(self.inputs.batch_size)
            deform = tf.convert_to_tensor(deform)
            deform = tf.keras.backend.tile(deform, [batch_size, 1, 1, 1, 1])
            for idx, batch_data in enumerate(tf_dataset.batch(batch_size=batch_size)):
                if batch_data.shape[0] != deform.shape[0]:
                    deform = deform[:batch_data.shape[0], :, :, :, :]
                moved = transform.predict([batch_data, deform]).squeeze()
                if len(moved.shape) == 4:
                    moved_data = np.transpose(moved, (1, 2, 3, 0))
                else:
                    moved_data = moved[:, :, :, np.newaxis]
                warped_np[:, :, :, idx * batch_size:(idx + 1) * batch_size] = moved_data
            del transform
            del tf_dataset
            del moved
            del moved_data

        # affine to MNI152
        origin = (*vxm_atlas.origin, bold_origin[3])
        spacing = (*vxm_atlas.spacing, bold_spacing[3])
        direction = bold_direction.copy()
        direction[:3, :3] = vxm_atlas.direction

        warped_img = ants.from_numpy(warped_np, origin=origin, spacing=spacing, direction=direction)
        del warped_np
        moved_img = ants.apply_transforms(MNI152_2mm, warped_img, fwdtrf_atlas2MNI152_2mm, imagetype=3)
        del warped_img

        if verbose:
            affine_info = nib.load(MNI152_2mm_file).affine
            header_info = nib.load(bold_fsnative2mm_file).header
            nib_img = nib.Nifti1Image(moved_img.numpy().astype(int), affine=affine_info, header=header_info)
            nib.save(nib_img, warped_file)
            output_bolds.append(warped_file)
        return moved_img

    def _run_interface(self, runtime):
        preprocess_dir = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        subj = self.inputs.subject_id.split('-')[1]
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj_func_dir = preprocess_dir / 'func'
        subj_anat_dir = Path(preprocess_dir) / 'anat'
        subj_func_dir.mkdir(parents=True, exist_ok=True)

        subject_id = self.inputs.subject_id
        atlas_type = self.inputs.atlas_type

        norm_fsnative_file = Path(self.inputs.subjects_dir) / subject_id / 'mri' / 'norm.mgz'
        norm_fsnative2mm_file = subj_anat_dir / f'{subject_id}_norm_2mm.nii.gz'
        if not norm_fsnative2mm_file.exists():
            sh.mri_convert('-ds', 2, 2, 2,
                           '-i', norm_fsnative_file,
                           '-o', norm_fsnative2mm_file)

        args = []
        bold_files = []
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        output_bolds = []
        for idx, bids_bold in enumerate(bids_bolds):
            bold_file = Path(bids_bold.path)
            bold_files.append(bold_file)
            args.append([subj_func_dir, bold_file])

            bold_mc_file = subj_func_dir / bold_file.name.replace('.nii.gz', '_skip_reorient_faln_mc.nii.gz')

            register_dat_file = subj_func_dir / bold_file.name.replace('.nii.gz',
                                                                       '_skip_reorient_faln_mc_from_mc_to_fsnative_bbregister_rigid.dat')
            bold_fsnative2mm_file = subj_func_dir / bold_file.name.replace('.nii.gz',
                                                                           '_skip_reorient_faln_mc_space-fsnative2mm.nii.gz')  # save reg to T1 result file

            bold_fsnative2mm_img = self.bold_mc_to_fsnative2mm_ants(bold_mc_file, norm_fsnative2mm_file, register_dat_file,
                                                                    str(bold_fsnative2mm_file), subj_func_dir,
                                                                    output_bolds, verbose=True)

            ants_affine_trt_file = subj_anat_dir / f'{subject_id}_from_fsnative_to_vxm{atlas_type}_ants_affine.mat'
            vxm_nonrigid_trt_file = subj_anat_dir / f'{subject_id}_from_fsnative_to_vxm{atlas_type}_vxm_nonrigid.nii.gz'
            bold_atlas_file = subj_func_dir / bold_file.name.replace('.nii.gz',
                                                                     f'_skip_reorient_faln_mc_space-{atlas_type}.nii.gz')  # save reg to MNI152 result file
            self.vxm_warp_bold_2mm(bold_fsnative2mm_img, bold_mc_file,
                                   ants_affine_trt_file, vxm_nonrigid_trt_file, bold_atlas_file,
                                   output_bolds, verbose=True)

        self.check_output(output_bolds)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        outputs["data_path"] = self.inputs.data_path

        return outputs

    def create_sub_node(self):
        return []
