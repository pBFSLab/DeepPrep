from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, File, TraitedSpec, Directory, Str
import sh
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
        VxmRegistraion_output_files = [f'{subject_id}_norm_norigid_vxm_{atlas_type}.nii.gz',
                                       f'{subject_id}_norm_norigid_{atlas_type}.nii.gz',
                                       f'{subject_id}_vxm_deformation_field_from_norm_to_vxm_{atlas_type}.nii.gz',
                                       f'{subject_id}_norm_affined_vxm_{atlas_type}.npz',
                                       f'{subject_id}_ants_affine_trf_from_norm_to_vxm_{atlas_type}.mat']
        for filename in VxmRegistraion_output_files:
            if not (output_dir / filename).exists():
                return FileExistsError(output_dir / filename)

    def _run_interface(self, runtime):
        subject_id = self.inputs.subject_id
        deepprep_subj_path = Path(self.inputs.derivative_deepprep_path) / subject_id
        atlas_type = self.inputs.atlas_type

        func_dir = Path(deepprep_subj_path) / 'func'
        transform_dir = Path(deepprep_subj_path) / 'transform'
        func_dir.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)

        norm = Path(self.inputs.subjects_dir) / subject_id / 'mri' / 'norm.mgz'

        trf_path = transform_dir / f'{subject_id}_ants_affine_trf_from_norm_to_vxm_{atlas_type}.mat'  # affine trf from native T1 to vxm_MNI152
        vxm_warp = transform_dir / f'{subject_id}_vxm_deformation_field_from_norm_to_vxm_{atlas_type}.nii.gz'  # norigid warp from native T1 to vxm_MNI152
        vxm_input_npz = transform_dir / f'{subject_id}_norm_affined_vxm_{atlas_type}.npz'  # npz

        vxm_warped_path = func_dir / f'{subject_id}_norm_norigid_vxm_{atlas_type}.nii.gz'
        warped_path = func_dir / f'{subject_id}_norm_norigid_{atlas_type}.nii.gz'

        # atlas and model
        vxm_model_path = Path(self.inputs.vxm_model_path)
        model_file = self.inputs.model_file  # vxm_model_path / atlas_type / f'model.h5'
        atlas_path = vxm_model_path / atlas_type / f'{atlas_type}_brain.nii.gz'  # MNI152空间模板
        vxm_atlas_path = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.nii.gz'  # vxm_MNI152空间模板
        vxm_atlas_npz_path = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.npz'
        vxm2atlas_trf = vxm_model_path / atlas_type / f'{atlas_type}_vxm2atlas.mat'  # trf from vxm_MNI152 to MNI152

        # ####################### ants affine transform norm to vxm_atlas
        norm = ants.image_read(str(norm))
        vxm_atlas = ants.image_read(str(vxm_atlas_path))
        tx = ants.registration(fixed=vxm_atlas, moving=norm, type_of_transform='Affine')
        trf = ants.read_transform(tx['fwdtransforms'][0])
        ants.write_transform(trf, str(trf_path))
        affined = tx['warpedmovout']  # vxm的输入，应用deformation_field，输出moved
        vol = affined.numpy() / 255.0  # vxm模型输入，输入模型用来计算deformation_field
        np.savez_compressed(vxm_input_npz, vol=vol)  # vxm输入，

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
        add_feat_axis = True
        moving_divide_255 = vxm.py.utils.load_volfile(str(vxm_input_npz), add_batch_axis=True,
                                                      add_feat_axis=add_feat_axis)
        fixed, fixed_affine = vxm.py.utils.load_volfile(str(vxm_atlas_npz_path), add_batch_axis=True,
                                                        add_feat_axis=add_feat_axis,
                                                        ret_affine=True)
        vxm_atlas_nib = nib.load(str(vxm_atlas_path))
        fixed_affine = vxm_atlas_nib.affine.copy()
        inshape = moving_divide_255.shape[1:-1]
        nb_feats = moving_divide_255.shape[-1]

        with tf.device(device):
            # load model and predict
            warp = vxm.networks.VxmDense.load(model_file).register(moving_divide_255, fixed)
            # warp = vxm.networks.VxmDenseSemiSupervisedSeg.load(args.model).register(moving, fixed)
            moving = affined.numpy()[np.newaxis, ..., np.newaxis]
            moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

        # save warp
        vxm.py.utils.save_volfile(warp.squeeze(), str(vxm_warp), fixed_affine)

        # save moved image
        vxm.py.utils.save_volfile(moved.squeeze(), str(vxm_warped_path), fixed_affine)

        # affine to atlas
        atlas = ants.image_read(str(atlas_path))
        vxm_warped = ants.image_read(str(vxm_warped_path))
        warped = ants.apply_transforms(fixed=atlas, moving=vxm_warped, transformlist=[str(vxm2atlas_trf)])
        ants.image_write(warped, str(warped_path))

        self.check_output(transform_dir)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subjects_dir"] = self.inputs.subjects_dir
        return outputs

    def create_sub_node(self):
        if self.bold_only == 'True':
            from interface.create_node_bold_new import create_VxmRegNormMNI152_node
            node = create_VxmRegNormMNI152_node(self.inputs.subject_id,
                                                self.inputs.task,
                                                self.inputs.atlas_type,
                                                self.inputs.preprocess_method)
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

    def register_dat_to_fslmat(self, mov_file, ref_file, reg_file, fslmat_file):
        sh.tkregister2('--mov', mov_file,
                       '--targ', ref_file,
                       '--reg', reg_file,
                       '--fslregout', fslmat_file,
                       '--noedit')

    def register_dat_to_trf(self, mov_file: Path, subject_id, ref_file, reg_file, transform_dir, trf_file):
        import SimpleITK as sitk

        fsltrf_file = os.path.join(transform_dir, f'{subject_id}_fsl_tkregister2_trf_from_reg_bold_to_norm_2mm.fsl')
        self.register_dat_to_fslmat(mov_file, ref_file, reg_file, fsltrf_file)
        # create frame0  # 直接使用subj/func/template.nii.gz 不可以，template.nii.gz 没有进行mc操作
        first_frame_file = mov_file.parent / mov_file.name.replace('.nii.gz', '_frame0.nii.gz')
        sh.mri_convert(mov_file, first_frame_file, '--frame', 0)

        tfm_file = os.path.join(transform_dir, f'{subject_id}_itk_trf_from_reg_bold_to_norm_2mm.tfm')
        c3d_affine_tool = Path(self.inputs.resource_dir) / 'c3d_affine_tool'
        cmd = f'{c3d_affine_tool} -ref {ref_file} -src {first_frame_file} {fsltrf_file} -fsl2ras -oitk {tfm_file}'
        os.system(cmd)
        trf_sitk = sitk.ReadTransform(tfm_file)
        trf = ants.new_ants_transform()
        trf.set_parameters(trf_sitk.GetParameters())
        trf.set_fixed_parameters(trf_sitk.GetFixedParameters())
        ants.write_transform(trf, trf_file)

    def native_bold_to_T1_2mm_ants(self, bold_file: Path, subject_id, subj_t1_file, reg_file, save_file: str,
                                   func_dir, transform_dir, verbose=False):
        """
        bold_file : moving
        subj_t1_file : norm_2mm.nii.gz
        reg_file : bbregister.register.dat
        """
        subj_t1_2mm_file = os.path.join(func_dir, f'{subject_id}_norm_2mm.nii.gz')
        sh.mri_convert('-ds', 2, 2, 2,
                       '-i', subj_t1_file,
                       '-o', subj_t1_2mm_file)
        trf_file = os.path.join(transform_dir, f'{subject_id}_ants_trf_from_reg_bold_to_norm_2mm.mat')
        self.register_dat_to_trf(bold_file, subject_id, subj_t1_2mm_file, reg_file, transform_dir, trf_file)
        bold_img = ants.image_read(str(bold_file))
        fixed = ants.image_read(subj_t1_2mm_file)
        affined_bold_img = ants.apply_transforms(fixed=fixed, moving=bold_img, transformlist=[trf_file], imagetype=3)
        if verbose:
            ants.image_write(affined_bold_img, save_file)
        return affined_bold_img

    def vxm_warp_bold_2mm(self, resid_t1, affine_file, warp_file, warped_file, verbose=True):
        import voxelmorph as vxm
        import tensorflow as tf

        vxm_model_path = Path(self.inputs.vxm_model_path)
        atlas_type = self.inputs.atlas_type

        vxm_atlas_file = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.nii.gz'
        MNI152_2mm_file = vxm_model_path / atlas_type / f'{atlas_type}_brain.nii.gz'
        MNI152_2mm = ants.image_read(str(MNI152_2mm_file))
        vxm_atlas = ants.image_read(str(vxm_atlas_file))
        if isinstance(resid_t1, str):
            bold_img = ants.image_read(resid_t1)
        else:
            bold_img = resid_t1
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

        fwdtrf_MNI152_2mm = [str(affine_file)]
        trf_file = vxm_model_path / atlas_type / f'{atlas_type}_vxm2atlas.mat'
        fwdtrf_atlas2MNI152_2mm = [str(trf_file)]
        deform, deform_affine = vxm.py.utils.load_volfile(str(warp_file), add_batch_axis=True, ret_affine=True)

        # affine to MNI152 croped
        affined_np = ants.apply_transforms(vxm_atlas, bold_img, fwdtrf_MNI152_2mm, imagetype=3).numpy()

        # voxelmorph warp
        warped_np = np.zeros(shape=(*vxm_atlas.shape, n_frame), dtype=np.float32)
        with tf.device(device):
            transform = vxm.networks.Transform(vxm_atlas.shape, interp_method='linear', nb_feats=1)
            tf_dataset = tf.data.Dataset.from_tensor_slices(np.transpose(affined_np, (3, 0, 1, 2)))
            del affined_np
            batch_size = 16
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
        moved_np = moved_img.numpy()
        del moved_img

        # save
        origin = (*MNI152_2mm.origin, bold_origin[3])
        spacing = (*MNI152_2mm.spacing, bold_spacing[3])
        direction = bold_direction.copy()
        direction[:3, :3] = MNI152_2mm.direction
        warped_bold_img = ants.from_numpy(moved_np, origin=origin, spacing=spacing, direction=direction)
        del moved_np
        warped_file = str(warped_file)
        if verbose:
            ants.image_write(warped_bold_img, warped_file)
        return warped_bold_img

    def _run_interface(self, runtime):
        preprocess_dir = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        subj = self.inputs.subject_id.split('-')[1]
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj_func_dir = preprocess_dir / 'func'
        subj_transform_dir = Path(preprocess_dir) / 'transform'
        subj_func_dir.mkdir(parents=True, exist_ok=True)

        norm_path = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri' / 'norm.mgz'
        subject_id = self.inputs.subject_id
        atlas_type = self.inputs.atlas_type

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

            bold_moving_file = subj_func_dir / bold_file.name.replace('.nii.gz', '_skip_reorient_faln_mc.nii.gz')

            reg_file = subj_func_dir / bold_file.name.replace('.nii.gz',
                                                              '_skip_reorient_faln_mc_bbregister.register.dat')
            bold_t1_file = subj_func_dir / bold_file.name.replace('.nii.gz',
                                                                  '_skip_reorient_faln_mc_native_T1_2mm.nii.gz')  # save reg to T1 result file
            bold_t1_out = self.native_bold_to_T1_2mm_ants(bold_moving_file, subject_id, norm_path, reg_file,
                                                          str(bold_t1_file), subj_func_dir, subj_transform_dir,
                                                          verbose=True)

            warp_file = subj_transform_dir / f'{subject_id}_vxm_deformation_field_from_norm_to_vxm_{atlas_type}.nii.gz'
            affine_file = subj_transform_dir / f'{subject_id}_ants_affine_trf_from_norm_to_vxm_{atlas_type}.mat'
            warped_file = subj_func_dir / bold_file.name.replace(
                '.nii.gz',
                f'_skip_reorient_faln_mc_native_T1_2mm_{self.inputs.atlas_type}.nii.gz')  # save reg to MNI152 result file
            self.vxm_warp_bold_2mm(bold_t1_out, affine_file, warp_file, warped_file, verbose=True)
            output_bolds.append(bold_t1_file)
            output_bolds.append(warped_file)

        self.check_output(output_bolds)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        outputs["data_path"] = self.inputs.data_path

        return outputs

    def create_sub_node(self):
        return []
