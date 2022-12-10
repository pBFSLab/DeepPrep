from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, File, TraitedSpec, Directory, Str
from interface.run import multipool_run, multipool_BidsBolds, multipool_BidsBolds_2, Pool
import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import bids
import os
import tensorflow as tf
import ants
import shutil
import voxelmorph as vxm

from app.filters.filters import bandpass_nifti


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


class BoldSkipReorientInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc="subject id", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class BoldSkipReorientOutputSpec(TraitedSpec):
    data_path = Directory(exists=True, desc="data path")
    subject_id = Str(exists=True, desc="subject id")


class BoldSkipReorient(BaseInterface):
    input_spec = BoldSkipReorientInputSpec
    output_spec = BoldSkipReorientOutputSpec

    time = 16 / 60  # 运行时间：分钟
    cpu = 2.5  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(BoldSkipReorient, self).__init__()

    def check_output(self, subj_func_dir: Path, bolds: list):
        filenames = ['_skip_reorient.nii.gz',
                     ]
        for bold in bolds:
            for filename in filenames:
                file_path = subj_func_dir / bold.name.replace('.nii.gz', filename)
                if not file_path.exists():
                    raise FileExistsError(file_path)

    def dimstr2dimno(self, dimstr):
        if 'x' in dimstr:
            return 0

        if 'y' in dimstr:
            return 1

        if 'z' in dimstr:
            return 2

    def swapdim(self, infile, a, b, c, outfile):
        '''
        infile  - str. Path to file to read and swap dimensions of.
        a       - str. New x dimension.
        b       - str. New y dimension.
        c       - str. New z dimension.
        outfile - str. Path to file to create.

        Returns None.
        '''

        # Read original file.
        img = nib.load(infile)

        # Build orientation matrix.
        ornt = np.zeros((3, 2))
        order_strs = [a, b, c]
        dim_order = list(map(self.dimstr2dimno, order_strs))
        i_dim = np.argsort(dim_order)
        for i, dim in enumerate(i_dim):
            ornt[i, 1] = -1 if '-' in order_strs[dim] else 1

        ornt[:, 0] = i_dim

        # Transform and save.
        newimg = img.as_reoriented(ornt)
        nib.save(newimg, outfile)

    def cmd(self, subj_func_dir: Path, bold: Path):

        # skip 0 frame
        nskip = 0
        if nskip > 0:
            skip_bold = subj_func_dir / bold.name.replace('.nii.gz', '_skip.nii.gz')
            sh.mri_convert('-i', bold, '--nskip', nskip, '-o', skip_bold, _out=sys.stdout)
            bold = skip_bold

        # reorient
        reorient_skip_bold = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient.nii.gz')
        self.swapdim(str(bold), 'x', '-y', 'z', str(reorient_skip_bold))

    def _run_interface(self, runtime):
        preprocess_dir = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        subj = self.inputs.subject_id.split('-')[1]
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj_func_dir = Path(preprocess_dir) / 'func'
        subj_func_dir.mkdir(parents=True, exist_ok=True)

        args = []
        bold_files = []
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            bold_file = Path(bids_bold.path)
            bold_files.append(bold_file)
            args.append([subj_func_dir, bold_file])

        pool = Pool(2)
        pool.starmap(self.cmd, args)
        pool.close()
        pool.join()

        self.check_output(subj_func_dir, bold_files)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id

        return outputs

    def create_sub_node(self):
        from interface.create_node_bold_new import create_StcMc_node
        node = create_StcMc_node(self.inputs.subject_id,
                                 self.inputs.task,
                                 self.inputs.atlas_type,
                                 self.inputs.preprocess_method)

        return node


class StcMcInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject_id', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class StcMcOutputSpec(TraitedSpec):
    data_path = Directory(exists=True, desc="data path")
    subject_id = Str(exists=True, desc="subject id")


class StcMc(BaseInterface):
    input_spec = StcMcInputSpec
    output_spec = StcMcOutputSpec

    time = 300 / 60  # 运行时间：分钟
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(StcMc, self).__init__()

    def check_output(self, subj_func_dir: Path, bolds: list):
        filenames = ['_skip_reorient_faln.nii.gz',
                     '_skip_reorient_faln_mc.nii.gz',
                     '_skip_reorient_faln_mc.mcdat',
                     ]
        for bold in bolds:
            for filename in filenames:
                file_path = subj_func_dir / bold.name.replace('.nii.gz', filename)
                if not file_path.exists():
                    raise FileExistsError(file_path)

    def cmd(self, subj_func_dir: Path, bold: Path, run: str):
        tmp_run = subj_func_dir / run
        if tmp_run.exists():
            shutil.rmtree(tmp_run)
        link_dir = tmp_run / self.inputs.subject_id / 'bold' / run
        if not link_dir.exists():
            link_dir.mkdir(parents=True, exist_ok=True)
        link_files = os.listdir(subj_func_dir)
        link_files.remove(run)
        for link_file in link_files:
            try:
                os.symlink(subj_func_dir / link_file,
                           link_dir / link_file)
            except:
                continue

        # STC
        input_fname = bold.name.replace('.nii.gz', '_skip_reorient')
        faln_fname = bold.name.replace('.nii.gz', '_skip_reorient_faln')
        mc_fname = bold.name.replace('.nii.gz', '_skip_reorient_faln_mc')
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', tmp_run,
            '-fsd', 'bold',
            '-so', 'odd',
            '-ngroups', 1,
            '-i', input_fname,
            '-o', faln_fname,
            '-nolog']
        sh.stc_sess(*shargs, _out=sys.stdout)

        # MkTemplate
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', tmp_run,
            '-fsd', 'bold',
            '-funcstem', faln_fname,
            '-nolog']
        sh.mktemplate_sess(*shargs, _out=sys.stdout)

        # Mc
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', tmp_run,
            '-per-session',
            '-fsd', 'bold',
            '-fstem', faln_fname,
            '-fmcstem', mc_fname,
            '-nolog']
        sh.mc_sess(*shargs, _out=sys.stdout)

        ori_path = subj_func_dir
        try:
            # Stc
            shutil.move(link_dir / f'{faln_fname}.nii.gz',
                        ori_path / f'{faln_fname}.nii.gz')
            shutil.move(link_dir / f'{faln_fname}.nii.gz.log',
                        ori_path / f'{faln_fname}.nii.gz.log')

            # bold template
            shutil.move(link_dir.parent / f'template.nii.gz',
                        ori_path / f'template_frame0.nii.gz')
            shutil.move(link_dir.parent / f'template.log',
                        ori_path / f'template_frame0.log')
            shutil.move(link_dir / f'template.nii.gz',
                        ori_path / f'template_frame_mid.nii.gz')
            shutil.move(link_dir / f'template.log',
                        ori_path / f'template_frame_mid.log')

            # Mc
            shutil.move(link_dir / f'{mc_fname}.nii.gz',
                        ori_path / f'{mc_fname}.nii.gz')
            shutil.move(link_dir / f'{mc_fname}.mat.aff12.1D',
                        ori_path / f'{mc_fname}.mat.aff12.1D')
            shutil.move(link_dir / f'{mc_fname}.nii.gz.mclog',
                        ori_path / f'{mc_fname}.nii.gz.mclog')
            shutil.move(link_dir / f'{mc_fname}.mcdat',
                        ori_path / f'{mc_fname}.mcdat')
            shutil.move(link_dir / 'mcextreg', ori_path / f'{mc_fname}.mcextreg')
            shutil.move(link_dir / 'mcdat2extreg.log', ori_path / f'{mc_fname}.mcdat2extreg.log')
        except:
            pass
        shutil.rmtree(ori_path / run)

    def _run_interface(self, runtime):
        preprocess_dir = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        subj = self.inputs.subject_id.split('-')[1]
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj_func_dir = Path(preprocess_dir) / 'func'
        subj_func_dir.mkdir(parents=True, exist_ok=True)

        args = []
        bold_files = []
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            run = f'{idx + 1:03d}'
            bold_file = Path(bids_bold.path)
            bold_files.append(bold_file)
            args.append([subj_func_dir, bold_file, run])

        pool = Pool(2)
        pool.starmap(self.cmd, args)
        pool.close()
        pool.join()

        self.check_output(subj_func_dir, bold_files)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self):
        if self.bold_only == 'True':
            from interface.create_node_bold_new import create_Register_node
            return create_Register_node(self.inputs.subject_id,
                                        self.inputs.task,
                                        self.inputs.atlas_type,
                                        self.inputs.preprocess_method)
        return []


class RegisterInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject_id', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class RegisterOutputSpec(TraitedSpec):
    data_path = Directory(exists=True, desc="data path")
    subject_id = Str(exists=True, desc="subject id")


class Register(BaseInterface):
    input_spec = RegisterInputSpec
    output_spec = RegisterOutputSpec

    time = 141 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(Register, self).__init__()

    def check_output(self, subj_func_dir: Path, bolds: list):
        filenames = ['_skip_reorient_faln_mc_bbregister.register.dat',
                     ]
        for bold in bolds:
            for filename in filenames:
                file_path = subj_func_dir / bold.name.replace('.nii.gz', filename)
                if not file_path.exists():
                    raise FileExistsError

    def cmd(self, subj_func_dir: Path, bold: Path):
        mov = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc.nii.gz')
        reg = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc_bbregister.register.dat')
        print(os.environ["SUBJECTS_DIR"])
        shargs = [
            '--bold',
            '--s', self.inputs.subject_id,
            '--mov', mov,
            '--reg', reg]
        sh.bbregister(*shargs, _out=sys.stdout)

    def _run_interface(self, runtime):
        preprocess_dir = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        subj = self.inputs.subject_id.split('-')[1]
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj_func_dir = Path(preprocess_dir) / 'func'
        subj_func_dir.mkdir(parents=True, exist_ok=True)

        args = []
        bold_files = []
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            bold_file = Path(bids_bold.path)
            bold_files.append(bold_file)
            args.append([subj_func_dir, bold_file])

        pool = Pool(2)
        pool.starmap(self.cmd, args)
        pool.close()
        pool.join()

        self.check_output(subj_func_dir, bold_files)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self):
        from interface.create_node_bold_new import create_Mkbrainmask_node
        node = create_Mkbrainmask_node(self.inputs.subject_id,
                                       self.inputs.task,
                                       self.inputs.atlas_type,
                                       self.inputs.preprocess_method)

        return node


class MkBrainmaskInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject_id', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class MkBrainmaskOutputSpec(TraitedSpec):
    data_path = Directory(exists=True, desc="data path")
    subject_id = Str(exists=True, desc="subject id")


class MkBrainmask(BaseInterface):
    input_spec = MkBrainmaskInputSpec
    output_spec = MkBrainmaskOutputSpec

    time = 5 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(MkBrainmask, self).__init__()

    def check_output(self, subj_func_dir: Path, bolds: list):
        filenames = ['.func.aseg.nii.gz',
                     '.func.wm.nii.gz',
                     '.func.ventricles.nii.gz',
                     '.brainmask.nii.gz',
                     '.brainmask.bin.nii.gz',
                     ]
        for bold in bolds:
            for filename in filenames:
                file_path = subj_func_dir / bold.name.replace('.nii.gz', filename)
                if not file_path.exists():
                    raise FileExistsError(file_path)

    def cmd(self, subj_func_dir: Path, bold: Path):
        # project aparc+aseg to mc
        seg = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri/aparc+aseg.mgz'  # Recon
        mov = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc.nii.gz')
        reg = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc_bbregister.register.dat')
        func = subj_func_dir / bold.name.replace('.nii.gz', '.func.aseg.nii.gz')
        wm = subj_func_dir / bold.name.replace('.nii.gz', '.func.wm.nii.gz')
        vent = subj_func_dir / bold.name.replace('.nii.gz', '.func.ventricles.nii.gz')
        # project bold to brainmask.mgz
        targ = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri/brainmask.mgz'  # Recon
        mask = subj_func_dir / bold.name.replace('.nii.gz', '.brainmask.nii.gz')
        binmask = subj_func_dir / bold.name.replace('.nii.gz', '.brainmask.bin.nii.gz')

        shargs = [
            '--seg', seg,
            '--temp', mov,
            '--reg', reg,
            '--o', func]
        sh.mri_label2vol(*shargs, _out=sys.stdout)

        shargs = [
            '--i', func,
            '--wm',
            '--erode', 1,
            '--o', wm]
        sh.mri_binarize(*shargs, _out=sys.stdout)

        shargs = [
            '--i', func,
            '--ventricles',
            '--o', vent]
        sh.mri_binarize(*shargs, _out=sys.stdout)

        shargs = [
            '--reg', reg,
            '--targ', targ,
            '--mov', mov,
            '--inv',
            '--o', mask]
        sh.mri_vol2vol(*shargs, _out=sys.stdout)

        shargs = [
            '--i', mask,
            '--o', binmask,
            '--min', 0.0001]
        sh.mri_binarize(*shargs, _out=sys.stdout)

    def _run_interface(self, runtime):
        preprocess_dir = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        subj = self.inputs.subject_id.split('-')[1]
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj_func_dir = Path(preprocess_dir) / 'func'
        subj_func_dir.mkdir(parents=True, exist_ok=True)

        args = []
        bold_files = []
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            bold_file = Path(bids_bold.path)
            bold_files.append(bold_file)
            args.append([subj_func_dir, bold_file])

        pool = Pool(2)
        pool.starmap(self.cmd, args)
        pool.close()
        pool.join()

        self.check_output(subj_func_dir, bold_files)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self):
        if self.bold_only == 'True':
            from interface.create_node_bold_new import create_VxmRegistraion_node
            node = create_VxmRegistraion_node(self.inputs.subject_id,
                                                self.inputs.task,
                                                self.inputs.atlas_type,
                                                self.inputs.preprocess_method)
        else:
            from interface.create_node_bold_new import create_VxmRegNormMNI152_node
            node = create_VxmRegNormMNI152_node(self.inputs.subject_id,
                                                self.inputs.task,
                                                self.inputs.atlas_type,
                                                self.inputs.preprocess_method)

        return node


class RestGaussInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc="subject id", mandatory=True)
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    data_path = Directory(exists=True, desc='data path', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class RestGaussOutputSpec(TraitedSpec):
    subject_id = Str(exists=True, desc="subject id")


class RestGauss(BaseInterface):
    input_spec = RestGaussInputSpec
    output_spec = RestGaussOutputSpec

    time = 11 / 60  # 运行时间：分钟
    cpu = 0  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(RestGauss, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task

        for run in runs:
            RestGauss_output_files = [f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(RestGauss_output_files) <= set(output_list)
            if not check_result:
                return FileExistsError

    def cmd(self, run):
        from app.filters.filters import gauss_nifti

        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        mc = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
        fcmri_dir = Path(preprocess_dir) / self.inputs.subject_id / 'fcmri'
        Path(fcmri_dir).mkdir(parents=True, exist_ok=True)
        gauss_nifti(str(mc), 1000000000)

    def _run_interface(self, runtime):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        runs = sorted(
            [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        multipool_run(self.cmd, runs, Multi_Num=8)

        self.check_output(runs)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self):
        from interface.create_node_bold_new import create_RestBandpass_node
        node = create_RestBandpass_node(self.inputs.subject_id,
                                        self.inputs.task,
                                        self.inputs.atlas_type,
                                        self.inputs.preprocess_method)

        return node


class RestBandpassInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    data_path = Directory(exists=True, desc='data path', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class RestBandpassOutputSpec(TraitedSpec):
    subject_id = Str(exists=True, desc='subject')
    data_path = Directory(exists=True, desc='data path')


class RestBandpass(BaseInterface):
    input_spec = RestBandpassInputSpec
    output_spec = RestBandpassOutputSpec

    time = 120 / 60  # 运行时间：分钟
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(RestBandpass, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task

        for run in runs:
            RestBandpass_output_files = [f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(RestBandpass_output_files) <= set(output_list)
            if not check_result:
                return FileExistsError

    def cmd(self, idx, bids_entities, bids_path):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        entities = dict(bids_entities)
        if 'RepetitionTime' in entities:
            TR = entities['RepetitionTime']
        else:
            bold = ants.image_read(bids_path)
            TR = bold.spacing[3]
        run = f'{idx + 1:03}'
        gauss_path = preprocess_dir / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz'
        bandpass_nifti(str(gauss_path), TR)

    def _run_interface(self, runtime):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj = self.inputs.subject_id.split('-')[1:]
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        all_idx = []
        all_bids_entities = []
        all_bids_path = []
        for idx, bids_bold in enumerate(bids_bolds):
            all_idx.append(idx)
            all_bids_entities.append(bids_bold.entities)
            all_bids_path.append(bids_bold.path)
        multipool_BidsBolds(self.cmd, all_idx, all_bids_entities, all_bids_path, Multi_Num=8)
        runs = sorted(
            [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])

        self.check_output(runs)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        outputs["data_path"] = self.inputs.data_path

        return outputs

    def create_sub_node(self):
        from interface.create_node_bold_new import create_RestRegression_node
        node = create_RestRegression_node(self.inputs.subject_id,
                                          self.inputs.task,
                                          self.inputs.atlas_type,
                                          self.inputs.preprocess_method)

        return node


class RestRegressionInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    data_path = Directory(exists=True, desc='data_path', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class RestRegressionOutputSpec(TraitedSpec):
    subject_id = Str(exists=True, desc='subject')
    data_path = Directory(exists=True, desc='data_path')


class RestRegression(BaseInterface):
    input_spec = RestRegressionInputSpec
    output_spec = RestRegressionOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB
    def __init__(self):
        super(RestRegression, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task

        for run in runs:
            RestRegression_bold_output_files = [
                f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_snr.nii.gz',
                f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_sd1.nii.gz',
                f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz']
            bold_output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_bold_result = set(RestRegression_bold_output_files) <= set(bold_output_list)

            RestRegression_fcmri_output_files = [f"{sub}_bld{run}_mov_regressor.dat",
                                                 f"{sub}_bld{run}_pca_regressor_dt.dat",
                                                 f"{sub}_bld{run}_regressors.dat",
                                                 f"{sub}_bld{run}_ventricles_regressor_dt.dat",
                                                 f"{sub}_bld{run}_vent_wm_dt.dat",
                                                 f"{sub}_bld{run}_WB_regressor_dt.dat",
                                                 f"{sub}_bld{run}_wm_regressor_dt.dat"]
            fcmri_output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'fcmri')
            check_fcmri_result = set(RestRegression_fcmri_output_files) <= set(fcmri_output_list)

            RestRegression_movement_output_files = [f"{sub}_bld{run}_rest_reorient_skip_faln_mc.dat",
                                                    f"{sub}_bld{run}_rest_reorient_skip_faln_mc.ddat",
                                                    f"{sub}_bld{run}_rest_reorient_skip_faln_mc.par",
                                                    f"{sub}_bld{run}_rest_reorient_skip_faln_mc.rdat",
                                                    f"{sub}_bld{run}_rest_reorient_skip_faln_mc.rddat"]
            movement_output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'movement')
            check_movement_result = set(RestRegression_movement_output_files) <= set(movement_output_list)

        if not check_bold_result and not check_fcmri_result and not check_movement_result:
            return FileExistsError

    def setenv_smooth_downsampling(self):
        subjects_dir = Path(self.inputs.subjects_dir)
        fsaverage6_dir = subjects_dir / 'fsaverage6'
        if not fsaverage6_dir.exists():
            src_fsaverage6_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage6'
            try:
                os.symlink(src_fsaverage6_dir, fsaverage6_dir)
            except FileExistsError:
                pass

        fsaverage5_dir = subjects_dir / 'fsaverage5'
        if not fsaverage5_dir.exists():
            src_fsaverage5_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage5'
            try:
                os.symlink(src_fsaverage5_dir, fsaverage5_dir)
            except FileExistsError:
                pass

        fsaverage4_dir = subjects_dir / 'fsaverage4'
        if not fsaverage4_dir.exists():
            src_fsaverage4_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage4'
            try:
                os.symlink(src_fsaverage4_dir, fsaverage4_dir)
            except FileExistsError:
                pass

    # smooth_downsampling
    # def cmd(self, hemi, subj_surf_path, dst_resid_file, dst_reg_file):
    #     from deepprep_pipeline.app.surface_projection import surface_projection as sp
    #     fs6_path = sp.indi_to_fs6(subj_surf_path, f'{self.inputs.subject_id}', dst_resid_file, dst_reg_file,
    #                               hemi)
    #     sm6_path = sp.smooth_fs6(fs6_path, hemi)
    #     sp.downsample_fs6_to_fs4(sm6_path, hemi)
    def _run_interface(self, runtime):
        from app.regressors.regressors import compile_regressors, regression
        from app.surface_projection import surface_projection as sp

        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'

        deepprep_subj_path = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        fcmri_dir = preprocess_dir / self.inputs.subject_id / 'fcmri'
        bold_dir = preprocess_dir / self.inputs.subject_id / 'bold'
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj = self.inputs.subject_id.split('-')[1]
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            run = f'{idx + 1:03}'
            bpss_path = f'{preprocess_dir}/{self.inputs.subject_id}/bold/{run}/{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz'

            all_regressors = compile_regressors(Path(preprocess_dir), Path(bold_dir), run, self.inputs.subject_id,
                                                Path(fcmri_dir), bpss_path)
            regression(bpss_path, all_regressors)

        self.setenv_smooth_downsampling()
        deepprep_subj_path = Path(deepprep_subj_path)

        subj_bold_dir = Path(preprocess_dir) / f'{self.inputs.subject_id}' / 'bold'
        for idx, bids_bold in enumerate(bids_bolds):
            run = f"{idx + 1:03}"
            entities = dict(bids_bold.entities)
            # subj = entities['subject']
            file_prefix = Path(bids_bold.path).name.replace('.nii.gz', '')
            if 'session' in entities:
                subj_func_path = deepprep_subj_path / f"ses-{entities['session']}" / 'func'
                subj_surf_path = deepprep_subj_path / f"ses-{entities['session']}" / 'surf'
            else:
                subj_func_path = deepprep_subj_path / 'func'
                subj_surf_path = deepprep_subj_path / 'surf'
            subj_surf_path.mkdir(exist_ok=True)
            src_resid_file = subj_bold_dir / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
            dst_resid_file = subj_func_path / f'{file_prefix}_resid.nii.gz'
            shutil.copy(src_resid_file, dst_resid_file)

            dst_reg_file = subj_func_path / f'{file_prefix}_bbregister.register.dat'
            # hemi = ['lh','rh']
            # multiregressionpool(self.cmd, hemi, subj_surf_path, dst_resid_file, dst_reg_file, Multi_Num=2)
            for hemi in ['lh', 'rh']:
                fs6_path = sp.indi_to_fs6(subj_surf_path, f'{self.inputs.subject_id}', dst_resid_file, dst_reg_file,
                                          hemi)
                sm6_path = sp.smooth_fs6(fs6_path, hemi)
                sp.downsample_fs6_to_fs4(sm6_path, hemi)

            runs = sorted(
                [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
            self.check_output(runs)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        outputs["data_path"] = self.inputs.data_path

        return outputs

    def create_sub_node(self):
        from interface.create_node_bold_new import create_VxmRegNormMNI152_node
        node = create_VxmRegNormMNI152_node(self.inputs.subject_id,
                                            self.inputs.task,
                                            self.inputs.atlas_type,
                                            self.inputs.preprocess_method)

        return node


class SmoothInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    data_path = Directory(exists=True, desc='data_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    MNI152_T1_2mm_brain_mask = File(exists=True, desc='MNI152 brain mask path', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class SmoothOutputSpec(TraitedSpec):
    subject_id = Str(exists=True, desc='subject')
    data_path = Directory(exists=True, desc='data_path')


class Smooth(BaseInterface):
    input_spec = SmoothInputSpec
    output_spec = SmoothOutputSpec

    time = 68 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(Smooth, self).__init__()

    def check_output(self, subj_func_path, file_prefix):
        sub = self.inputs.subject_id

        Smooth_output_files = [f'{file_prefix}_resid_MNI2mm_sm6.nii.gz']  # TODO MNI2mm 要不要优化
        output_list = os.listdir(subj_func_path)
        check_result = set(Smooth_output_files) <= set(output_list)
        if not check_result:
            return FileExistsError

    def save_bold(self, warped_img, temp_file, bold_file, save_file):
        ants.image_write(warped_img, str(temp_file))
        bold_info = nib.load(bold_file)
        affine_info = nib.load(temp_file)
        bold2 = nib.Nifti1Image(warped_img.numpy(), affine=affine_info.affine, header=bold_info.header)
        del bold_info
        del affine_info
        os.remove(temp_file)
        nib.save(bold2, save_file)

    def bold_smooth_6_ants(self, t12mm: str, t12mm_sm6_file: Path,
                           temp_file: Path, bold_file: Path, verbose=False):

        # mask file
        # MNI152_T1_2mm_brain_mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
        brain_mask = ants.image_read(self.inputs.MNI152_T1_2mm_brain_mask)

        if isinstance(t12mm, str):
            bold_img = ants.image_read(t12mm)
        else:
            bold_img = t12mm

        bold_origin = bold_img.origin
        bold_spacing = bold_img.spacing
        bold_direction = bold_img.direction.copy()

        # smooth
        smoothed_img = ants.from_numpy(bold_img.numpy(), bold_origin[:3], bold_spacing[:3],
                                       bold_direction[:3, :3].copy(), has_components=True)
        # mask
        smoothed_np = ants.smooth_image(smoothed_img, sigma=6, FWHM=True).numpy()
        del smoothed_img
        mask_np = brain_mask.numpy()
        masked_np = np.zeros(smoothed_np.shape, dtype=np.float32)
        idx = mask_np == 1
        masked_np[idx, :] = smoothed_np[idx, :]
        del smoothed_np
        masked_img = ants.from_numpy(masked_np, bold_origin, bold_spacing, bold_direction)
        del masked_np
        if verbose:
            # save
            self.save_bold(masked_img, temp_file, bold_file, t12mm_sm6_file)
            # ants.image_write(masked_img, str(t12mm_sm6_file))
        return masked_img

    def cmd(self, bids_entities, bids_path):
        entities = dict(bids_entities)
        file_prefix = Path(bids_path).name.replace('.nii.gz', '')
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        derivative_deepprep_path = Path(self.inputs.derivative_deepprep_path)
        deepprep_subj_path = derivative_deepprep_path / self.inputs.subject_id
        if 'session' in entities:
            ses = entities['session']
            subj_func_path = Path(deepprep_subj_path) / f'ses-{ses}' / 'func'
        else:
            subj_func_path = Path(deepprep_subj_path) / 'func'
        if self.inputs.preprocess_method == 'rest':
            bold_file = subj_func_path / f'{file_prefix}_resid.nii.gz'
            save_file = subj_func_path / f'{file_prefix}_resid_MIN2mm_sm6.nii.gz'
        else:
            bold_file = subj_func_path / f'{file_prefix}_mc.nii.gz'
            save_file = subj_func_path / f'{file_prefix}_mc_MIN2mm.nii.gz'
        if self.inputs.preprocess_method == 'rest':
            temp_file = Path(preprocess_dir) / f'{file_prefix}_MNI2mm_sm6_temp.nii.gz'
            warped_file = subj_func_path / f'{self.inputs.subject_id}_MNI2mm.nii.gz'
            warped_img = ants.image_read(str(warped_file))
            self.bold_smooth_6_ants(warped_img, save_file, temp_file, bold_file, verbose=True)
        else:
            temp_file = Path(preprocess_dir) / f'{file_prefix}_MNI2mm_temp.nii.gz'
            warped_file = subj_func_path / f'{self.inputs.subject_id}_MNI2mm.nii.gz'
            warped_img = ants.image_read(str(warped_file))
            self.save_bold(warped_img, temp_file, bold_file, save_file)

    def _run_interface(self, runtime):
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj = self.inputs.subject_id.split('-')[1]
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        bids_entities = []
        bids_path = []
        for bids_bold in bids_bolds:
            bids_entities.append(bids_bold.entities)
            bids_path.append(bids_bold.path)
        multipool_BidsBolds_2(self.cmd, bids_entities, bids_path, Multi_Num=8)

        derivative_deepprep_path = Path(self.inputs.derivative_deepprep_path)
        deepprep_subj_path = derivative_deepprep_path / self.inputs.subject_id
        for i in range(len(bids_entities)):
            entities = dict(bids_entities[i])
            file_prefix = Path(bids_path[i]).name.replace('.nii.gz', '')
            if 'session' in entities:
                ses = entities['session']
                subj_func_path = Path(deepprep_subj_path) / f'ses-{ses}' / 'func'
            else:
                subj_func_path = Path(deepprep_subj_path) / 'func'
            self.check_output(subj_func_path, file_prefix)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        outputs["data_path"] = self.inputs.data_path

        return outputs
