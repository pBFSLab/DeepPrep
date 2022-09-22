from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, File, TraitedSpec, Directory, Str
import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import ants
import bids



class BoldSkipReorientInputSpec(BaseInterfaceInputSpec):
    preprocess_dir = Directory(exists=True, desc="preprocess dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    bold = File(exists=True, desc="'bold'/run/f'{subj}_bld{run}_rest.nii.gz'", mandatory=True)

    skip_bold = File(exists=False, desc="'bold'/run/f'{subj}_bld{run}_rest_skip.nii.gz'", mandatory=True)
    reorient_skip_bold = File(exists=False, desc="'bold'/run/f'{subj}_bld{run}_rest_reorient_skip.nii.gz'",
                              mandatory=True)


class BoldSkipReorientOutputSpec(TraitedSpec):
    skip_bold = File(exists=True, desc="'bold'/run/f'{subj}_bld{run}_rest_skip.nii.gz'")
    reorient_skip_bold = File(exists=True, desc="'bold'/run/f'{subj}_bld{run}_rest_reorient_skip.nii.gz'")


class BoldSkipReorient(BaseInterface):
    input_spec = BoldSkipReorientInputSpec
    output_spec = BoldSkipReorientOutputSpec

    time = 4.3 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 2.5  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

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

    def _run_interface(self, runtime):
        # skip 0 frame
        sh.mri_convert('-i', self.inputs.bold, '-o', self.inputs.skip_bold, _out=sys.stdout)

        # reorient
        self.swapdim(str(self.inputs.skip_bold), 'x', '-y', 'z', str(self.inputs.reorient_skip_bold))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["skip_bold"] = self.inputs.skip_bold
        outputs["reorient_skip_bold"] = self.inputs.reorient_skip_bold

        return outputs


class MotionCorrectionInputSpec(BaseInterfaceInputSpec):
    preprocess_dir = Directory(exists=True, desc="preprocess dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    skip_faln = File(exists=True, desc="'bold'/run/f'{subj}_bld_rest_reorient_skip_faln.nii.gz'", mandatory=True)

    skip_faln_mc = File(exists=False, desc="'bold'/run/f'{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'", mandatory=True)


class MotionCorrectionOutputSpec(TraitedSpec):
    skip_faln_mc = File(exists=True, desc="'bold'/run/f'{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'")


class MotionCorrection(BaseInterface):
    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    time = 400 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 0  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', self.inputs.preprocess_dir,
            '-per-session',
            '-fsd', 'bold',
            '-fstem', f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln',
            '-fmcstem', f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc',
            '-nolog']
        sh.mc_sess(*shargs, _out=sys.stdout)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["skip_faln_mc"] = self.inputs.skip_faln_mc

        return outputs


class StcInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)
    skip = File(exists=True, desc='{subj}_bld_rest_reorient_skip', mandatory=True)
    faln = File(exists=False, desc='{subj}_bld_rest_reorient_skip_faln', mandatory=True)


class StcOutputSpec(TraitedSpec):
    faln = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln')


class Stc(BaseInterface):
    input_spec = StcInputSpec
    output_spec = StcOutputSpec

    time = 214 / 60  # 运行时间：分钟
    cpu = 6  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        input_fname = f'{self.inputs.subject_id}_bld_rest_reorient_skip'
        output_fname = f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln'
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', self.inputs.preprocess_dir,
            '-fsd', 'bold',
            '-so', 'odd',
            '-ngroups', 1,
            '-i', input_fname,
            '-o', output_fname,
            '-nolog']
        sh.stc_sess(*shargs, _out=sys.stdout)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["faln"] = self.inputs.faln

        return outputs


class RegisterInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)
    mov = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc.nii.gz', mandatory=True)
    reg = File(exists=False, desc='{subj}_bld_rest_reorient_skip_faln_mc.register.dat', mandatory=True)


class RegisterOutputSpec(TraitedSpec):
    reg = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc.register.dat')


class Register(BaseInterface):
    input_spec = RegisterInputSpec
    output_spec = RegisterOutputSpec

    time = 96 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        shargs = [
            '--bold',
            '--s', self.inputs.subject_id,
            '--mov', self.inputs.mov,
            '--reg', self.inputs.reg]
        sh.bbregister(*shargs, _out=sys.stdout)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["reg"] = self.inputs.reg

        return outputs


class MkBrainmaskInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)
    seg = File(exists=True, desc='mri/aparc+aseg.mgz', mandatory=True)
    targ = File(exists=True, desc='mri/brainmask.mgz', mandatory=True)
    mov = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc.nii.gz', mandatory=True)
    reg = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc.register.dat', mandatory=True)

    func = File(exists=False, desc='{subj}.func.aseg.nii', mandatory=True)
    wm = File(exists=False, desc='{subj}.func.wm.nii.gz', mandatory=True)
    vent = File(exists=False, desc='{subj}.func.ventricles.nii.gz', mandatory=True)
    mask = File(exists=False, desc='{subj}.brainmask.nii.gz', mandatory=True)
    binmask = File(exists=False, desc='{subj}.brainmask.bin.nii.gz', mandatory=True)


class MkBrainmaskOutputSpec(TraitedSpec):
    func = File(exists=True, desc='{subj}.func.aseg.nii')
    wm = File(exists=True, desc='{subj}.func.wm.nii.gz')
    vent = File(exists=True, desc='{subj}.func.ventricles.nii.gz')
    mask = File(exists=True, desc='{subj}.brainmask.nii.gz')
    binmask = File(exists=True, desc='{subj}.brainmask.bin.nii.gz')


class MkBrainmask(BaseInterface):
    input_spec = MkBrainmaskInputSpec
    output_spec = MkBrainmaskOutputSpec

    time = 4 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        shargs = [
            '--seg', self.inputs.seg,
            '--temp', self.inputs.mov,
            '--reg', self.inputs.reg,
            '--o', self.inputs.func]
        sh.mri_label2vol(*shargs, _out=sys.stdout)

        shargs = [
            '--i', self.inputs.func,
            '--wm',
            '--erode', 1,
            '--o', self.inputs.wm]
        sh.mri_binarize(*shargs, _out=sys.stdout)

        shargs = [
            '--i', self.inputs.func,
            '--ventricles',
            '--o', self.inputs.vent]
        sh.mri_binarize(*shargs, _out=sys.stdout)

        shargs = [
            '--reg', self.inputs.reg,
            '--targ', self.inputs.targ,
            '--mov', self.inputs.mov,
            '--inv',
            '--o', self.inputs.mask]
        sh.mri_vol2vol(*shargs, _out=sys.stdout)

        shargs = [
            '--i', self.inputs.mask,
            '--o', self.inputs.binmask,
            '--min', 0.0001]
        sh.mri_binarize(*shargs, _out=sys.stdout)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["func"] = self.inputs.func
        outputs["mask"] = self.inputs.mask
        outputs["wm"] = self.inputs.wm
        outputs["vent"] = self.inputs.vent
        outputs["binmask"] = self.inputs.binmask

        return outputs


class VxmRegistraionInputSpec(BaseInterfaceInputSpec):
    atlas_type = Str(desc="atlas type", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    norm = File(exists=True, desc="mri/norm.mgz", mandatory=True)
    model_file = File(exists=True, desc="atlas_type/model.h5", mandatory=True)
    atlas = File(exists=True, desc="model_path/{atlas_type}_brain.nii.gz", mandatory=True)
    vxm_atlas = File(exists=True, desc="model_path/{atlas_type}_brain.nii.gz", mandatory=True)
    vxm_atlas_npz = File(exists=True, desc="model_path/{atlas_type}_brain_vxm.npz", mandatory=True)
    vxm2atlas_trf = File(exists=True, desc="model_path/{atlas_type}_vxm2atlas.mat", mandatory=True)

    vxm_warp = File(exists=False, desc="tmpdir/warp.nii.gz", mandatory=True)
    vxm_warped = File(exists=False, desc="tmpdir/warped.nii.gz", mandatory=True)
    trf = File(exists=False, desc="tmpdir/sub-{subj}_affine.mat", mandatory=True)
    warp = File(exists=False, desc="tmpdir/sub-{subj}_warp.nii.gz", mandatory=True)
    warped = File(exists=False, desc="tmpdir/sub-{subj}_warped.nii.gz", mandatory=True)
    npz = File(exists=False, desc="tmpdir/vxminput.npz", mandatory=True)


class VxmRegistraionOutputSpec(TraitedSpec):
    vxm_warp = File(exists=True, desc="tmpdir/warp.nii.gz")
    vxm_warped = File(exists=True, desc="tmpdir/warped.nii.gz")
    trf = File(exists=True, desc="tmpdir/sub-{subj}_affine.mat")
    warp = File(exists=True, desc="tmpdir/sub-{subj}_warp.nii.gz")
    warped = File(exists=True, desc="tmpdir/sub-{subj}_warped.nii.gz")
    npz = File(exists=True, desc="tmpdir/vxminput.npz")


class VxmRegistraion(BaseInterface):
    input_spec = VxmRegistraionInputSpec
    output_spec = VxmRegistraionOutputSpec

    time = 15 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 14  # 最大cpu占用：个
    gpu = 2703  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        import tensorflow as tf
        import ants
        import shutil
        import voxelmorph as vxm

        model_path = Path(__file__).parent.parent / 'model' / 'voxelmorph' / self.inputs.atlas_type
        # atlas
        if self.inputs.atlas_type == 'MNI152_T1_1mm':
            atlas_path = '../../data/atlas/MNI152_T1_1mm_brain.nii.gz'
            vxm_atlas_path = '../../data/atlas/MNI152_T1_1mm_brain_vxm.nii.gz'
            vxm_atlas_npz_path = '../../data/atlas/MNI152_T1_1mm_brain_vxm.npz'
            vxm2atlas_trf = '../../data/atlas/MNI152_T1_1mm_vxm2atlas.mat'
        elif self.inputs.atlas_type == 'MNI152_T1_2mm':
            atlas_path = model_path / 'MNI152_T1_2mm_brain.nii.gz'
            vxm_atlas_path = model_path / 'MNI152_T1_2mm_brain_vxm.nii.gz'
            vxm_atlas_npz_path = model_path / 'MNI152_T1_2mm_brain_vxm.npz'
            vxm2atlas_trf = model_path / 'MNI152_T1_2mm_vxm2atlas.mat'
        elif self.inputs.atlas_type == 'FS_T1_2mm':
            atlas_path = '../../data/atlas/FS_T1_2mm_brain.nii.gz'
            vxm_atlas_path = '../../data/atlas/FS_T1_2mm_brain_vxm.nii.gz'
            vxm_atlas_npz_path = '../../data/atlas/FS_T1_2mm_brain_vxm.npz'
            vxm2atlas_trf = '../../data/atlas/FS_T1_2mm_vxm2atlas.mat'
        else:
            raise Exception('atlas type error')

        norm = ants.image_read(str(self.inputs.norm))
        vxm_atlas = ants.image_read(str(vxm_atlas_path))
        tx = ants.registration(fixed=vxm_atlas, moving=norm, type_of_transform='Affine')
        trf = ants.read_transform(tx['fwdtransforms'][0])
        ants.write_transform(trf, str(self.inputs.trf))
        affined = tx['warpedmovout']
        vol = affined.numpy() / 255.0
        np.savez_compressed(self.inputs.npz, vol=vol)

        # voxelmorph
        # tensorflow device handling
        gpuid = '0'
        device, nb_devices = vxm.tf.utils.setup_device(gpuid)

        # load moving and fixed images
        add_feat_axis = True
        moving = vxm.py.utils.load_volfile(str(self.inputs.npz), add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed, fixed_affine = vxm.py.utils.load_volfile(str(vxm_atlas_npz_path), add_batch_axis=True,
                                                        add_feat_axis=add_feat_axis,
                                                        ret_affine=True)
        vxm_atlas_nib = nib.load(str(vxm_atlas_path))
        fixed_affine = vxm_atlas_nib.affine.copy()
        inshape = moving.shape[1:-1]
        nb_feats = moving.shape[-1]

        with tf.device(device):
            # load model and predict
            warp = vxm.networks.VxmDense.load(self.inputs.model_file).register(moving, fixed)
            # warp = vxm.networks.VxmDenseSemiSupervisedSeg.load(args.model).register(moving, fixed)
            moving = affined.numpy()[np.newaxis, ..., np.newaxis]
            moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

        # save warp
        vxm.py.utils.save_volfile(warp.squeeze(), str(self.inputs.vxm_warp), fixed_affine)
        shutil.copy(self.inputs.vxm_warp, self.inputs.warp)

        # save moved image
        vxm.py.utils.save_volfile(moved.squeeze(), str(self.inputs.vxm_warped), fixed_affine)

        # affine to atlas
        atlas = ants.image_read(str(atlas_path))
        vxm_warped = ants.image_read(str(self.inputs.vxm_warped))
        warped = ants.apply_transforms(fixed=atlas, moving=vxm_warped, transformlist=[str(vxm2atlas_trf)])
        Path(self.inputs.warped).parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(warped, str(self.inputs.warped))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["vxm_warp"] = self.inputs.vxm_warp
        outputs["vxm_warped"] = self.inputs.vxm_warped
        outputs["trf"] = self.inputs.trf
        outputs["warp"] = self.inputs.warp
        outputs["warped"] = self.inputs.warped
        outputs["npz"] = self.inputs.npz

        return outputs


class RestGaussInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, mandatory=True, desc='subject')
    preprocess_dir = Directory(exists=True, mandatory=True, desc='preprocess_dir')
    mc = File(exists=True, mandatory=True, desc='{subj}_bld_rest_reorient_skip_faln_mc.nii.gz')
    gauss = File(mandatory=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz')


class RestGaussOutputSpec(TraitedSpec):
    gauss = File(exists=True, mandatory=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz')


class RestGauss(BaseInterface):
    input_spec = RestGaussInputSpec
    output_spec = RestGaussOutputSpec

    time = 3.5 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        from app.filters.filters import gauss_nifti
        fcmri_dir = f'{self.inputs.preprocess_dir} / {self.inputs.subject_id} / fcmri'
        Path(fcmri_dir).mkdir(parents=True, exist_ok=True)
        self.inputs.gauss = gauss_nifti(str(self.inputs.mc), 1000000000)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["gauss"] = self.inputs.gauss
        return outputs


class RestBandpassInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    subj = Str(exists=True, desc='subj', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)
    data_path = Directory(exists=True, desc='data path', mandatory=True)
    gauss = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz', mandatory=True)
    bpss = File(exists=False, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz', mandatory=True)


class RestBandpassOutputSpec(TraitedSpec):
    bpss = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz')


class RestBandpass(BaseInterface):
    input_spec = RestBandpassInputSpec
    output_spec = RestBandpassOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        from app.filters.filters import bandpass_nifti
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)

        if self.inputs.task is None:
            bids_bolds = layout.get(subject=self.inputs.subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=self.inputs.subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            entities = dict(bids_bold.entities)
            print(entities)
            if 'RepetitionTime' in entities:
                TR = entities['RepetitionTime']
            else:
                bold = ants.image_read(bids_bold.path)
                TR = bold.spacing[3]
            run = f'{idx + 1:03}'
            gauss_path = f'{self.inputs.preprocess_dir}/{self.inputs.subject_id}/bold/{run}/{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz'
            bandpass_nifti(gauss_path, TR)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["bpss"] = self.inputs.bpss

        return outputs


class RestRegressionInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    subj = Str(exists=True, desc='subj', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)
    fcmri_dir = Directory(exists=True, desc='fcmri_dir', mandatory=True)
    bold_dir = Directory(exists=True, desc='bold_dir', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    data_path = Directory(exists=True, desc='data_path', mandatory=True)
    bpss = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz', mandatory=True)


    resid = File(exists=False, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz', mandatory=True)
    resid_snr = File(exists=False, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_snr.nii.gz', mandatory=True)
    resid_sd1 = File(exists=False, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_sd1.nii.gz', mandatory=True)


class RestRegressionOutputSpec(TraitedSpec):
    resid = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz')
    resid_snr = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_snr.nii.gz')
    resid_sd1 = File(exists=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_sd1.nii.gz')


class RestRegression(BaseInterface):
    input_spec = RestRegressionInputSpec
    output_spec = RestRegressionOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        from app.regressors.regressors import compile_regressors, regression
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)

        if self.inputs.task is None:
            bids_bolds = layout.get(subject=self.inputs.subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=self.inputs.subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            run = f'{idx + 1:03}'
            bpss_path = f'{self.inputs.preprocess_dir}/{self.inputs.subject_id}/bold/{run}/{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz'

            all_regressors = compile_regressors(Path(self.inputs.preprocess_dir), Path(self.inputs.bold_dir), run, self.inputs.subject_id,
                                            Path(self.inputs.fcmri_dir), bpss_path)
            regression(bpss_path, all_regressors)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["resid"] = self.inputs.resid
        outputs["resid_snr"] = self.inputs.resid_snr
        outputs["resid_sd1"] = self.inputs.resid_sd1

        return outputs
