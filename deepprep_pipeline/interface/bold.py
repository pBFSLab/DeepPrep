from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, Directory, Str
import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import os
import ants
import bids
import pandas as pd
import csv
from sklearn.decomposition import PCA

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

    time = 0 / 60  # 运行时间：分钟 / 单run测试时间
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

    time = 823 / 60  # 运行时间：分钟
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
    fcmri = File(exists=True, mandatory=True, desc='fcmri')
    mc = File(exists=True, mandatory=True, desc='{subj}_bld_rest_reorient_skip_faln_mc.nii.gz')
    gauss = File(mandatory=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz')


class RestGaussOutputSpec(TraitedSpec):
    gauss = File(exists=True, mandatory=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz')


class RestGauss(BaseInterface):
    input_spec = RestGaussInputSpec
    output_spec = RestGaussOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        self.inputs.gauss = gauss_nifti(str(self.inputs.mc), 1000000000)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["gauss"] = self.inputs.gauss
        return outputs


class RestBandpassInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, mandatory=True, desc='subject')
    preprocess_dir = Directory(exists=True, mandatory=True, desc='preprocess_dir')
    fcmri = File(exists=True, mandatory=True, desc='fcmri')
    gauss = File(exists=True, mandatory=True, desc='{subj}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz')
    bpss = File(mandatory=True, desc='bpss_path')
    bids_bolds = File(exists=True, mandatory=True, desc='bids_bolds')


class RestBandpassOutputSpec(TraitedSpec):
    bpss = File(exists=True, mandatory=True, desc='bpss_path')


class RestBandpass(BaseInterface):
    input_spec = RestBandpassInputSpec
    output_spec = RestBandpassOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        def qnt_nifti(bpss_path, maskpath, outpath):
            '''

            bpss_path - path. Path of bold after bpass process.
            maskpath - Path to file containing mask.
            outpath  - Path to file to place the output.
            '''

            # Open mask.
            mask_img = nib.load(maskpath)
            mask = mask_img.get_data().flatten() > 0
            nvox = float(mask.sum())
            assert nvox > 0, 'Null mask found in %s' % maskpath

            p = 0
            with outpath.open('w') as f:
                img = nib.load(bpss_path)
                data = img.get_data()
                for iframe in range(data.shape[-1]):
                    frame = data[:, :, :, iframe].flatten()
                    total = frame[mask].sum()
                    q = total / nvox

                    if iframe == 0:
                        diff = 0.0
                    else:
                        diff = q - p
                    f.write('%10.4f\t%10.4f\n' % (q, diff))
                    p = q

        def regressor_PCA_singlebold(pca_data, n):
            pca = PCA(n_components=n, random_state=False)
            pca_regressor = pca.fit_transform(pca_data.T)
            return pca_regressor

        def regressors_PCA(bpss_path, maskpath, outpath):
            '''
            Generate PCA regressor from outer points of brain.
                bpss_path - path. Path of bold after bpass process.
                maskpath - Path to file containing mask.
                outpath  - Path to file to place the output.
            '''
            # PCA parameter.
            n = 10

            # Open mask.
            mask_img = nib.load(maskpath)
            mask = mask_img.get_data().swapaxes(0, 1)
            mask = mask.flatten(order='F') == 0
            nvox = float(mask.sum())
            assert nvox > 0, 'Null mask found in %s' % maskpath

            with outpath.open('w') as f:
                img = nib.load(bpss_path)
                data = img.get_data().swapaxes(0, 1)
                vol_data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]), order='F')
                pca_data = vol_data[mask]
                pca_regressor = regressor_PCA_singlebold(pca_data, n)
                for iframe in range(data.shape[-1]):
                    for idx in range(n):
                        f.write('%10.4f\t' % (pca_regressor[iframe, idx]))
                    f.write('\n')

        def build_movement_regressors(subject, bldrun, bold_path: Path, movement_path: Path, fcmri_path: Path):
            # *.mcdata -> *.par
            mcdat_file = bold_path / bldrun / f'{subject}_bld_rest_reorient_skip_faln_mc.mcdat'
            par_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.par'
            mcdat = pd.read_fwf(mcdat_file, header=None).to_numpy()
            par = mcdat[:, 1:7]
            par_txt = list()
            for row in par:
                par_txt.append(f'{row[0]:.4f}  {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}  {row[4]:.4f}  {row[5]:.4f}')
            with open(par_file, 'w') as f:
                f.write('\n'.join(par_txt))

            # *.par -> *.dat
            dat_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.dat'
            dat = mcdat[:, [4, 5, 6, 1, 2, 3]]
            dat_txt = list()
            for idx, row in enumerate(dat):
                dat_line = f'{idx + 1}{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}{1:10.6f}'
                dat_txt.append(dat_line)
            with open(dat_file, 'w') as f:
                f.write('\n'.join(dat_txt))

            # *.par -> *.ddat
            ddat_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.ddat'
            ddat = mcdat[:, [4, 5, 6, 1, 2, 3]]
            ddat = ddat[1:, :] - ddat[:-1, ]
            ddat = np.vstack((np.zeros((1, 6)), ddat))
            ddat_txt = list()
            for idx, row in enumerate(ddat):
                if idx == 0:
                    ddat_line = f'{idx + 1}{0:10.6f}{0:10.6f}{0:10.6f}{0:10.6f}{0:10.6f}{0:10.6f}{1:10.6f}'
                else:
                    ddat_line = f'{idx + 1}{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}{0:10.6f}'
                ddat_txt.append(ddat_line)
            with open(ddat_file, 'w') as f:
                f.write('\n'.join(ddat_txt))

            # *.par -> *.rdat
            rdat_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.rdat'
            rdat = mcdat[:, [4, 5, 6, 1, 2, 3]]
            # rdat_average = np.zeros(rdat.shape[1])
            # for idx, row in enumerate(rdat):
            #     rdat_average = (row + rdat_average * idx) / (idx + 1)
            rdat_average = rdat.mean(axis=0)
            rdat = rdat - rdat_average
            rdat_txt = list()
            for idx, row in enumerate(rdat):
                rdat_line = f'{idx + 1}{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}{1:10.6f}'
                rdat_txt.append(rdat_line)
            with open(rdat_file, 'w') as f:
                f.write('\n'.join(rdat_txt))

            # *.rdat, *.ddat -> *.rddat
            rddat_file = movement_path / f'{subject}_bld{bldrun}_rest_reorient_skip_faln_mc.rddat'
            rddat = np.hstack((rdat, ddat))
            rddat_txt = list()
            for idx, row in enumerate(rddat):
                rddat_line = f'{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}\t' + \
                             f'{row[6]:10.6f}{row[7]:10.6f}{row[8]:10.6f}{row[9]:10.6f}{row[10]:10.6f}{row[11]:10.6f}'
                rddat_txt.append(rddat_line)
            with open(rddat_file, 'w') as f:
                f.write('\n'.join(rddat_txt))

            regressor_dat_file = fcmri_path / f'{subject}_mov_regressor.dat'
            rddat = np.around(rddat, 6)
            n = rddat.shape[0]
            ncol = rddat.shape[1]
            x = np.zeros(n)
            for i in range(n):
                x[i] = -1. + 2. * i / (n - 1)

            sxx = n * (n + 1) / (3. * (n - 1))

            sy = np.zeros(ncol)
            sxy = np.zeros(ncol)
            a0 = np.zeros(ncol)
            a1 = np.zeros(ncol)
            for j in range(ncol - 1):
                sy[j] = 0
                sxy[j] = 0
                for i in range(n):
                    sy[j] += rddat[i, j]
                    sxy[j] += rddat[i, j] * x[i]
                a0[j] = sy[j] / n
                a1[j] = sxy[j] / sxx
                for i in range(n):
                    rddat[i, j] -= a1[j] * x[i]

            regressor_dat_txt = list()
            for idx, row in enumerate(rddat):
                regressor_dat_line = f'{row[0]:10.6f}{row[1]:10.6f}{row[2]:10.6f}{row[3]:10.6f}{row[4]:10.6f}{row[5]:10.6f}' + \
                                     f'{row[6]:10.6f}{row[7]:10.6f}{row[8]:10.6f}{row[9]:10.6f}{row[10]:10.6f}{row[11]:10.6f}'
                regressor_dat_txt.append(regressor_dat_line)
            with open(regressor_dat_file, 'w') as f:
                f.write('\n'.join(regressor_dat_txt))

        def compile_regressors(preprocess_dir, bold_path, bldrun, subject, fcmri_path, bpss_path):
            # Compile the regressors.
            movement_path = preprocess_dir / subject / 'movement'
            movement_path.mkdir(exist_ok=True)

            # wipe mov regressors, if there
            mov_regressor_common_path = fcmri_path / ('%s_mov_regressor.dat' % subject)
            build_movement_regressors(subject, bldrun, bold_path, movement_path, fcmri_path)
            mov_regressor_path = fcmri_path / ('%s_bld%s_mov_regressor.dat' % (subject, bldrun))
            os.rename(mov_regressor_common_path, mov_regressor_path)

            mask_path = bold_path / bldrun / ('%s.brainmask.bin.nii.gz' % subject)
            out_path = fcmri_path / ('%s_bld%s_WB_regressor_dt.dat' % (subject, bldrun))
            qnt_nifti(bpss_path, str(mask_path), out_path)

            mask_path = bold_path / bldrun / ('%s.func.ventricles.nii.gz' % subject)
            vent_out_path = fcmri_path / ('%s_bld%s_ventricles_regressor_dt.dat' % (subject, bldrun))
            qnt_nifti(bpss_path, str(mask_path), vent_out_path)

            mask_path = bold_path / bldrun / ('%s.func.wm.nii.gz' % subject)
            wm_out_path = fcmri_path / ('%s_bld%s_wm_regressor_dt.dat' % (subject, bldrun))
            qnt_nifti(bpss_path, str(mask_path), wm_out_path)

            pasted_out_path = fcmri_path / ('%s_bld%s_vent_wm_dt.dat' % (subject, bldrun))
            with pasted_out_path.open('w') as f:
                sh.paste(vent_out_path, wm_out_path, _out=f)

            # Generate PCA regressors of bpss nifti.
            mask_path = bold_path / bldrun / ('%s.brainmask.nii.gz' % subject)
            pca_out_path = fcmri_path / ('%s_bld%s_pca_regressor_dt.dat' % (subject, bldrun))
            regressors_PCA(bpss_path, str(mask_path), pca_out_path)

            fnames = [
                fcmri_path / ('%s_bld%s_mov_regressor.dat' % (subject, bldrun)),
                fcmri_path / ('%s_bld%s_WB_regressor_dt.dat' % (subject, bldrun)),
                fcmri_path / ('%s_bld%s_vent_wm_dt.dat' % (subject, bldrun)),
                fcmri_path / ('%s_bld%s_pca_regressor_dt.dat' % (subject, bldrun))]
            all_regressors_path = fcmri_path / ('%s_bld%s_regressors.dat' % (subject, bldrun))
            regressors = []
            for fname in fnames:
                with fname.open('r') as f:
                    regressors.append(
                        np.array([
                            list(map(float, line.replace('-', ' -').strip().split()))
                            for line in f]))
            regressors = np.hstack(regressors)
            with all_regressors_path.open('w') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(regressors)

            # Prepare regressors datas for download
            download_all_regressors_path = Path(str(all_regressors_path).replace('.dat', '_download.txt'))
            num_row = len(regressors[:, 0])
            frame_no = np.arange(num_row).reshape((num_row, 1))
            download_regressors = np.concatenate((frame_no, regressors), axis=1)
            label_header = ['Frame', 'dL', 'dP', 'dS', 'pitch', 'yaw', 'roll',
                            'dL_d', 'dP_d', 'dS_d', 'pitch_d', 'yaw_d', 'roll_d',
                            'WB', 'WB_d', 'vent', 'vent_d', 'wm', 'wm_d',
                            'comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6', 'comp7', 'comp8', 'comp9', 'comp10']
            with download_all_regressors_path.open('w') as f:
                csv.writer(f, delimiter=' ').writerows([label_header])
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(download_regressors)

            return all_regressors_path

        for idx, bids_bold in enumerate(self.inputs.bids_bolds):
            entities = dict(bids_bold.entities)
            if 'RepetitionTime' in entities:
                TR = entities['RepetitionTime']
            else:
                bold = ants.image_read(bids_bold.path)
                TR = bold.spacing[3]
            self.inputs.bpss = bandpass_nifti(self.inputs.gauss, TR)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["bpss"] = self.inputs.bpss

        return outputs


class RestRegressionInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, mandatory=True, desc='subject')
    preprocess_dir = Directory(exists=True, mandatory=True, desc='preprocess_dir')
    fcmri = File(exists=True, mandatory=True, desc='fcmri')
    bold = File(exists=True, mandatory=True, desc='bold')
    bpss = File(mandatory=True, desc='bpss_path')
    all_regressors = File(mandatory=True, desc='all_regressors_path')


class RestRegressionOutputSpec(TraitedSpec):
    all_regressors = File(exists=True, mandatory=True, desc='all_regressors_path')


class RestRegression(BaseInterface):
    input_spec = RestRegressionInputSpec
    output_spec = RestRegressionOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        regression(self.inputs.bpss, self.inputs.all_regressors)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["all_regressors"] = self.inputs.all_regressors

        return outputs