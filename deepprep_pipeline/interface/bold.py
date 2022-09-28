from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, File, TraitedSpec, Directory, Str, traits
from run import multipool, multipool_run, multipool_BidsBolds, multipool_BidsBolds_2, multiregressionpool
import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import bids
import os
import shutil
import tensorflow as tf
import ants
import shutil
import deepprep_pipeline.voxelmorph as vxm

from deepprep_pipeline.app.filters.filters import bandpass_nifti


class BoldSkipReorientInputSpec(BaseInterfaceInputSpec):
    preprocess_dir = Directory(exists=True, desc="preprocess dir", mandatory=True)
    subject_id = Str(exists=True, desc="subject id", mandatory=True)
    subj = Str(exists=True, desc='subj', mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    deepprep_subj_path = Directory(exists=True, desc='deepprep_subj_path', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)


class BoldSkipReorientOutputSpec(TraitedSpec):
    preprocess_dir = Directory(exists=False, desc="preprocess dir")


class BoldSkipReorient(BaseInterface):
    input_spec = BoldSkipReorientInputSpec
    output_spec = BoldSkipReorientOutputSpec

    time = 14 / 60  # 运行时间：分钟
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

    def cmd(self, run):

        subj_bold_dir = Path(self.inputs.preprocess_dir) / f'{self.inputs.subject_id}' / 'bold'

        bold = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld{run}_rest.nii.gz'
        skip_bold = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld{run}_rest_skip.nii.gz'
        reorient_skip_bold = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld{run}_rest_reorient_skip.nii.gz'
        # skip 0 frame
        sh.mri_convert('-i', bold, '-o', skip_bold, _out=sys.stdout)

        # reorient
        self.swapdim(str(skip_bold), 'x', '-y', 'z', str(reorient_skip_bold))
        shutil.copy(subj_bold_dir / run / f'{self.inputs.subject_id}_bld{run}_rest_reorient_skip.nii.gz',
                  subj_bold_dir / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip.nii.gz')

    def _run_interface(self, runtime):
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        sess = layout.get_session(subject=self.inputs.subj)
        deepprep_subj_path = Path(self.inputs.deepprep_subj_path)
        tmpdir = deepprep_subj_path / 'tmp'
        trf_file = tmpdir / f'{self.inputs.subject_id}_affine.mat'
        warp_file = tmpdir / f'{self.inputs.subject_id}_warp.nii.gz'
        warped_file = tmpdir / f'{self.inputs.subject_id}_warped.nii.gz'
        subj_bold_dir = Path(self.inputs.preprocess_dir) / f'{self.inputs.subject_id}' / 'bold'
        subj_bold_dir.mkdir(parents=True, exist_ok=True)
        if len(sess) == 0:
            subj_func_path = deepprep_subj_path / 'func'
            subj_func_path.mkdir(exist_ok=True)
            shutil.copy(trf_file, subj_func_path / f'{self.inputs.subject_id}_affine.mat')
            shutil.copy(warp_file, subj_func_path / f'{self.inputs.subject_id}_warp.nii.gz')
            shutil.copy(warped_file, subj_func_path / f'{self.inputs.subject_id}_warped.nii.gz')
        else:
            for ses in sess:
                if self.inputs.task is None:
                    bids_bolds = layout.get(subject=self.inputs.subj, session=ses, suffix='bold', extension='.nii.gz')
                else:
                    bids_bolds = layout.get(subject=self.inputs.subj, session=ses, task=self.inputs.task, suffix='bold',
                                            extension='.nii.gz')
                if len(bids_bolds) == 0:
                    continue
                subj_func_path = deepprep_subj_path / f'ses-{ses}' / 'func'
                subj_func_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(trf_file, subj_func_path / f'{self.inputs.subject_id}_affine.mat')
                shutil.copy(warp_file, subj_func_path / f'{self.inputs.subject_id}_warp.nii.gz')
                shutil.copy(warped_file, subj_func_path / f'{self.inputs.subject_id}_warped.nii.gz')
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=self.inputs.subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=self.inputs.subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            bids_file = Path(bids_bold.path)
            run = f'{idx + 1:03}'
            (subj_bold_dir / run).mkdir(exist_ok=True)
            shutil.copy(bids_file, subj_bold_dir / run / f'{self.inputs.subject_id}_bld{run}_rest.nii.gz')
        runs = sorted([d.name for d in (Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        multipool_run(self.cmd, runs, Multi_Num=8)


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["preprocess_dir"] = self.inputs.preprocess_dir

        return outputs


class MotionCorrectionInputSpec(BaseInterfaceInputSpec):
    preprocess_dir = Directory(exists=True, desc="preprocess dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)


class MotionCorrectionOutputSpec(TraitedSpec):
    preprocess_dir = Directory(exists=False, desc="preprocess dir")


class MotionCorrection(BaseInterface):
    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    time = 400 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 0  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB
    def cmd(self, run):
        # ln create 001
        # run mc
        # mv result file
        link_dir = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / self.inputs.subject_id / 'bold' / run
        if not link_dir.exists():
            link_dir.mkdir(parents=True, exist_ok=True)
        link_files = os.listdir(Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run)
        link_files.remove(self.inputs.subject_id)
        try:
            os.symlink(Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / 'template.nii.gz',
                       Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / self.inputs.subject_id / 'bold' / 'template.nii.gz')
        except:
            print()
        for link_file in link_files:
            try:
                os.symlink(Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / link_file
                                , link_dir / link_file )
            except:
                continue
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run ,
            '-per-session',
            '-fsd', 'bold',
            '-fstem', f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln',
            '-fmcstem', f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc',
            '-nolog']
        sh.mc_sess(*shargs, _out=sys.stdout)
        ori_path = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run
        try:
            shutil.copy(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz')
            shutil.copy(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.mat.aff12.1D',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.mat.aff12.1D')
            shutil.copy(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz.mclog',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz.mclog')
            shutil.copy(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.mcdat',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln__mc.mcdat')
            shutil.copy(link_dir / 'mcextreg', ori_path / 'mcextreg')
            shutil.copy(link_dir / 'mcdat2extreg.log', ori_path / 'mcdat2extreg.log')
        except:
            print(())
        shutil.rmtree(ori_path / self.inputs.subject_id)
    def _run_interface(self, runtime):
        # runs = sorted([d.name for d in (Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        # runs = ['001', '002', '003', '004', '005', '006', '007', '008']
        # multipool_run(self.cmd ,runs, Multi_Num=8)

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
        outputs["preprocess_dir"] = self.inputs.preprocess_dir

        return outputs


class StcInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)


class StcOutputSpec(TraitedSpec):
    preprocess_dir = Directory(exists=False, desc='preprocess_dir')


class Stc(BaseInterface):
    input_spec = StcInputSpec
    output_spec = StcOutputSpec

    time = 214 / 60  # 运行时间：分钟
    cpu = 6  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def cmd(self, run):
        link_dir = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / self.inputs.subject_id / 'bold' / run
        if not link_dir.exists():
            link_dir.mkdir(parents=True, exist_ok=True)
        link_files = os.listdir(Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run)
        link_files.remove(self.inputs.subject_id)
        try:
            os.symlink(Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / 'template.nii.gz',
                       Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / self.inputs.subject_id / 'bold' / 'template.nii.gz')
        except:
            print()
        for link_file in link_files:
            try:
                os.symlink(Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / link_file
                                , link_dir / link_file )
            except:
                continue
        input_fname = f'{self.inputs.subject_id}_bld_rest_reorient_skip'
        output_fname = f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln'
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run,
            '-fsd', 'bold',
            '-so', 'odd',
            '-ngroups', 1,
            '-i', input_fname,
            '-o', output_fname,
            '-nolog']
        sh.stc_sess(*shargs, _out=sys.stdout)
        ori_path = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run
        try:
            shutil.copy(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz')
            shutil.copy(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz.log',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz.log')
            shutil.copy(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz.log.bak',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz.log.bak')
        except:
            print(())
        shutil.rmtree(ori_path / self.inputs.subject_id)

    def _run_interface(self, runtime):
        runs = sorted([d.name for d in (Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if
                       d.is_dir()])
        # # runs = ['001', '002', '003', '004', '005', '006', '007', '008']
        # # # runs = ['001', '002', '003', '004']
        # # # runs = ['001', '002']
        # # runs = ['001']
        multipool_run(self.cmd, runs, Multi_Num=8)

        # input_fname = f'{self.inputs.subject_id}_bld_rest_reorient_skip'
        # output_fname = f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln'
        # shargs = [
        #     '-s', self.inputs.subject_id,
        #     '-d', self.inputs.preprocess_dir,
        #     '-fsd', 'bold',
        #     '-so', 'odd',
        #     '-ngroups', 1,
        #     '-i', input_fname,
        #     '-o', output_fname,
        #     '-nolog']
        # sh.stc_sess(*shargs, _out=sys.stdout)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["preprocess_dir"] = self.inputs.preprocess_dir

        return outputs


class RegisterInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)


class RegisterOutputSpec(TraitedSpec):
    preprocess_dir = Directory(exists=False, desc='preprocess_dir')


class Register(BaseInterface):
    input_spec = RegisterInputSpec
    output_spec = RegisterOutputSpec

    time = 382 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def cmd(self, run):
        mov = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
        reg = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.register.dat'
        shargs = [
            '--bold',
            '--s', self.inputs.subject_id,
            '--mov', mov,
            '--reg', reg]
        sh.bbregister(*shargs, _out=sys.stdout)

    def _run_interface(self, runtime):
        runs = sorted([d.name for d in (Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        multipool_run(self.cmd, runs, Multi_Num=8)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["preprocess_dir"] = self.inputs.preprocess_dir

        return outputs


class MkBrainmaskInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)


class MkBrainmaskOutputSpec(TraitedSpec):
    preprocess_dir = Directory(exists=False, desc='preprocess_dir')


class MkBrainmask(BaseInterface):
    input_spec = MkBrainmaskInputSpec
    output_spec = MkBrainmaskOutputSpec

    time = 18 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 2.7  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def cmd(self, run):
        seg = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri/aparc+aseg.mgz'  # TODO 这个应该由structure_workflow传进来
        mov = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
        reg = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.register.dat'
        func = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.func.aseg.nii'
        wm = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.func.wm.nii.gz'
        vent = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.func.ventricles.nii.gz'
        targ = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri/brainmask.mgz'  # TODO 这个应该由structure_workflow传进来
        mask = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.brainmask.nii.gz'
        binmask = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.brainmask.bin.nii.gz'
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
        runs = sorted([d.name for d in (Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        multipool_run(self.cmd, runs , Multi_Num=8)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["preprocess_dir"] = self.inputs.preprocess_dir

        return outputs


class VxmRegistraionInputSpec(BaseInterfaceInputSpec):
    atlas_type = Str(desc="atlas type", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    data_path = Directory(exists=True, desc='data path', mandatory=True)
    deepprep_subj_path = Directory(exists=True, desc='deepprep subjects dir', mandatory=True)
    preprocess_dir = Directory(exists=True, desc="preprocess dir", mandatory=True)
    norm = File(exists=True, desc="mri/norm.mgz", mandatory=True)
    model_file = File(exists=True, desc="atlas_type/model.h5", mandatory=True)
    model_path = Directory(exists=True, desc="voxelmorph model dir", mandatory=True)

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

    preprocess_dir = Directory(exists=True, desc="preprocess dir")


class VxmRegistraion(BaseInterface):
    input_spec = VxmRegistraionInputSpec
    output_spec = VxmRegistraionOutputSpec

    time = 15 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 14  # 最大cpu占用：个
    gpu = 2703  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        # import tensorflow as tf
        # import ants
        # import shutil
        # import deepprep_pipeline.voxelmorph as vxm

        Path(self.inputs.deepprep_subj_path).mkdir(exist_ok=True)

        tmpdir = Path(self.inputs.deepprep_subj_path) / 'tmp'
        tmpdir.mkdir(exist_ok=True)

        # model_path = Path(__file__).parent.parent / 'model' / 'voxelmorph' / self.inputs.atlas_type
        # atlas
        if self.inputs.atlas_type == 'MNI152_T1_1mm':
            atlas_path = '../../data/atlas/MNI152_T1_1mm_brain.nii.gz'
            vxm_atlas_path = '../../data/atlas/MNI152_T1_1mm_brain_vxm.nii.gz'
            vxm_atlas_npz_path = '../../data/atlas/MNI152_T1_1mm_brain_vxm.npz'
            vxm2atlas_trf = '../../data/atlas/MNI152_T1_1mm_vxm2atlas.mat'
        elif self.inputs.atlas_type == 'MNI152_T1_2mm':
            atlas_path = self.inputs.model_path / 'MNI152_T1_2mm_brain.nii.gz'
            vxm_atlas_path = self.inputs.model_path / 'MNI152_T1_2mm_brain_vxm.nii.gz'
            vxm_atlas_npz_path = self.inputs.model_path / 'MNI152_T1_2mm_brain_vxm.npz'
            vxm2atlas_trf = self.inputs.model_path / 'MNI152_T1_2mm_vxm2atlas.mat'
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

        outputs["preprocess_dir"] = self.inputs.preprocess_dir

        return outputs


class RestGaussInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, mandatory=True, desc='subject')
    preprocess_dir = Directory(exists=True, mandatory=True, desc='preprocess_dir')


class RestGaussOutputSpec(TraitedSpec):
    preprocess_dir = Directory(exists=False, desc='preprocess_dir')


class RestGauss(BaseInterface):
    input_spec = RestGaussInputSpec
    output_spec = RestGaussOutputSpec

    time = 11 / 60  # 运行时间：分钟
    cpu = 0  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def cmd(self, run):
        from deepprep_pipeline.app.filters.filters import gauss_nifti

        mc = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
        fcmri_dir = Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'fcmri'
        Path(fcmri_dir).mkdir(parents=True, exist_ok=True)
        gauss_nifti(str(mc), 1000000000)

    def _run_interface(self, runtime):
        runs = sorted([d.name for d in (Path(self.inputs.preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        multipool_run(self.cmd, runs, Multi_Num=8)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["preprocess_dir"] = self.inputs.preprocess_dir
        return outputs


class RestBandpassInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    subj = Str(exists=True, desc='subj', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)
    data_path = Directory(exists=True, desc='data path', mandatory=True)


class RestBandpassOutputSpec(TraitedSpec):
    preprocess_dir = Directory(exists=False, desc='preprocess_dir')


class RestBandpass(BaseInterface):
    input_spec = RestBandpassInputSpec
    output_spec = RestBandpassOutputSpec

    time = 120 / 60  # 运行时间：分钟
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def cmd(self, idx, bids_entities, bids_path):
        from deepprep_pipeline.app.filters.filters import bandpass_nifti

        entities = dict(bids_entities)
        print(entities)
        if 'RepetitionTime' in entities:
            TR = entities['RepetitionTime']
        else:
            bold = ants.image_read(bids_path)
            TR = bold.spacing[3]
        run = f'{idx + 1:03}'
        gauss_path = f'{self.inputs.preprocess_dir}/{self.inputs.subject_id}/bold/{run}/{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz'
        bandpass_nifti(gauss_path, TR)

    def _run_interface(self, runtime):
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)

        if self.inputs.task is None:
            bids_bolds = layout.get(subject=self.inputs.subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=self.inputs.subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        all_idx = []
        all_bids_entities = []
        all_bids_path = []
        for idx, bids_bold in enumerate(bids_bolds):
            all_idx.append(idx)
            all_bids_entities.append(bids_bold.entities)
            all_bids_path.append(bids_bold.path)
        multipool_BidsBolds(self.cmd, all_idx, all_bids_entities, all_bids_path, Multi_Num=8)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["preprocess_dir"] = self.inputs.preprocess_dir

        return outputs


class RestRegressionInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    subj = Str(exists=True, desc='subj', mandatory=True)
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)
    data_path = Directory(exists=True, desc='data_path', mandatory=True)
    deepprep_subj_path = Directory(exists=True, desc='deepprep_subj_path', mandatory=True)
    fcmri_dir = Directory(exists=False, desc='fcmri_dir', mandatory=True)
    bold_dir = Directory(exists=False, desc='bold_dir', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)


class RestRegressionOutputSpec(TraitedSpec):
    preprocess_dir = Directory(exists=False, desc='preprocess_dir')


class RestRegression(BaseInterface):
    input_spec = RestRegressionInputSpec
    output_spec = RestRegressionOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB

    def setenv_smooth_downsampling(self):
        subjects_dir = Path(self.inputs.subjects_dir)
        fsaverage6_dir = subjects_dir / 'fsaverage6'
        if not fsaverage6_dir.exists():
            src_fsaverage6_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage6'
            os.symlink(src_fsaverage6_dir, fsaverage6_dir)

        fsaverage5_dir = subjects_dir / 'fsaverage5'
        if not fsaverage5_dir.exists():
            src_fsaverage5_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage5'
            os.symlink(src_fsaverage5_dir, fsaverage5_dir)

        fsaverage4_dir = subjects_dir / 'fsaverage4'
        if not fsaverage4_dir.exists():
            src_fsaverage4_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage4'
            os.symlink(src_fsaverage4_dir, fsaverage4_dir)

    # smooth_downsampling
    # def cmd(self, hemi, subj_surf_path, dst_resid_file, dst_reg_file):
    #     from deepprep_pipeline.app.surface_projection import surface_projection as sp
    #     fs6_path = sp.indi_to_fs6(subj_surf_path, f'{self.inputs.subject_id}', dst_resid_file, dst_reg_file,
    #                               hemi)
    #     sm6_path = sp.smooth_fs6(fs6_path, hemi)
    #     sp.downsample_fs6_to_fs4(sm6_path, hemi)
    def _run_interface(self, runtime):
        from deepprep_pipeline.app.regressors.regressors import compile_regressors, regression
        from deepprep_pipeline.app.surface_projection import surface_projection as sp
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)

        if self.inputs.task is None:
            bids_bolds = layout.get(subject=self.inputs.subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=self.inputs.subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            run = f'{idx + 1:03}'
            bpss_path = f'{self.inputs.preprocess_dir}/{self.inputs.subject_id}/bold/{run}/{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz'

            all_regressors = compile_regressors(Path(self.inputs.preprocess_dir), Path(self.inputs.bold_dir), run, self.inputs.subject_id,
                                            Path(self.inputs.fcmri_dir), bpss_path)
            regression(bpss_path, all_regressors)

        self.setenv_smooth_downsampling()
        deepprep_subj_path = Path(self.inputs.deepprep_subj_path)

        subj_bold_dir = Path(self.inputs.preprocess_dir) / f'{self.inputs.subject_id}' / 'bold'
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
            src_mc_file = subj_bold_dir / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
            dst_mc_file = subj_func_path / f'{file_prefix}_mc.nii.gz'
            shutil.copy(src_mc_file, dst_mc_file)
            src_reg_file = subj_bold_dir / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.register.dat'
            dst_reg_file = subj_func_path / f'{file_prefix}_bbregister.register.dat'
            shutil.copy(src_reg_file, dst_reg_file)

            # hemi = ['lh','rh']
            # multiregressionpool(self.cmd, hemi, subj_surf_path, dst_resid_file, dst_reg_file, Multi_Num=2)
            for hemi in ['lh','rh']:
                fs6_path = sp.indi_to_fs6(subj_surf_path, f'{self.inputs.subject_id}', dst_resid_file, dst_reg_file,hemi)
                sm6_path = sp.smooth_fs6(fs6_path, hemi)
                sp.downsample_fs6_to_fs4(sm6_path, hemi)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["preprocess_dir"] = self.inputs.preprocess_dir
        # outputs["resid"] = self.inputs.resid
        # outputs["resid_snr"] = self.inputs.resid_snr
        # outputs["resid_sd1"] = self.inputs.resid_sd1

        return outputs


class VxmRegNormMNI152InputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    subj = Str(exists=True, desc='subj', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    data_path = Directory(exists=True, desc='data_path', mandatory=True)
    deepprep_subj_path = Directory(exists=True, desc='deepprep_subj_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='tmp/ task-{task}', mandatory=True)
    norm = File(exists=True, desc='mri/norm.mgz', mandatory=True)


class VxmRegNormMNI152OutputSpec(TraitedSpec):
    deepprep_subj_path = Directory(exists=True, desc='deepprep_subj_path')
    preprocess_dir = Directory(exists=True, desc='tmp/ task-{task}', mandatory=True)


class VxmRegNormMNI152(BaseInterface):
    input_spec = VxmRegNormMNI152InputSpec
    output_spec = VxmRegNormMNI152OutputSpec

    time = 503 / 60  # 运行时间：分钟
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def register_dat_to_fslmat(self, mov_file, ref_file, reg_file, fslmat_file):
        sh.tkregister2('--mov', mov_file,
                       '--targ', ref_file,
                       '--reg', reg_file,
                       '--fslregout', fslmat_file,
                       '--noedit')

    def register_dat_to_trf(self, mov_file, ref_file, reg_file, preprocess_dir, trf_file):
        import SimpleITK as sitk

        fsltrf_file = os.path.join(preprocess_dir, 'fsl_trf.fsl')
        self.register_dat_to_fslmat(mov_file, ref_file, reg_file, fsltrf_file)
        first_frame_file = os.path.join(preprocess_dir, 'frame0.nii.gz')
        bold = ants.image_read(str(mov_file))
        frame0_np = bold[:, :, :, 0]
        origin = bold.origin[:3]
        spacing = bold.spacing[:3]
        direction = bold.direction[:3, :3].copy()
        frame0 = ants.from_numpy(frame0_np, origin=origin, spacing=spacing, direction=direction)
        ants.image_write(frame0, str(first_frame_file))
        tfm_file = os.path.join(preprocess_dir, 'itk_trf.tfm')
        base_path, _ = os.path.split(os.path.abspath(__file__))
        base_path = Path(base_path).parent
        c3d_affine_tool = os.path.join(base_path, 'resource', 'c3d_affine_tool')
        cmd = f'{c3d_affine_tool} -ref {ref_file} -src {first_frame_file} {fsltrf_file} -fsl2ras -oitk {tfm_file}'
        os.system(cmd)
        trf_sitk = sitk.ReadTransform(tfm_file)
        trf = ants.new_ants_transform()
        trf.set_parameters(trf_sitk.GetParameters())
        trf.set_fixed_parameters(trf_sitk.GetFixedParameters())
        ants.write_transform(trf, trf_file)

    def native_bold_to_T1_2mm_ants(self, residual_file, subj, subj_t1_file, reg_file, save_file, preprocess_dir,
                                   verbose=False):
        subj_t1_2mm_file = os.path.join(os.path.split(save_file)[0], 'norm_2mm.nii.gz')
        sh.mri_convert('-ds', 2, 2, 2,
                       '-i', subj_t1_file,
                       '-o', subj_t1_2mm_file)
        trf_file = os.path.join(preprocess_dir, 'reg.mat')
        self.register_dat_to_trf(residual_file, subj_t1_2mm_file, reg_file, preprocess_dir, trf_file)
        bold_img = ants.image_read(str(residual_file))
        fixed = ants.image_read(subj_t1_2mm_file)
        affined_bold_img = ants.apply_transforms(fixed=fixed, moving=bold_img, transformlist=[trf_file], imagetype=3)
        if verbose:
            ants.image_write(affined_bold_img, save_file)
        return affined_bold_img

    def vxm_warp_bold_2mm(self,resid_t1, affine_file, warp_file, warped_file, verbose=True):
        import deepprep_pipeline.voxelmorph as vxm
        import tensorflow as tf
        import time

        # TODO 这里把模板名字写死会有问题，现在只支持MNI152_T1_2mm的分辨率？
        atlas_file = Path(__file__).parent.parent / 'model' / 'voxelmorph' / 'MNI152_T1_2mm' / 'MNI152_T1_2mm_brain_vxm.nii.gz'
        MNI152_2mm_file = Path(__file__).parent.parent/ 'model' / 'voxelmorph' / 'MNI152_T1_2mm' / 'MNI152_T1_2mm_brain.nii.gz'
        MNI152_2mm = ants.image_read(str(MNI152_2mm_file))
        atlas = ants.image_read(str(atlas_file))
        if isinstance(resid_t1, str):
            bold_img = ants.image_read(resid_t1)
        else:
            bold_img = resid_t1
        n_frame = bold_img.shape[3]
        bold_origin = bold_img.origin
        bold_spacing = bold_img.spacing
        bold_direction = bold_img.direction.copy()

        # tensorflow device handling
        gpuid = '0'
        device, nb_devices = vxm.tf.utils.setup_device(gpuid)

        fwdtrf_MNI152_2mm = [str(affine_file)]
        trf_file = Path(__file__).parent.parent / 'model' / 'voxelmorph' / 'MNI152_T1_2mm' / 'MNI152_T1_2mm_vxm2atlas.mat'
        fwdtrf_atlas2MNI152_2mm = [str(trf_file)]
        deform, deform_affine = vxm.py.utils.load_volfile(str(warp_file), add_batch_axis=True, ret_affine=True)

        # affine to MNI152 croped
        tic = time.time()
        # affined_img = ants.apply_transforms(atlas, bold_img, fwdtrf_MNI152_2mm, imagetype=3)
        affined_np = ants.apply_transforms(atlas, bold_img, fwdtrf_MNI152_2mm, imagetype=3).numpy()
        # print(sys.getrefcount(affined_img))
        # del affined_img
        toc = time.time()
        print(toc - tic)
        # gc.collect()
        # voxelmorph warp
        tic = time.time()
        warped_np = np.zeros(shape=(*atlas.shape, n_frame), dtype=np.float32)
        with tf.device(device):
            transform = vxm.networks.Transform(atlas.shape, interp_method='linear', nb_feats=1)
            # for idx in range(affined_np.shape[3]):
            #     frame_np = affined_np[:, :, :, idx]
            #     frame_np = frame_np[..., np.newaxis]
            #     frame_np = frame_np[np.newaxis, ...]
            #
            #     moved = transform.predict([frame_np, deform])
            #     warped_np[:, :, :, idx] = moved.squeeze()
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
                print(f'batch: {idx}')
            del transform
            del tf_dataset
            del moved
            del moved_data
        toc = time.time()
        print(toc - tic)

        # affine to MNI152
        tic = time.time()
        origin = (*atlas.origin, bold_origin[3])
        spacing = (*atlas.spacing, bold_spacing[3])
        direction = bold_direction.copy()
        direction[:3, :3] = atlas.direction

        warped_img = ants.from_numpy(warped_np, origin=origin, spacing=spacing, direction=direction)
        del warped_np
        moved_img = ants.apply_transforms(MNI152_2mm, warped_img, fwdtrf_atlas2MNI152_2mm, imagetype=3)
        del warped_img
        moved_np = moved_img.numpy()
        del moved_img
        toc = time.time()
        print(toc - tic)

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
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=self.inputs.subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=self.inputs.subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')

        bids_entities = []
        bids_path = []
        for bids_bold in bids_bolds:
            entities = dict(bids_bold.entities)
            file_prefix = Path(bids_bold.path).name.replace('.nii.gz', '')
            if 'session' in entities:
                ses = entities['session']
                subj_func_path = Path(self.inputs.deepprep_subj_path) / f'ses-{ses}' / 'func'
            else:
                subj_func_path = Path(self.inputs.deepprep_subj_path) / 'func'
            if self.inputs.preprocess_method == 'rest':
                bold_file = subj_func_path / f'{file_prefix}_resid.nii.gz'
            else:
                bold_file = subj_func_path / f'{file_prefix}_mc.nii.gz'

            reg_file = subj_func_path / f'{file_prefix}_bbregister.register.dat'
            bold_t1_file = subj_func_path / f'{self.inputs.subj}_native_t1_2mm.nii.gz'
            bold_t1_out = self.native_bold_to_T1_2mm_ants(bold_file, self.inputs.subj, self.inputs.norm, reg_file,
                                                          bold_t1_file, self.inputs.preprocess_dir, verbose=False)

            warp_file = subj_func_path / f'sub-{self.inputs.subj}_warp.nii.gz'
            affine_file = subj_func_path / f'sub-{self.inputs.subj}_affine.mat'
            warped_file = subj_func_path / f'sub-{self.inputs.subj}_MNI2mm.nii.gz'
            warped_img = self.vxm_warp_bold_2mm(bold_t1_out, affine_file, warp_file, warped_file, verbose=True)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["deepprep_subj_path"] = self.inputs.deepprep_subj_path
        outputs["preprocess_dir"] = self.inputs.preprocess_dir

        return outputs


class SmoothInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    subj = Str(exists=True, desc='subj', mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    data_path = Directory(exists=True, desc='data_path', mandatory=True)
    deepprep_subj_path = Directory(exists=True, desc='deepprep_subj_path', mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='tmp/ task-{task}', mandatory=True)
    MNI152_T1_2mm_brain_mask = File(exists=True, desc='MNI152 brain mask path', mandatory=True)


class SmoothOutputSpec(TraitedSpec):
    deepprep_subj_path = Directory(exists=True, desc='deepprep_subj_path')


class Smooth(BaseInterface):
    input_spec = SmoothInputSpec
    output_spec = SmoothOutputSpec

    time = 68 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

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
        if 'session' in entities:
            ses = entities['session']
            subj_func_path = Path(self.inputs.deepprep_subj_path) / f'ses-{ses}' / 'func'
        else:
            subj_func_path = Path(self.inputs.deepprep_subj_path) / 'func'
        if self.inputs.preprocess_method == 'rest':
            bold_file = subj_func_path / f'{file_prefix}_resid.nii.gz'
            save_file = subj_func_path / f'{file_prefix}_resid_MIN2mm_sm6.nii.gz'
        else:
            bold_file = subj_func_path / f'{file_prefix}_mc.nii.gz'
            save_file = subj_func_path / f'{file_prefix}_mc_MIN2mm.nii.gz'
        if self.inputs.preprocess_method == 'rest':
            temp_file = Path(self.inputs.preprocess_dir) / f'{file_prefix}_MNI2mm_sm6_temp.nii.gz'
            warped_img = subj_func_path / f'{self.inputs.subject_id}_MNI2mm.nii.gz'
            self.bold_smooth_6_ants(str(warped_img), save_file, temp_file, bold_file, verbose=True)
        else:
            temp_file = Path(self.inputs.preprocess_dir) / f'{file_prefix}_MNI2mm_temp.nii.gz'
            warped_img = subj_func_path / f'{self.inputs.subject_id}_MNI2mm.nii.gz'
            self.save_bold(str(warped_img), temp_file, bold_file, save_file)
    def _run_interface(self, runtime):
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=self.inputs.subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=self.inputs.subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        bids_entities = []
        bids_path = []
        for bids_bold in bids_bolds:
            bids_entities.append(bids_bold.entities)
            bids_path.append(bids_bold.path)
        multipool_BidsBolds_2(self.cmd, bids_entities, bids_path, Multi_Num=8)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["deepprep_subj_path"] = self.inputs.deepprep_subj_path

        return outputs


class MkTemplateInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    preprocess_dir = Directory(exists=True, desc='preprocess_dir', mandatory=True)


class MkTemplateOutputSpec(TraitedSpec):
    preprocess_dir = Directory(exists=False, desc='preprocess_dir')


class MkTemplate(BaseInterface):
    input_spec = MkTemplateInputSpec
    output_spec = MkTemplateOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB
    def _run_interface(self, runtime):
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', self.inputs.preprocess_dir,
            '-fsd', 'bold',
            '-funcstem', f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln',
            '-nolog']
        sh.mktemplate_sess(*shargs, _out=sys.stdout)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["preprocess_dir"] = self.inputs.preprocess_dir

        return outputs

