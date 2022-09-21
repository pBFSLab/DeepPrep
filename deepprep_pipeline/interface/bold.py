from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, Directory, Str
import sys
import sh
import nibabel as nib
import numpy as np


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
    mov = File(exists=True,  desc='{subj}_bld_rest_reorient_skip_faln_mc.nii.gz', mandatory=True)
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