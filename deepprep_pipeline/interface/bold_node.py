from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, File, TraitedSpec, Directory, Str
from interface.run import multipool_run, multipool_BidsBolds, multipool_BidsBolds_2
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

    time = 14 / 60  # 运行时间：分钟
    cpu = 2.5  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(BoldSkipReorient, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task
        for run in runs:
            BoldSkipReorient_output_files = [f'{sub}_bld{run}_rest.nii.gz', f'{sub}_bld{run}_rest_reorient_skip.nii.gz',
                                             f'{sub}_bld{run}_rest_skip.nii.gz', f'{sub}_bld_rest_reorient_skip.nii.gz']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(BoldSkipReorient_output_files) <= set(output_list)
            if not check_result:
                raise FileExistsError

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
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        subj_bold_dir = preprocess_dir / f'{self.inputs.subject_id}' / 'bold'

        bold = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld{run}_rest.nii.gz'
        skip_bold = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld{run}_rest_skip.nii.gz'
        reorient_skip_bold = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld{run}_rest_reorient_skip.nii.gz'
        # skip 0 frame
        sh.mri_convert('-i', bold, '-o', skip_bold, _out=sys.stdout)

        # reorient
        self.swapdim(str(skip_bold), 'x', '-y', 'z', str(reorient_skip_bold))
        shutil.copy(subj_bold_dir / run / f'{self.inputs.subject_id}_bld{run}_rest_reorient_skip.nii.gz',
                    subj_bold_dir / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip.nii.gz')

    def _run_interface(self, runtime):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        deepprep_subj_path = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        subj = self.inputs.subject_id.split('-')[1]
        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        sess = layout.get_session(subject=subj)
        tmpdir = deepprep_subj_path / 'tmp'
        trf_file = tmpdir / f'{self.inputs.subject_id}_affine.mat'
        warp_file = tmpdir / f'{self.inputs.subject_id}_warp.nii.gz'
        warped_file = tmpdir / f'{self.inputs.subject_id}_warped.nii.gz'
        subj_bold_dir = Path(preprocess_dir) / f'{self.inputs.subject_id}' / 'bold'
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
                    bids_bolds = layout.get(subject=subj, session=ses, suffix='bold', extension='.nii.gz')
                else:
                    bids_bolds = layout.get(subject=subj, session=ses, task=self.inputs.task, suffix='bold',
                                            extension='.nii.gz')
                if len(bids_bolds) == 0:
                    continue
                subj_func_path = deepprep_subj_path / f'ses-{ses}' / 'func'
                subj_func_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(trf_file, subj_func_path / f'{self.inputs.subject_id}_affine.mat')
                shutil.copy(warp_file, subj_func_path / f'{self.inputs.subject_id}_warp.nii.gz')
                shutil.copy(warped_file, subj_func_path / f'{self.inputs.subject_id}_warped.nii.gz')
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        for idx, bids_bold in enumerate(bids_bolds):
            bids_file = Path(bids_bold.path)
            run = f'{idx + 1:03}'
            (subj_bold_dir / run).mkdir(exist_ok=True)
            shutil.copy(bids_file, subj_bold_dir / run / f'{self.inputs.subject_id}_bld{run}_rest.nii.gz')
        runs = sorted(
            [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        multipool_run(self.cmd, runs, Multi_Num=8)

        self.check_output(runs)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id

        return outputs

    def create_sub_node(self):
        from interface.create_node_bold import create_Stc_node
        node = create_Stc_node(self.inputs.subject_id,
                               self.inputs.task,
                               self.inputs.atlas_type,
                               self.inputs.preprocess_method)

        return node


class MotionCorrectionInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject_id', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class MotionCorrectionOutputSpec(TraitedSpec):
    data_path = Directory(exists=True, desc="data path")
    subject_id = Str(exists=True, desc="subject id")


class MotionCorrection(BaseInterface):
    input_spec = MotionCorrectionInputSpec
    output_spec = MotionCorrectionOutputSpec

    time = 400 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 0  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(MotionCorrection, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task
        for run in runs:
            MotionCorrection_output_files = ['mcextreg', f'{sub}_bld_rest_reorient_skip_faln_mc.mat.aff12.1D',
                                             f'{sub}_bld_rest_reorient_skip_faln_mc.mcdat',
                                             f'{sub}_bld_rest_reorient_skip_faln_mc.nii.gz']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(MotionCorrection_output_files) <= set(output_list)
            if not check_result:
                return FileExistsError

    def cmd(self, run):
        # ln create 001
        # run mc
        # mv result file
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        link_dir = preprocess_dir / self.inputs.subject_id / 'bold' / run / self.inputs.subject_id / 'bold' / run
        if not link_dir.exists():
            link_dir.mkdir(parents=True, exist_ok=True)
        link_files = os.listdir(Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run)
        link_files.remove(self.inputs.subject_id)
        try:
            os.symlink(Path(preprocess_dir) / self.inputs.subject_id / 'bold' / 'template.nii.gz',
                       Path(
                           preprocess_dir) / self.inputs.subject_id / 'bold' / run / self.inputs.subject_id / 'bold' / 'template.nii.gz')
        except:
            pass
        for link_file in link_files:
            try:
                os.symlink(Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run / link_file,
                           link_dir / link_file)
            except:
                continue
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run,
            '-per-session',
            '-fsd', 'bold',
            '-fstem', f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln',
            '-fmcstem', f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc',
            '-nolog']
        sh.mc_sess(*shargs, _out=sys.stdout)
        ori_path = Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run
        try:
            shutil.move(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz')
            shutil.move(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.mat.aff12.1D',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.mat.aff12.1D')
            shutil.move(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz.mclog',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz.mclog')
            shutil.move(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.mcdat',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.mcdat')
            shutil.move(link_dir / 'mcextreg', ori_path / 'mcextreg')
            shutil.move(link_dir / 'mcdat2extreg.log', ori_path / 'mcdat2extreg.log')
        except:
            pass
        shutil.rmtree(ori_path / self.inputs.subject_id)

    def _run_interface(self, runtime):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        runs = sorted(
            [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        # runs = ['001', '002', '003', '004', '005', '006', '007', '008']
        multipool_run(self.cmd, runs, Multi_Num=8)

        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj = self.inputs.subject_id.split('-')[1]
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        subj_bold_dir = Path(preprocess_dir) / f'{self.inputs.subject_id}' / 'bold'
        deepprep_subj_path = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        deepprep_subj_path = Path(deepprep_subj_path)
        for idx, bids_bold in enumerate(bids_bolds):
            run = f"{idx + 1:03}"
            entities = dict(bids_bold.entities)
            file_prefix = Path(bids_bold.path).name.replace('.nii.gz', '')
            if 'session' in entities:
                subj_func_path = deepprep_subj_path / f"ses-{entities['session']}" / 'func'
            else:
                subj_func_path = deepprep_subj_path / 'func'
            src_mc_file = subj_bold_dir / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
            dst_mc_file = subj_func_path / f'{file_prefix}_mc.nii.gz'
            shutil.copy(src_mc_file, dst_mc_file)

        self.check_output(runs)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self):
        return []


class StcInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject_id', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class StcOutputSpec(TraitedSpec):
    data_path = Directory(exists=True, desc="data path")
    subject_id = Str(exists=True, desc="subject id")


class Stc(BaseInterface):
    input_spec = StcInputSpec
    output_spec = StcOutputSpec

    time = 214 / 60  # 运行时间：分钟
    cpu = 6  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(Stc, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task
        for run in runs:
            Stc_output_files = [f'{sub}_bld_rest_reorient_skip_faln.nii.gz']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(Stc_output_files) <= set(output_list)
            if not check_result:
                return FileExistsError

    def cmd(self, run):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        link_dir = Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run / self.inputs.subject_id / 'bold' / run
        if not link_dir.exists():
            link_dir.mkdir(parents=True, exist_ok=True)
        link_files = os.listdir(Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run)
        link_files.remove(self.inputs.subject_id)
        try:
            os.symlink(Path(preprocess_dir) / self.inputs.subject_id / 'bold' / 'template.nii.gz',
                       Path(
                           preprocess_dir) / self.inputs.subject_id / 'bold' / run / self.inputs.subject_id / 'bold' / 'template.nii.gz')
        except:
            pass
        for link_file in link_files:
            try:
                os.symlink(Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run / link_file,
                           link_dir / link_file)
            except:
                continue
        input_fname = f'{self.inputs.subject_id}_bld_rest_reorient_skip'
        output_fname = f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln'
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run,
            '-fsd', 'bold',
            '-so', 'odd',
            '-ngroups', 1,
            '-i', input_fname,
            '-o', output_fname,
            '-nolog']
        sh.stc_sess(*shargs, _out=sys.stdout)
        ori_path = Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run
        try:
            shutil.move(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz')
            shutil.move(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz.log',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz.log')
            shutil.move(link_dir / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz.log.bak',
                        ori_path / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln.nii.gz.log.bak')
        except:
            pass
        shutil.rmtree(ori_path / self.inputs.subject_id)

    def _run_interface(self, runtime):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        runs = sorted([d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if
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

        self.check_output(runs)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self):
        from interface.create_node_bold import create_MkTemplate_node
        node = create_MkTemplate_node(self.inputs.subject_id,
                                      self.inputs.task,
                                      self.inputs.atlas_type,
                                      self.inputs.preprocess_method)

        return node


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

    time = 382 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(Register, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task
        for run in runs:
            Register_output_files = [f'{sub}_bld_rest_reorient_skip_faln_mc.register.dat',
                                     f'{sub}_bld_rest_reorient_skip_faln_mc.register.dat.mincost',
                                     f'{sub}_bld_rest_reorient_skip_faln_mc.register.dat.param',
                                     f'{sub}_bld_rest_reorient_skip_faln_mc.register.dat.sum']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(Register_output_files) <= set(output_list)
            if not check_result:
                return FileExistsError

    def cmd(self, run):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        mov = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
        reg = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.register.dat'
        shargs = [
            '--bold',
            '--s', self.inputs.subject_id,
            '--mov', mov,
            '--reg', reg]
        sh.bbregister(*shargs, _out=sys.stdout)

    def _run_interface(self, runtime):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        runs = sorted(
            [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        multipool_run(self.cmd, runs, Multi_Num=8)

        layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
        subj = self.inputs.subject_id.split('-')[1]
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
        subj_bold_dir = Path(preprocess_dir) / f'{self.inputs.subject_id}' / 'bold'
        deepprep_subj_path = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        deepprep_subj_path = Path(deepprep_subj_path)
        for idx, bids_bold in enumerate(bids_bolds):
            run = f"{idx + 1:03}"
            entities = dict(bids_bold.entities)
            file_prefix = Path(bids_bold.path).name.replace('.nii.gz', '')
            if 'session' in entities:
                subj_func_path = deepprep_subj_path / f"ses-{entities['session']}" / 'func'
            else:
                subj_func_path = deepprep_subj_path / 'func'
            src_reg_file = subj_bold_dir / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.register.dat'
            dst_reg_file = subj_func_path / f'{file_prefix}_bbregister.register.dat'
            shutil.copy(src_reg_file, dst_reg_file)

        self.check_output(runs)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self):
        return []


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

    time = 18 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 2.7  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(MkBrainmask, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task
        for run in runs:
            MkBrainmask_output_files = [f'{sub}.brainmask.bin.nii.gz', f'{sub}.brainmask.nii.gz',
                                        f'{sub}.func.aseg.nii', f'{sub}.func.ventricles.nii.gz', f'{sub}.func.wm.nii.gz']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(MkBrainmask_output_files) <= set(output_list)
            if not check_result:
                return FileExistsError

    def cmd(self, run):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        seg = Path(
            self.inputs.subjects_dir) / self.inputs.subject_id / 'mri/aparc+aseg.mgz'  # TODO 这个应该由structure_workflow传进来
        mov = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
        reg = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.register.dat'
        func = Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.func.aseg.nii'
        wm = Path(preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.func.wm.nii.gz'
        vent = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.func.ventricles.nii.gz'
        targ = Path(
            self.inputs.subjects_dir) / self.inputs.subject_id / 'mri/brainmask.mgz'  # TODO 这个应该由structure_workflow传进来
        mask = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.brainmask.nii.gz'
        binmask = Path(
            preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}.brainmask.bin.nii.gz'
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
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        runs = sorted(
            [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])

        multipool_run(self.cmd, runs, Multi_Num=8)

        self.check_output(runs)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self):
        from interface.create_node_bold import create_RestGauss_node, create_VxmRegNormMNI152_node
        if self.inputs.task == 'rest':
            node = create_RestGauss_node(self.inputs.subject_id,
                                         self.inputs.task,
                                         self.inputs.atlas_type,
                                         self.inputs.preprocess_method)
        else:
            node = create_VxmRegNormMNI152_node(self.inputs.subject_id,
                                                self.inputs.task,
                                                self.inputs.atlas_type,
                                                self.inputs.preprocess_method)

        return node


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

    time = 15 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 14  # 最大cpu占用：个
    gpu = 2703  # 最大gpu占用：MB

    def __init__(self):
        super(VxmRegistraion, self).__init__()

    def check_output(self):
        sub = self.inputs.subject_id
        VxmRegistraion_output_files = ['warped.nii.gz', 'warp.nii.gz', 'vxminput.npz', f'{sub}_warped.nii.gz',
                                       f'{sub}_warp.nii.gz', f'{sub}_affine.mat']
        output_list = os.listdir(Path(self.inputs.derivative_deepprep_path) / sub / 'tmp')
        check_result = set(VxmRegistraion_output_files) <= set(output_list)
        if not check_result:
            return FileExistsError

    def _run_interface(self, runtime):
        # import tensorflow as tf
        # import ants
        # import shutil
        # import deepprep_pipeline.voxelmorph as vxm
        subject_id = self.inputs.subject_id
        deepprep_subj_path = Path(self.inputs.derivative_deepprep_path) / subject_id

        norm = Path(self.inputs.subjects_dir) / subject_id / 'mri' / 'norm.mgz'
        trf_path = deepprep_subj_path / 'tmp' / f'{subject_id}_affine.mat'
        warp_path = deepprep_subj_path / 'tmp' / f'{subject_id}_warp.nii.gz'
        warped_path = deepprep_subj_path / 'tmp' / f'{subject_id}_warped.nii.gz'
        vxm_warp = deepprep_subj_path / 'tmp' / 'warp.nii.gz'
        vxm_warped_path = deepprep_subj_path / 'tmp' / 'warped.nii.gz'
        npz = deepprep_subj_path / 'tmp' / 'vxminput.npz'
        Path(deepprep_subj_path).mkdir(exist_ok=True)

        tmpdir = Path(deepprep_subj_path) / 'tmp'
        tmpdir.mkdir(exist_ok=True)

        # atlas and model
        vxm_model_path = Path(self.inputs.vxm_model_path)
        atlas_type = self.inputs.atlas_type
        model_file = self.inputs.model_file  # vxm_model_path / atlas_type / f'model.h5'
        atlas_path = vxm_model_path / atlas_type / f'{atlas_type}_brain.nii.gz'
        vxm_atlas_path = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.nii.gz'
        vxm_atlas_npz_path = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.npz'
        vxm2atlas_trf = vxm_model_path / atlas_type / f'{atlas_type}_vxm2atlas.mat'

        norm = ants.image_read(str(norm))
        vxm_atlas = ants.image_read(str(vxm_atlas_path))
        tx = ants.registration(fixed=vxm_atlas, moving=norm, type_of_transform='Affine')
        trf = ants.read_transform(tx['fwdtransforms'][0])
        ants.write_transform(trf, str(trf_path))
        affined = tx['warpedmovout']
        vol = affined.numpy() / 255.0
        np.savez_compressed(npz, vol=vol)

        # voxelmorph
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
        moving = vxm.py.utils.load_volfile(str(npz), add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed, fixed_affine = vxm.py.utils.load_volfile(str(vxm_atlas_npz_path), add_batch_axis=True,
                                                        add_feat_axis=add_feat_axis,
                                                        ret_affine=True)
        vxm_atlas_nib = nib.load(str(vxm_atlas_path))
        fixed_affine = vxm_atlas_nib.affine.copy()
        inshape = moving.shape[1:-1]
        nb_feats = moving.shape[-1]

        with tf.device(device):
            # load model and predict
            warp = vxm.networks.VxmDense.load(model_file).register(moving, fixed)
            # warp = vxm.networks.VxmDenseSemiSupervisedSeg.load(args.model).register(moving, fixed)
            moving = affined.numpy()[np.newaxis, ..., np.newaxis]
            moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

        # save warp
        vxm.py.utils.save_volfile(warp.squeeze(), str(vxm_warp), fixed_affine)
        shutil.copy(vxm_warp, warp_path)

        # save moved image
        vxm.py.utils.save_volfile(moved.squeeze(), str(vxm_warped_path), fixed_affine)

        # affine to atlas
        atlas = ants.image_read(str(atlas_path))
        vxm_warped = ants.image_read(str(vxm_warped_path))
        warped = ants.apply_transforms(fixed=atlas, moving=vxm_warped, transformlist=[str(vxm2atlas_trf)])
        Path(warped_path).parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(warped, str(warped_path))

        self.check_output()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subjects_dir"] = self.inputs.subjects_dir
        return outputs

    def create_sub_node(self):
        from interface.create_node_bold import create_BoldSkipReorient_node
        node = create_BoldSkipReorient_node(self.inputs.subject_id, self.inputs.task, self.inputs.atlas_type,
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
            output_list = os.listdir(Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
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
        from interface.create_node_bold import create_RestBandpass_node
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
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp'/ f'task-{task}' / sub / 'bold' / run)
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
        from interface.create_node_bold import create_RestRegression_node
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

    def check_output(self,runs):
        sub = self.inputs.subject_id
        task = self.inputs.task

        for run in runs:
            RestRegression_bold_output_files = [f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_snr.nii.gz',
                                                f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_sd1.nii.gz',
                                                f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz']
            bold_output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp'/ f'task-{task}' / sub / 'bold' / run)
            check_bold_result = set(RestRegression_bold_output_files) <= set(bold_output_list)

            RestRegression_fcmri_output_files = [f"{sub}_bld{run}_mov_regressor.dat", f"{sub}_bld{run}_pca_regressor_dt.dat",
                                                 f"{sub}_bld{run}_regressors.dat", f"{sub}_bld{run}_ventricles_regressor_dt.dat",
                                                 f"{sub}_bld{run}_vent_wm_dt.dat", f"{sub}_bld{run}_WB_regressor_dt.dat",
                                                 f"{sub}_bld{run}_wm_regressor_dt.dat"]
            fcmri_output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp'/ f'task-{task}' / sub / 'fcmri')
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
        from interface.create_node_bold import create_VxmRegNormMNI152_node
        node = create_VxmRegNormMNI152_node(self.inputs.subject_id,
                                            self.inputs.task,
                                            self.inputs.atlas_type,
                                            self.inputs.preprocess_method)

        return node


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

    time = 503 / 60  # 运行时间：分钟
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(VxmRegNormMNI152, self).__init__()

    def check_output(self, subj_func_path, file_prefix):
        sub = self.inputs.subject_id

        VxmRegNormMNI152_output_files = [f'{sub}_norm_2mm.nii.gz', f'{sub}_MNI2mm.nii.gz', f'{file_prefix}_bbregister.register.dat']
        output_list = os.listdir(subj_func_path)
        check_result = set(VxmRegNormMNI152_output_files) <= set(output_list)
        if not check_result:
            return FileExistsError

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
        c3d_affine_tool = Path(self.inputs.resource_dir) / 'c3d_affine_tool'
        cmd = f'{c3d_affine_tool} -ref {ref_file} -src {first_frame_file} {fsltrf_file} -fsl2ras -oitk {tfm_file}'
        os.system(cmd)
        trf_sitk = sitk.ReadTransform(tfm_file)
        trf = ants.new_ants_transform()
        trf.set_parameters(trf_sitk.GetParameters())
        trf.set_fixed_parameters(trf_sitk.GetFixedParameters())
        ants.write_transform(trf, trf_file)

    def native_bold_to_T1_2mm_ants(self, residual_file, subject_id, subj_t1_file, reg_file, save_file, preprocess_dir,
                                   verbose=False):
        subj_t1_2mm_file = os.path.join(os.path.split(save_file)[0], f'{subject_id}_norm_2mm.nii.gz')
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

    def vxm_warp_bold_2mm(self, resid_t1, affine_file, warp_file, warped_file, verbose=True):
        import voxelmorph as vxm
        import tensorflow as tf
        import time

        vxm_model_path = Path(self.inputs.vxm_model_path)
        atlas_type = self.inputs.atlas_type

        atlas_file = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.nii.gz'
        MNI152_2mm_file = vxm_model_path / atlas_type / f'{atlas_type}_brain.nii.gz'
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
        tic = time.time()
        # affined_img = ants.apply_transforms(atlas, bold_img, fwdtrf_MNI152_2mm, imagetype=3)
        affined_np = ants.apply_transforms(atlas, bold_img, fwdtrf_MNI152_2mm, imagetype=3).numpy()
        # print(sys.getrefcount(affined_img))
        # del affined_img
        # toc = time.time()
        # print(toc - tic)
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
                # print(f'batch: {idx}')
            del transform
            del tf_dataset
            del moved
            del moved_data
        # toc = time.time()
        # print(toc - tic)

        # affine to MNI152
        # tic = time.time()
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
        # toc = time.time()
        # print(toc - tic)

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
        subj = self.inputs.subject_id.split('-')[1]
        derivative_deepprep_path = Path(self.inputs.derivative_deepprep_path)
        deepprep_subj_path = derivative_deepprep_path / self.inputs.subject_id
        norm_path = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri' / 'norm.mgz'
        preprocess_dir = derivative_deepprep_path / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        if self.inputs.task is None:
            bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
        else:
            bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')

        subject_id = self.inputs.subject_id
        bids_entities = []
        bids_path = []
        for bids_bold in bids_bolds:
            entities = dict(bids_bold.entities)
            file_prefix = Path(bids_bold.path).name.replace('.nii.gz', '')
            if 'session' in entities:
                ses = entities['session']
                subj_func_path = Path(deepprep_subj_path) / f'ses-{ses}' / 'func'
            else:
                subj_func_path = Path(deepprep_subj_path) / 'func'
            if self.inputs.preprocess_method == 'rest':
                bold_file = subj_func_path / f'{file_prefix}_resid.nii.gz'
            else:
                bold_file = subj_func_path / f'{file_prefix}_mc.nii.gz'

            reg_file = subj_func_path / f'{file_prefix}_bbregister.register.dat'
            bold_t1_file = subj_func_path / f'{subject_id}_native_t1_2mm.nii.gz'
            bold_t1_out = self.native_bold_to_T1_2mm_ants(bold_file, subject_id, norm_path, reg_file,
                                                          bold_t1_file, preprocess_dir, verbose=False)

            warp_file = subj_func_path / f'{subject_id}_warp.nii.gz'
            affine_file = subj_func_path / f'{subject_id}_affine.mat'
            warped_file = subj_func_path / f'{subject_id}_MNI2mm.nii.gz'
            warped_img = self.vxm_warp_bold_2mm(bold_t1_out, affine_file, warp_file, warped_file, verbose=True)

            self.check_output(subj_func_path,file_prefix)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["subject_id"] = self.inputs.subject_id
        outputs["data_path"] = self.inputs.data_path

        return outputs

    def create_sub_node(self):
        from interface.create_node_bold import create_Smooth_node
        node = create_Smooth_node(self.inputs.subject_id,
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

        Smooth_output_files = [f'{file_prefix}_resid_MNI2mm_sm6.nii.gz'] # TODO MNI2mm 要不要优化
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


class MkTemplateInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject', mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    task = Str(exists=True, desc='task', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)


class MkTemplateOutputSpec(TraitedSpec):
    data_path = Directory(exists=True, desc="data path")
    subject_id = Str(exists=True, desc="subject id")


class MkTemplate(BaseInterface):
    input_spec = MkTemplateInputSpec
    output_spec = MkTemplateOutputSpec

    # time = 120 / 60  # 运行时间：分钟
    # cpu = 2  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB

    def __init__(self):
        super(MkTemplate, self).__init__()

    def check_output(self, runs):
        sub = self.inputs.subject_id
        task = self.inputs.task
        for run in runs:
            MkTemplate_output_files = ['template.nii.gz']
            output_list = os.listdir(
                Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
            check_result = set(MkTemplate_output_files) <= set(output_list)
            if not check_result:
                return FileExistsError

    def _run_interface(self, runtime):
        preprocess_dir = Path(
            self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
        shargs = [
            '-s', self.inputs.subject_id,
            '-d', preprocess_dir,
            '-fsd', 'bold',
            '-funcstem', f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln',
            '-nolog']
        sh.mktemplate_sess(*shargs, _out=sys.stdout)

        runs = sorted(
            [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
        self.check_output(runs)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self):
        from interface.create_node_bold import create_MotionCorrection_node
        node = create_MotionCorrection_node(self.inputs.subject_id,
                                            self.inputs.task,
                                            self.inputs.atlas_type,
                                            self.inputs.preprocess_method)

        return node
