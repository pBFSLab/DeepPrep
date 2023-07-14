# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : NingAnMe <ninganme0317@gmail.com>
# @Author : Cong Lin       @Email : lincong <lincong8722@gmail.com>
# @Author : Youjia Zhang   @Email : youjia <ireneyou33@gmail.com>
# @Author : Zhenyu Sun     @Email : Kid-sunzhenyu <sun25939789@gmail.com>

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, TraitedSpec, Directory, Str
from deepprep.interface.run import Pool
import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import bids
import os
import shutil

# TODO 现在每个BOLD 的 node 都是又重新获取的BIDS layer，然后获取数据，这样的操作太多其实是冗余的，应该在scheduler的时候获取一次就可以了
# TODO 将BOLD输出文件的命名放到filename_settings中，以 {bold_filename} 为基础的后缀 比如 _skip.nii.gz，不提供给用户修改接口


class BoldSkipReorientInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc="subject id", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)
    nskip_frame = Str(default_value="0", desc='skip n frames', mandatory=False)
    multiprocess = Str(default_value="1", desc="using for pool threads set", mandatory=False)


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

    def reorient_to_ras(self, input_path, output_path):
        img = nib.load(input_path)
        orig_ornt = nib.orientations.io_orientation(img.header.get_sform())
        RAS_ornt = nib.orientations.axcodes2ornt('RAS')
        if np.array_equal(orig_ornt, RAS_ornt) is True:
            print(f"{input_path} is already in RAS orientation. Copying to {output_path}.")
            shutil.copy(input_path, output_path)
        else:
            newimg = img.as_reoriented(orig_ornt)
            nib.save(newimg, output_path)
            print(f"Successfully reorient {input_path} to RAS orientation and saved to {output_path}.")

    def cmd(self, subj_func_dir: Path, bold: Path, nskip_frame: int):

        skip_bold = subj_func_dir / bold.name.replace('.nii.gz', '_skip.nii.gz')
        reorient_skip_bold = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient.nii.gz')

        # skip 0 frame
        if nskip_frame > 0:
            sh.mri_convert('-i', bold, '--nskip', nskip_frame, '-o', skip_bold, _out=sys.stdout)
        else:
            skip_bold = bold

        # reorient
        self.reorient_to_ras(skip_bold, reorient_skip_bold)

    def _run_interface(self, runtime):
        preprocess_dir = Path(self.inputs.derivative_deepprep_path) / self.inputs.subject_id
        # TODO 这里不应该每个node都调取一次BIDSLayout的命令，这样会导致程序运行变慢
        # TODO layout 应该是全局的，但是目前不知道layout是否可以序列化，能否支持multiprocessing

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
            args.append([subj_func_dir, bold_file, int(self.inputs.nskip_frame)])

        cpu_threads = int(self.inputs.multiprocess)
        if cpu_threads > 1:
            pool = Pool(cpu_threads)
            pool.starmap(self.cmd, args)
            pool.close()
            pool.join()
        else:
            for arg in args:
                self.cmd(*arg)

        self.check_output(subj_func_dir, bold_files)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id

        return outputs

    def create_sub_node(self, settings):
        from interface.create_node_bold import create_StcMc_node
        node = create_StcMc_node(self.inputs.subject_id,
                                 self.inputs.task,
                                 self.inputs.atlas_type,
                                 self.inputs.preprocess_method,
                                 settings)

        return node


class StcMcInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject_id', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)
    multiprocess = Str(default_value="1", desc="using for pool threads set", mandatory=False)


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
        filenames = [
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
                src_file = subj_func_dir / link_file
                dst_file = link_dir / link_file
                dst_file.symlink_to(src_file)
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

        """
        mktemplate-sess 会生成两个template
        1. 一个放到 bold/template.nii.gz，使用的是run 001的first frame，供mc-sess --per-session使用
        2. 一个放到 bold/run/template.nii.gz 使用的是每个run的mid frame，供mc-sess --per-run参数使用(default)
        """
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
            '-per-run',
            '-fsd', 'bold',
            '-fstem', faln_fname,
            '-fmcstem', mc_fname,
            '-nolog']
        sh.mc_sess(*shargs, _out=sys.stdout)

        ori_path = subj_func_dir
        try:
            # Stc
            DEBUG = False
            if DEBUG:
                shutil.move(link_dir / f'{faln_fname}.nii.gz',
                            ori_path / f'{faln_fname}.nii.gz')
                shutil.move(link_dir / f'{faln_fname}.nii.gz.log',
                            ori_path / f'{faln_fname}.nii.gz.log')
            else:
                (ori_path / f'{input_fname}.nii.gz').unlink(missing_ok=True)

            # Template reference for mc
            shutil.copyfile(link_dir / 'template.nii.gz',
                            ori_path / f'{faln_fname}_boldref.nii.gz')
            shutil.copyfile(link_dir / 'template.log',
                            ori_path / f'{faln_fname}_boldref.log')

            # Mc
            shutil.move(link_dir / f'{mc_fname}.nii.gz',
                        ori_path / f'{mc_fname}.nii.gz')

            shutil.move(link_dir / f'{mc_fname}.mcdat',
                        ori_path / f'{mc_fname}.mcdat')

            if DEBUG:
                shutil.move(link_dir / f'{mc_fname}.mat.aff12.1D',
                            ori_path / f'{mc_fname}.mat.aff12.1D')
                shutil.move(link_dir / f'{mc_fname}.nii.gz.mclog',
                            ori_path / f'{mc_fname}.nii.gz.mclog')
                shutil.move(link_dir / 'mcextreg',
                            ori_path / f'{mc_fname}.mcextreg')
                shutil.move(link_dir / 'mcdat2extreg.log',
                            ori_path / f'{mc_fname}.mcdat2extreg.log')
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

        cpu_threads = int(self.inputs.multiprocess)
        if cpu_threads > 1:
            pool = Pool(cpu_threads)
            pool.starmap(self.cmd, args)
            pool.close()
            pool.join()
        else:
            for arg in args:
                self.cmd(*arg)

        self.check_output(subj_func_dir, bold_files)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self, settings):
        if settings.BOLD_ONLY:
            from interface.create_node_bold import create_Register_node
            return create_Register_node(self.inputs.subject_id,
                                        self.inputs.task,
                                        self.inputs.atlas_type,
                                        self.inputs.preprocess_method,
                                        settings)
        return []


class RegisterInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject_id', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)
    multiprocess = Str(default_value="1", desc="using for pool threads set", mandatory=False)


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
        filenames = ['_skip_reorient_faln_mc_from_mc_to_fsnative_bbregister_rigid.dat',  # _skip_reorient_faln_mc_from_mc_to_fsnative_bbregister_rigid.dat
                     ]
        for bold in bolds:
            for filename in filenames:
                file_path = subj_func_dir / bold.name.replace('.nii.gz', filename)
                if not file_path.exists():
                    raise FileExistsError

    def cmd(self, subj_func_dir: Path, bold: Path):
        mov = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc.nii.gz')
        reg = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc_from_mc_to_fsnative_bbregister_rigid.dat')
        print(os.environ["SUBJECTS_DIR"])
        shargs = [
            '--bold',
            '--s', self.inputs.subject_id,
            '--mov', mov,
            '--reg', reg]
        sh.bbregister(*shargs, _out=sys.stdout)

        DEBUG = False
        if not DEBUG:
            (subj_func_dir / reg.name.replace('.dat', '.dat.mincost')).unlink(missing_ok=True)
            (subj_func_dir / reg.name.replace('.dat', '.dat.param')).unlink(missing_ok=True)
            (subj_func_dir / reg.name.replace('.dat', '.dat.sum')).unlink(missing_ok=True)

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

        cpu_threads = int(self.inputs.multiprocess)
        if cpu_threads > 1:
            pool = Pool(cpu_threads)
            pool.starmap(self.cmd, args)
            pool.close()
            pool.join()
        else:
            for arg in args:
                self.cmd(*arg)

        self.check_output(subj_func_dir, bold_files)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self, settings):
        from interface.create_node_bold import create_Mkbrainmask_node
        node = create_Mkbrainmask_node(self.inputs.subject_id,
                                       self.inputs.task,
                                       self.inputs.atlas_type,
                                       self.inputs.preprocess_method,
                                       settings)

        return node


class MkBrainmaskInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(exists=True, desc='subject_id', mandatory=True)
    task = Str(exists=True, desc="task", mandatory=True)
    data_path = Directory(exists=True, desc="data path", mandatory=True)
    subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
    derivative_deepprep_path = Directory(exists=True, desc="derivative_deepprep_path", mandatory=True)
    preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
    atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)
    multiprocess = Str(default_value="1", desc="using for pool threads set", mandatory=False)


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
        filenames = ['_skip_reorient_faln_mc.anat.aseg.nii.gz',
                     '_skip_reorient_faln_mc.anat.wm.nii.gz',
                     '_skip_reorient_faln_mc.anat.ventricles.nii.gz',
                     '_skip_reorient_faln_mc.anat.brainmask.nii.gz',
                     '_skip_reorient_faln_mc.anat.brainmask.bin.nii.gz',
                     ]
        for bold in bolds:
            for filename in filenames:
                file_path = subj_func_dir / bold.name.replace('.nii.gz', filename)
                if not file_path.exists():
                    raise FileExistsError(file_path)

    def cmd(self, subj_func_dir: Path, bold: Path):
        mov = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc.nii.gz')
        reg = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc_from_mc_to_fsnative_bbregister_rigid.dat')

        # project aparc+aseg to mc
        seg = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri' / 'aparc+aseg.mgz'  # Recon
        func = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc.anat.aseg.nii.gz')
        wm = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc.anat.wm.nii.gz')
        vent = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc.anat.ventricles.nii.gz')

        # project brainmask.mgz to mc
        targ = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri' / 'brainmask.mgz'  # Recon
        mask = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc.anat.brainmask.nii.gz')
        binmask = subj_func_dir / bold.name.replace('.nii.gz', '_skip_reorient_faln_mc.anat.brainmask.bin.nii.gz')

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

        cpu_threads = int(self.inputs.multiprocess)
        if cpu_threads > 1:
            pool = Pool(cpu_threads)
            pool.starmap(self.cmd, args)
            pool.close()
            pool.join()
        else:
            for arg in args:
                self.cmd(*arg)

        self.check_output(subj_func_dir, bold_files)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["data_path"] = self.inputs.data_path
        outputs["subject_id"] = self.inputs.subject_id
        return outputs

    def create_sub_node(self, settings):
        if settings.BOLD_ONLY:
            from interface.create_node_bold import create_VxmRegistraion_node
            node = create_VxmRegistraion_node(self.inputs.subject_id,
                                                self.inputs.task,
                                                self.inputs.atlas_type,
                                                self.inputs.preprocess_method,
                                                settings)
        else:
            from interface.create_node_bold import create_VxmRegNormMNI152_node
            node = create_VxmRegNormMNI152_node(self.inputs.subject_id,
                                                self.inputs.task,
                                                self.inputs.atlas_type,
                                                self.inputs.preprocess_method,
                                                settings)

        return node
