from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, File, TraitedSpec, Directory, Str
from deepprep.interface.run import multipool_run, multipool_BidsBolds, multipool_BidsBolds_2, Pool
import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import bids
import os
import ants
import shutil

# from deepprep.app.filters.filters import bandpass_nifti

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

    def create_sub_node(self):
        from interface.create_node_bold_new import create_StcMc_node
        node = create_StcMc_node(self.inputs.subject_id,
                                 self.inputs.task,
                                 self.inputs.atlas_type,
                                 self.inputs.preprocess_method,
                                 self.inputs.settings)

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

    def create_sub_node(self):
        if self.bold_only == 'True':
            from interface.create_node_bold_new import create_Register_node
            return create_Register_node(self.inputs.subject_id,
                                        self.inputs.task,
                                        self.inputs.atlas_type,
                                        self.inputs.preprocess_method,
                                        self.inputs.settings)
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

    def create_sub_node(self):
        from interface.create_node_bold_new import create_Mkbrainmask_node
        node = create_Mkbrainmask_node(self.inputs.subject_id,
                                       self.inputs.task,
                                       self.inputs.atlas_type,
                                       self.inputs.preprocess_method,
                                       self.inputs.settings)

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

    def create_sub_node(self):
        if self.bold_only == 'True':
            from interface.create_node_bold_new import create_VxmRegistraion_node
            node = create_VxmRegistraion_node(self.inputs.subject_id,
                                                self.inputs.task,
                                                self.inputs.atlas_type,
                                                self.inputs.preprocess_method,
                                                self.inputs.settings)
        else:
            from interface.create_node_bold_new import create_VxmRegNormMNI152_node
            node = create_VxmRegNormMNI152_node(self.inputs.subject_id,
                                                self.inputs.task,
                                                self.inputs.atlas_type,
                                                self.inputs.preprocess_method,
                                                self.inputs.settings)

        return node


# class RestGaussInputSpec(BaseInterfaceInputSpec):
#     subject_id = Str(exists=True, desc="subject id", mandatory=True)
#     subjects_dir = Directory(exists=True, desc='subjects_dir', mandatory=True)
#     data_path = Directory(exists=True, desc='data path', mandatory=True)
#     task = Str(exists=True, desc="task", mandatory=True)
#     derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
#     preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
#     atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)
#
#
# class RestGaussOutputSpec(TraitedSpec):
#     subject_id = Str(exists=True, desc="subject id")
#
#
# class RestGauss(BaseInterface):
#     input_spec = RestGaussInputSpec
#     output_spec = RestGaussOutputSpec
#
#     time = 11 / 60  # 运行时间：分钟
#     cpu = 0  # 最大cpu占用：个
#     gpu = 0  # 最大gpu占用：MB
#
#     def __init__(self):
#         super(RestGauss, self).__init__()
#
#     def check_output(self, runs):
#         sub = self.inputs.subject_id
#         task = self.inputs.task
#
#         for run in runs:
#             RestGauss_output_files = [f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz']
#             output_list = os.listdir(
#                 Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
#             check_result = set(RestGauss_output_files) <= set(output_list)
#             if not check_result:
#                 return FileExistsError
#
#     def cmd(self, run):
#         from app.filters.filters import gauss_nifti
#
#         preprocess_dir = Path(
#             self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
#         mc = Path(
#             preprocess_dir) / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
#         fcmri_dir = Path(preprocess_dir) / self.inputs.subject_id / 'fcmri'
#         Path(fcmri_dir).mkdir(parents=True, exist_ok=True)
#         gauss_nifti(str(mc), 1000000000)
#
#     def _run_interface(self, runtime):
#         preprocess_dir = Path(
#             self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
#         runs = sorted(
#             [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
#         multipool_run(self.cmd, runs, Multi_Num=8)
#
#         self.check_output(runs)
#
#         return runtime
#
#     def _list_outputs(self):
#         outputs = self._outputs().get()
#         outputs["subject_id"] = self.inputs.subject_id
#         return outputs
#
#     def create_sub_node(self):
#         from interface.create_node_bold_new import create_RestBandpass_node
#         node = create_RestBandpass_node(self.inputs.subject_id,
#                                         self.inputs.task,
#                                         self.inputs.atlas_type,
#                                         self.inputs.preprocess_method)
#
#         return node


# class RestBandpassInputSpec(BaseInterfaceInputSpec):
#     subject_id = Str(exists=True, desc='subject', mandatory=True)
#     task = Str(exists=True, desc='task', mandatory=True)
#     data_path = Directory(exists=True, desc='data path', mandatory=True)
#     derivative_deepprep_path = Directory(exists=True, desc='derivative_deepprep_path', mandatory=True)
#     preprocess_method = Str(exists=True, desc='preprocess method', mandatory=True)
#     atlas_type = Str(exists=True, desc='MNI152_T1_2mm', mandatory=True)
#
#
# class RestBandpassOutputSpec(TraitedSpec):
#     subject_id = Str(exists=True, desc='subject')
#     data_path = Directory(exists=True, desc='data path')
#
#
# class RestBandpass(BaseInterface):
#     input_spec = RestBandpassInputSpec
#     output_spec = RestBandpassOutputSpec
#
#     time = 120 / 60  # 运行时间：分钟
#     cpu = 2  # 最大cpu占用：个
#     gpu = 0  # 最大gpu占用：MB
#
#     def __init__(self):
#         super(RestBandpass, self).__init__()
#
#     def check_output(self, runs):
#         sub = self.inputs.subject_id
#         task = self.inputs.task
#
#         for run in runs:
#             RestBandpass_output_files = [f'{sub}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz']
#             output_list = os.listdir(
#                 Path(self.inputs.derivative_deepprep_path) / sub / 'tmp' / f'task-{task}' / sub / 'bold' / run)
#             check_result = set(RestBandpass_output_files) <= set(output_list)
#             if not check_result:
#                 return FileExistsError
#
#     def cmd(self, idx, bids_entities, bids_path):
#         preprocess_dir = Path(
#             self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
#         entities = dict(bids_entities)
#         if 'RepetitionTime' in entities:
#             TR = entities['RepetitionTime']
#         else:
#             bold = ants.image_read(bids_path)
#             TR = bold.spacing[3]
#         run = f'{idx + 1:03}'
#         gauss_path = preprocess_dir / self.inputs.subject_id / 'bold' / run / f'{self.inputs.subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz'
#         bandpass_nifti(str(gauss_path), TR)
#
#     def _run_interface(self, runtime):
#         preprocess_dir = Path(
#             self.inputs.derivative_deepprep_path) / self.inputs.subject_id / 'tmp' / f'task-{self.inputs.task}'
#         layout = bids.BIDSLayout(str(self.inputs.data_path), derivatives=False)
#         subj = self.inputs.subject_id.split('-')[1:]
#         if self.inputs.task is None:
#             bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
#         else:
#             bids_bolds = layout.get(subject=subj, task=self.inputs.task, suffix='bold', extension='.nii.gz')
#         all_idx = []
#         all_bids_entities = []
#         all_bids_path = []
#         for idx, bids_bold in enumerate(bids_bolds):
#             all_idx.append(idx)
#             all_bids_entities.append(bids_bold.entities)
#             all_bids_path.append(bids_bold.path)
#         multipool_BidsBolds(self.cmd, all_idx, all_bids_entities, all_bids_path, Multi_Num=8)
#         runs = sorted(
#             [d.name for d in (Path(preprocess_dir) / self.inputs.subject_id / 'bold').iterdir() if d.is_dir()])
#
#         self.check_output(runs)
#
#         return runtime
#
#     def _list_outputs(self):
#         outputs = self._outputs().get()
#         outputs["subject_id"] = self.inputs.subject_id
#         outputs["data_path"] = self.inputs.data_path
#
#         return outputs
#
#     def create_sub_node(self):
#         from interface.create_node_bold_new import create_RestRegression_node
#         node = create_RestRegression_node(self.inputs.subject_id,
#                                           self.inputs.task,
#                                           self.inputs.atlas_type,
#                                           self.inputs.preprocess_method)
#
#         return node
#

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
    #     from deepprep.app.surface_projection import surface_projection as sp
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
                                            self.inputs.preprocess_method,
                                            self.inputs.settings)

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
