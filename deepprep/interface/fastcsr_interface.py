# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : NingAnMe <ninganme0317@gmail.com>
# @Author : Cong Lin       @Email : lincong <lincong8722@gmail.com>
# @Author : Youjia Zhang   @Email : youjia <ireneyou33@gmail.com>
# @Author : Zhenyu Sun     @Email : Kid-sunzhenyu <sun25939789@gmail.com>

import os
from pathlib import Path
from multiprocessing import Pool, Process, Lock
import logging
import subprocess
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, Directory, Str

from interface.run import run_cmd_with_timing


def log_msg(msg, lock, level):
    if level == logging.INFO:
        if lock is not None:
            with lock:
                logging.info(msg)
        else:
            logging.info(msg)
    elif level == logging.ERROR:
        if lock is not None:
            with lock:
                logging.error(msg)
        else:
            logging.error(msg)


class FastCSRInputSpec(BaseInterfaceInputSpec):
    python_interpret = File(exists=True, mandatory=True, desc='the python interpret to use')
    fastcsr_py = File(exists=True, mandatory=True, desc="FastCSR script")

    subjects_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    orig_file = File(exists=True, desc='mri/orig.mgz')
    filled_file = File(exists=True, desc='mri/filled.mgz')
    aseg_presurf_file = File(exists=True, desc='mri/aseg.presurf.mgz')
    brainmask_file = File(exists=True, desc='mri/brainmask.mgz')
    wm_file = File(exists=True, desc='mri/wm.mgz')
    brain_finalsurfs_file = File(exists=True, desc='mri/brain.finalsurfs.mgz')


class FastCSROutputSpec(TraitedSpec):
    lh_orig_file = File(exists=True, desc='the output seg image: surf/lh.orig')
    rh_orig_file = File(exists=True, desc='the output seg image: surf/rh.orig')
    lh_orig_premesh_file = File(exists=True, desc='the output seg image: surf/lh.orig.premesh')
    rh_orig_premesh_file = File(exists=True, desc='the output seg image: surf/rh.orig.premesh')


class FastCSR(BaseInterface):
    input_spec = FastCSRInputSpec
    output_spec = FastCSROutputSpec

    time = 120 / 60  # 运行时间：分钟
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        subjects_dir = self.inputs.subjects_dir
        subject_id = self.inputs.subject_id
        cmd = f'{self.inputs.python_interpret} {self.inputs.fastcsr_py} --sd {subjects_dir} --sid {subject_id} ' \
              f'--optimizing_surface off --parallel_scheduling on'
        run_cmd_with_timing(cmd)
        for hemi in ['lh', 'rh']:
            orig = Path(subjects_dir) / subject_id / 'surf' / f'{hemi}.orig'
            orig_premesh = Path(subjects_dir) / subject_id / 'surf' / f'{hemi}.orig.premesh'
            cmd = f'cp {orig} {orig_premesh}'
            run_cmd_with_timing(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        subject_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id
        outputs['lh_orig_file'] = subject_dir / 'surf' / 'lh.orig'
        outputs['rh_orig_file'] = subject_dir / 'surf' / 'rh.orig'
        outputs['lh_orig_premesh_file'] = subject_dir / 'surf' / 'lh.orig.premesh'
        outputs['rh_orig_premesh_file'] = subject_dir / 'surf' / 'rh.orig.premesh'
        return outputs


class FastCSRModelInputSpec(BaseInterfaceInputSpec):
    python_interpret = File(exists=True, mandatory=True, desc='the python interpret to use')
    fastcsr_py = File(exists=True, mandatory=True, desc="FastCSR script")

    subjects_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    orig_file = File(exists=True, desc='mri/orig.mgz')
    filled_file = File(exists=True, desc='mri/filled.mgz')
    aseg_presurf_file = File(exists=True, desc='mri/aseg.presurf.mgz')
    brainmask_file = File(exists=True, desc='mri/brainmask.mgz')
    wm_file = File(exists=True, desc='mri/wm.mgz')
    brain_finalsurfs_file = File(exists=True, desc='mri/brain.finalsurfs.mgz')


class FastCSRModelOutputSpec(TraitedSpec):
    lh_levelset_file = File(exists=True, desc='the output seg image: surf/lh_levelset.nii.gz')
    rh_levelset_file = File(exists=True, desc='the output seg image: surf/rh_levelset.nii.gz')


class FastCSRModel(BaseInterface):
    input_spec = FastCSRModelInputSpec
    output_spec = FastCSRModelOutputSpec

    time = 120 / 60  # 运行时间：分钟
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        fastcsr_path = Path(self.inputs.fastcsr_py).parent

        logging.info('-------------------Generate mri/brain.finalsurfs.mgz file-------------------------')
        if not ((subjects_dir / subject_id / 'mri/lh_levelset.nii.gz').exists() and
                (subjects_dir / subject_id / 'mri/rh_levelset.nii.gz').exists()):
            cmd_pool = list()
            cmd = f"{self.inputs.python_interpret} {fastcsr_path / 'fastcsr_model_infer.py'} --fastcsr_subjects_dir {subjects_dir} --subj {subject_id} --hemi lh".split()
            cmd_pool.append(cmd)
            cmd = f"{self.inputs.python_interpret} {fastcsr_path / 'fastcsr_model_infer.py'} --fastcsr_subjects_dir {subjects_dir} --subj {subject_id} --hemi rh".split()
            cmd_pool.append(cmd)
            lh_process = subprocess.Popen(cmd_pool[0], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            rh_process = subprocess.Popen(cmd_pool[1], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            lh_retcode = lh_process.wait()
            rh_retcode = rh_process.wait()
            if lh_retcode != 0 or rh_retcode != 0:
                msg = 'Levelset regression model inference failed.'
                log_msg(msg, None, logging.ERROR)
                exit(-1)
            msg = 'Levelset model regression inference completed.'
            log_msg(msg, None, logging.INFO)
        # logging.info('-------------------Generate mri/brain.finalsurfs.mgz file-------------------------')
        # if not (subjects_dir / subject_id / 'mri' / 'brain.finalsurfs.mgz').exists():
        #     cmd = f"{self.inputs.python_interpret} {fastcsr_path / 'brain_finalsurfs_model_infer.py'} --fastcsr_subjects_dir {subjects_dir} --subj {subject_id}".split()
        #     ret = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        #     if ret.returncode == 0:
        #         msg = 'Brain_finalsurfs regression model inference completed.'
        #         log_msg(msg, None, logging.INFO)
        #     else:
        #         msg = 'Brain_finalsurfs regression model inference failed.'
        #         log_msg(msg, None, logging.ERROR)
        #         exit(-1)
        # else:
        #     logging.info('-------------------Exists mri/brain.finalsurfs.mgz file-------------------------')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        subject_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id
        outputs['lh_levelset_file'] = subject_dir / 'mri' / 'lh_levelset.nii.gz'
        outputs['rh_levelset_file'] = subject_dir / 'mri' / 'rh_levelset.nii.gz'
        return outputs


class FastCSRSurfaceInputSpec(BaseInterfaceInputSpec):
    python_interpret = File(exists=True, mandatory=True, desc='the python interpret to use')
    fastcsr_py = File(exists=True, mandatory=True, desc="FastCSR script")

    subjects_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    lh_levelset_file = File(exists=True, desc='mri/lh_levelset.nii.gz')
    rh_levelset_file = File(exists=True, desc='mri/rh_levelset.nii.gz')


class FastCSRSurfaceOutputSpec(TraitedSpec):
    lh_orig_file = File(exists=True, desc='the output seg image: surf/lh.orig')
    rh_orig_file = File(exists=True, desc='the output seg image: surf/rh.orig')
    lh_orig_premesh_file = File(exists=True, desc='the output seg image: surf/lh.orig.premesh')
    rh_orig_premesh_file = File(exists=True, desc='the output seg image: surf/rh.orig.premesh')


class FastCSRSurface(BaseInterface):
    input_spec = FastCSRSurfaceInputSpec
    output_spec = FastCSRSurfaceOutputSpec

    time = 120 / 60  # 运行时间：分钟
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        fastcsr_path = Path(self.inputs.fastcsr_py).parent

        surfix = 'orig'

        cmd_pool = list()
        cmd = f"{self.inputs.python_interpret} {fastcsr_path / 'levelset2surf.py'} --fastcsr_subjects_dir {subjects_dir} --subj {subject_id} --hemi lh --suffix {surfix}"
        # os.system(cmd)
        cmd_pool.append([cmd])
        cmd = f"{self.inputs.python_interpret} {fastcsr_path / 'levelset2surf.py'} --fastcsr_subjects_dir {subjects_dir} --subj {subject_id} --hemi rh --suffix {surfix}"
        # os.system(cmd)
        cmd_pool.append([cmd])
        pool = Pool(2)
        pool.starmap(run_cmd_with_timing, cmd_pool)
        pool.close()
        pool.join()
        msg = 'Surface generation completed.'
        log_msg(msg, None, logging.INFO)

        for hemi in ['lh', 'rh']:
            orig = subjects_dir / subject_id / 'surf' / f'{hemi}.orig'
            orig_premesh = subjects_dir / subject_id / 'surf' / f'{hemi}.orig.premesh'
            cmd = f'cp {orig} {orig_premesh}'
            run_cmd_with_timing(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        subject_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id
        outputs['lh_orig_file'] = subject_dir / 'surf' / 'lh.orig'
        outputs['rh_orig_file'] = subject_dir / 'surf' / 'rh.orig'
        outputs['lh_orig_premesh_file'] = subject_dir / 'surf' / 'lh.orig.premesh'
        outputs['rh_orig_premesh_file'] = subject_dir / 'surf' / 'rh.orig.premesh'
        return outputs
