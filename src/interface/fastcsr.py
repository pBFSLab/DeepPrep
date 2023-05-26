from pathlib import Path
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, Directory, Str

from run import run_cmd_with_timing


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
        subjects_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id
        outputs['lh_orig_file'] = subjects_dir / 'surf' / 'lh.orig'
        outputs['rh_orig_file'] = subjects_dir / 'surf' / 'rh.orig'
        outputs['lh_orig_premesh_file'] = subjects_dir / 'surf' / 'lh.orig.premesh'
        outputs['rh_orig_premesh_file'] = subjects_dir / 'surf' / 'rh.orig.premesh'
        return outputs
