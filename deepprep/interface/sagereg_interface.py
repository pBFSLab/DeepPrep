from pathlib import Path
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, Directory, Str
from interface.run import run_cmd_with_timing, multipool


class SageRegInputSpec(BaseInterfaceInputSpec):
    python_interpret = File(exists=True, mandatory=True, desc='the python interpret to use')
    sagereg_py = File(exists=True, mandatory=True, desc="SageReg script")

    deepprep_home = Directory(exists=True, desc='DeepPrep HOME path', mandatory=True)
    subjects_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    freesurfer_home = Directory(exists=True, desc='FreeSurfer HOME path', mandatory=True)
    lh_sulc = File(exists=True, desc='surf/lh.sulc')
    rh_sulc = File(exists=True, desc='surf/rh.sulc')
    lh_curv = File(exists=True, desc='surf/lh.curv')
    rh_curv = File(exists=True, desc='surf/rh.curv')
    lh_sphere = File(exists=True, desc='surf/lh.sphere')
    rh_sphere = File(exists=True, desc='surf/rh.sphere')


class SageRegOutputSpec(TraitedSpec):
    lh_sphere_reg = File(exists=True, desc='the output seg image: surf/lh.sphere.reg')
    rh_sphere_reg = File(exists=True, desc='the output seg image: surf/rh.sphere.reg')


class SageReg(BaseInterface):
    input_spec = SageRegInputSpec
    output_spec = SageRegOutputSpec

    time = 5 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 3500  # 最大gpu占用：MB

    def cmd(self, hemi):
        subjects_dir = self.inputs.subjects_dir
        subject_id = self.inputs.subject_id
        model_path = Path(self.inputs.deepprep_home) / 'model' / 'SageReg' / 'model_files'
        cmd = f'{self.inputs.python_interpret} {self.inputs.sagereg_py} --sd {subjects_dir} --sid {subject_id} ' \
              f'--fsd {self.inputs.freesurfer_home} --hemi {hemi} --model_path {model_path}'
        run_cmd_with_timing(cmd)

    def _run_interface(self, runtime):
        multipool(self.cmd, Multi_Num=2)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        subjects_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id
        outputs['lh_sphere_reg'] = subjects_dir / 'surf' / f'lh.sphere.reg'
        outputs['rh_sphere_reg'] = subjects_dir / 'surf' / f'rh.sphere.reg'
        return outputs
