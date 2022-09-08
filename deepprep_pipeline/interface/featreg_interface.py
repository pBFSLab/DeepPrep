from pathlib import Path
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, Directory, Str
from run import run_cmd_with_timing


class FeatRegInputSpec(BaseInterfaceInputSpec):
    python_interpret = File(exists=True, mandatory=True, desc='the python interpret to use')
    featreg_py = File(exists=True, mandatory=True, desc="FeatReg script")

    subject_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    freesurfer_home = Directory(exists=True, desc='FreeSurfer HOME path', mandatory=True)
    hemisphere = Str(desc='hemisphere: lh or rh', mandatory=True)
    sulc_file = File(exists=True, desc='surf/?h.sulc')
    curv_file = File(exists=True, desc='surf/?h.curv')
    sphere_file = File(exists=True, desc='surf/?h.sphere')


class FeatRegOutputSpec(TraitedSpec):
    sphere_reg_file = File(exists=True, desc='the output seg image: surf/?h.sphere.reg')


class FeatReg(BaseInterface):
    input_spec = FeatRegInputSpec
    output_spec = FeatRegOutputSpec

    time = 5 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 3500  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        subject_dir = self.inputs.subject_dir
        subject_id = self.inputs.subject_id
        cmd = f'{self.inputs.python_interpret} {self.inputs.featreg_py} --sd {subject_dir} --sid {subject_id} ' \
              f'--fsd {self.inputs.freesurfer_home} --hemi {self.inputs.hemisphere}'
        run_cmd_with_timing(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        subject_dir = Path(self.inputs.subject_dir) / self.inputs.subject_id
        outputs['sphere_reg_file'] = subject_dir / 'surf' / f'{self.inputs.hemisphere}.sphere.reg'
        return outputs
