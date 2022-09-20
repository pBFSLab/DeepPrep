from pathlib import Path
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, Directory, Str
from run import run_cmd_with_timing, multipool


class FeatRegInputSpec(BaseInterfaceInputSpec):
    python_interpret = File(exists=True, mandatory=True, desc='the python interpret to use')
    featreg_py = File(exists=True, mandatory=True, desc="FeatReg script")

    subjects_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    freesurfer_home = Directory(exists=True, desc='FreeSurfer HOME path', mandatory=True)
    lh_sulc = File(exists=True, desc='surf/lh.sulc')
    rh_sulc = File(exists=True, desc='surf/rh.sulc')
    lh_curv = File(exists=True, desc='surf/lh.curv')
    rh_curv = File(exists=True, desc='surf/rh.curv')
    lh_sphere = File(exists=True, desc='surf/lh.sphere')
    rh_sphere = File(exists=True, desc='surf/rh.sphere')


class FeatRegOutputSpec(TraitedSpec):
    lh_sphere_reg = File(exists=True, desc='the output seg image: surf/lh.sphere.reg')
    rh_sphere_reg = File(exists=True, desc='the output seg image: surf/rh.sphere.reg')


class FeatReg(BaseInterface):
    input_spec = FeatRegInputSpec
    output_spec = FeatRegOutputSpec

    time = 5 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 3500  # 最大gpu占用：MB

    def cmd(self, hemi):
        subjects_dir = self.inputs.subjects_dir
        subject_id = self.inputs.subject_id
        cmd = f'{self.inputs.python_interpret} {self.inputs.featreg_py} --sd {subjects_dir} --sid {subject_id} ' \
              f'--fsd {self.inputs.freesurfer_home} --hemi {hemi}'
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
