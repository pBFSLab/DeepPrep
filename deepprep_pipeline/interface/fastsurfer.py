from pathlib import Path
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, Directory, traits_extension
from cmd import run_cmd_with_timing


class SegmentInputSpec(BaseInterfaceInputSpec):
    python_interpret = File(exists=True, mandatory=True, desc='the python interpret to use')
    in_file = File(exists=True, mandatory=True, desc='name of file to process. Default: mri/orig.mgz')
    out_file = File(mandatory=True,
                    desc='name under which segmentation will be saved. Default: mri/aparc.DKTatlas+aseg.deep.mgz. '
                         'If a separate subfolder is desired (e.g. FS conform, add it to the name: '
                         'mri/aparc.DKTatlas+aseg.deep.mgz)')  # Do not set exists=True !!
    conformed_file = File(desc='Name under which the conformed input image will be saved, in the same directory as '
                               'the segmentation (the input image is always conformed first, if it is not already '
                               'conformed). The original input image is saved in the output directory as '
                               '$id/mri/orig/001.mgz. Default: mri/conform.mgz.')
    eval_py = File(exists=True, mandatory=True, desc="FastSurfer segmentation script")
    network_sagittal_path = File(exists=True, mandatory=True, desc="path to pre-trained weights of sagittal network")
    network_coronal_path = File(exists=True, mandatory=True, desc="pre-trained weights of coronal network")
    network_axial_path = File(exists=True, mandatory=True, desc="pre-trained weights of axial network")


class SegmentOutputSpec(TraitedSpec):
    aseg_deep_file = File(exists=True, desc='the output seg image: mri/aparc.DKTatlas+aseg.deep.mgz')


class Segment(BaseInterface):
    input_spec = SegmentInputSpec
    output_spec = SegmentOutputSpec

    time = 24 / 60  # 运行时间：分钟
    cpu = 10  # 最大cpu占用：个
    gpu = 9756  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        if not traits_extension.isdefined(self.inputs.conformed_file):
            conformed_file = Path(self.inputs.out_file).parent / 'conformed.mgz'
        else:
            conformed_file = self.inputs.conformed_file
        cmd = f'{self.inputs.python_interpret} {self.inputs.eval_py} ' \
              f'--in_name {self.inputs.in_file} ' \
              f'--out_name {self.inputs.out_file} ' \
              f'--conformed_name {conformed_file} ' \
              '--order 1 ' \
              f'--network_sagittal_path {self.inputs.network_sagittal_path} ' \
              f'--network_coronal_path {self.inputs.network_coronal_path} ' \
              f'--network_axial_path {self.inputs.network_axial_path} ' \
              '--batch_size 1 --simple_run --run_viewagg_on check'
        run_cmd_with_timing(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['aseg_deep_file'] = self.inputs.out_file
        return outputs
