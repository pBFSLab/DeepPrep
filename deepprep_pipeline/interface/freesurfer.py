from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, Directory, Str
from nipype import Node, Workflow
from .cmd import run_cmd_with_timing
import os
import time
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bd', required=True, help='directory of bids type')
    parser.add_argument('--fsd', default=os.environ.get('FREESURFER_HOME'),
                        help='Output directory $FREESURFER_HOME (pass via environment or here)')
    parser.add_argument('--respective', default='off',
                        help='if on, while processing T1w file respectively')
    parser.add_argument('--rewrite', default='on',
                        help='set off, while not preprocess if subject recon path exist')
    parser.add_argument('--python', default='python3',
                        help='which python version to use')

    args = parser.parse_args()
    args_dict = vars(args)

    if args.fsd is None:
        args_dict['fsd'] = '/usr/local/freesurfer'
    args_dict['respective'] = True if args.respective == 'on' else False
    args_dict['rewrite'] = True if args.rewrite == 'on' else False

    return argparse.Namespace(**args_dict)


class BrainmaskInputSpec(BaseInterfaceInputSpec):
    subject_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    need_t1 = traits.BaseCBool(desc='bool', mandatory=True)
    nu_file = File(exists=True, desc="nu file", mandatory=True)
    mask_file = File(exists=True, desc="mask file", mandatory=True)

    T1_file = File(exists=False, desc="T1 file", mandatory=True)
    brainmask_file = File(exists=False, desc="brainmask file", mandatory=True)
    norm_file = File(exists=False, desc="norm file", mandatory=True)


class BrainmaskOutputSpec(TraitedSpec):
    brainmask_file = File(exists=True, desc="brainmask file")
    norm_file = File(exists=True, desc="norm file")
    T1_file = File(exists=False, desc="T1 file")


class Brainmask(BaseInterface):
    input_spec = BrainmaskInputSpec
    output_spec = BrainmaskOutputSpec

    def _run_interface(self, runtime):
        # create norm by masking nu 0.7s
        need_t1 = self.inputs.need_t1
        cmd = f'mri_mask {self.inputs.nu_file} {self.inputs.mask_file} {self.inputs.norm_file}'
        run_cmd_with_timing(cmd)

        if need_t1:  # T1.mgz 相比 orig.mgz 更平滑，对比度更高
            # create T1.mgz from nu 96.9s
            cmd = f'mri_normalize -g 1 -mprage {self.inputs.nu_file} {self.inputs.T1_file}'
            run_cmd_with_timing(cmd)

            # create brainmask by masking T1
            cmd = f'mri_mask {self.inputs.T1_file} {self.inputs.mask_file} {self.inputs.brainmask_file}'
            run_cmd_with_timing(cmd)
        else:
            cmd = f'ln -sf {self.inputs.norm_file} {self.inputs.brainmask_file}'
            run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["brainmask_file"] = self.inputs.brainmask_file
        outputs["norm_file"] = self.inputs.norm_file
        outputs["T1_file"] = self.inputs.T1_file

        return outputs


class N4BiasCorrectOutputSpec(TraitedSpec):
    nu_file = File(exists=True, desc='orig_nu.mgz')


class N4BiasCorrect(BaseInterface):
    output_spec = N4BiasCorrectOutputSpec

    def __init__(self, fastsurfer_home: Path):
        super(N4BiasCorrect, self).__init__()
        self.fastsurfer_bin = fastsurfer_home / "recon_surf"

    def _run_interface(self, runtime):
        # orig_nu nu correct
        py = self.fastsurfer_bin / "N4_bias_correct.py"
        cmd = f"{python} {py} --in {self.inputs.sub_mri_dir}/orig.mgz --out {self.inputs.sub_mri_dir}/orig_nu.mgz " \
              f"--mask {self.inputs.sub_mri_dir}/mask.mgz  --threads {self.inputs.threads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['nu_file'] = Path(f"{self.inputs.sub_mri_dir}/orig_nu.mgz")


class TalairachAndNuInputSpec(BaseInterfaceInputSpec):
    sub_mri_dir = File(exists=True, desc='subject path', mandatory=True)
    threads = traits.Int(desc='threads', mandatory=True)
    orig_nu_file = File(exists=True, desc='orig_nu.mgz path', mandatory=True)


class TalairachAndNuOutputSpec(TraitedSpec):
    talairach_file = File(exists=True, desc='talairach.lta')
    nu_file = File(exists=True, desc='nu.mgz')


class TalairachAndNu(BaseInterface):
    input_spec = TalairachAndNuInputSpec
    output_spec = TalairachAndNuOutputSpec

    def __init__(self):
        super(TalairachAndNu, self).__init__()

    def _run_interface(self, runtime):
        # talairach.xfm: compute talairach full head (25sec)
        cmd = f'cd {self.inputs.sub_mri_dir} && ' \
              f'talairach_avi --i {self.inputs.orig_nu_file} --xfm {self.inputs.sub_mri_dir}/transforms/talairach.auto.xfm'
        run_cmd_with_timing(cmd)
        cmd = f'cp {self.inputs.sub_mri_dir}/transforms/talairach.auto.xfm {self.inputs.sub_mri_dir}/transforms/talairach.xfm'
        run_cmd_with_timing(cmd)

        # talairach.lta:  convert to lta
        freesufer_home = os.environ['FREESURFER_HOME']
        cmd = f"lta_convert --src {self.inputs.sub_mri_dir}/orig.mgz --trg {freesufer_home}/average/mni305.cor.mgz " \
              f"--inxfm {self.inputs.sub_mri_dir}/transforms/talairach.xfm --outlta {self.inputs.sub_mri_dir}/transforms/talairach.xfm.lta " \
              f"--subject fsaverage --ltavox2vox"
        run_cmd_with_timing(cmd)

        # Since we do not run mri_em_register we sym-link other talairach transform files here
        cmd = f"ln -sf {self.inputs.sub_mri_dir}/transforms/talairach.xfm.lta {self.inputs.sub_mri_dir}/transforms/talairach_with_skull.lta"
        run_cmd_with_timing(cmd)
        cmd = f"ln -sf {self.inputs.sub_mri_dir}/transforms/talairach.xfm.lta {self.inputs.sub_mri_dir}/transforms/talairach.lta"
        run_cmd_with_timing(cmd)

        # Add xfm to nu
        cmd = f'mri_add_xform_to_header -c {self.inputs.sub_mri_dir}/transforms/talairach.xfm {self.inputs.sub_mri_dir}/orig_nu.mgz {self.inputs.sub_mri_dir}/nu.mgz'
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["talairach_file"] = Path(f"{self.inputs.sub_mri_dir}/transforms/talairach.lta")
        outputs['nu_file'] = Path(f"{self.inputs.sub_mri_dir}/nu.mgz")
        return outputs


if __name__ == '__main__':
    start_time = time.time()

    args = parse_args()

    data_path = Path(args.bd)

    python = args.python
    fastsurfer_home = Path.cwd() / "FastSurfer"

    N4_bias_correct_node = Node(N4BiasCorrect(fastsurfer_home), name="N4_bias_correct_node")
    talairach_and_nu_node = Node(TalairachAndNu(), name="talairach_and_nu_node")
    talairach_and_nu_node.inputs.sub_mri_dir = ''
    talairach_and_nu_node.inputs.threads = 30

    wf = Workflow(name="talairach_and_nu")
    wf.connect([(N4_bias_correct_node, talairach_and_nu_node), [("nu_file", "orig_nu_file")]])
