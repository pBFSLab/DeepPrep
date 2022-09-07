from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, Directory, Str
from nipype import Node, Workflow

from cmd import run_cmd_with_timing
import os
import time
from pathlib import Path
import argparse


def get_freesurfer_threads(threads: int):
    if threads and threads > 1:
        fsthreads = f'-threads {threads} -itkthreads {threads}'
    else:
        fsthreads = ''
    return fsthreads


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


class OrigAndRawavgInputSpec(BaseInterfaceInputSpec):
    t1w_files = traits.List(desc='t1w path or t1w paths', mandatory=True)
    subject_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    threads = traits.Int(desc='threads')


class OrigAndRawavgOutputSpec(TraitedSpec):
    orig_file = File(exists=True, desc='orig.mgz')
    rawavg_file = File(exists=True, desc='rawavg.mgz')


class OrigAndRawavg(BaseInterface):
    input_spec = OrigAndRawavgInputSpec
    output_spec = OrigAndRawavgOutputSpec

    def __init__(self):
        super(OrigAndRawavg, self).__init__()

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)

        files = ' -i '.join(self.inputs.t1w_files)
        cmd = f"recon-all -subject {self.inputs.subject_id} -i {files} -motioncor {fsthreads}"
        run_cmd_with_timing(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["orig_file"] = Path(f"{self.inputs.subject_dir}/{self.inputs.subject_id}/mri/orig.mgz")
        outputs['rawavg_file'] = Path(f"{self.inputs.subject_dir}/{self.inputs.subject_id}/mri/rawavg.mgz")
        return outputs


class FilledInputSpec(BaseInterfaceInputSpec):
    aseg_auto_file = File(exists=True, desc='mri/aseg.auto.mgz', mandatory=True)
    norm_file = File(exists=True, desc='mri/norm.mgz', mandatory=True)
    brainmask_file = File(exists=True, desc='mri/brainmask.mgz', mandatory=True)
    talairach_file = File(exists=True, desc='mri/transforms/talairach.lta')
    subject_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    threads = traits.Int(desc='threads')


class FilledOutputSpec(TraitedSpec):
    aseg_presurf_file = File(exists=True, desc='mri/aseg.presurf.mgz')
    brain_file = File(exists=True, desc='mri/brain.mgz')
    brain_finalsurfs_file = File(exists=True, desc='mri/brain.finalsurfs.mgz')
    wm_file = File(exists=True, desc='mri/wm.mgz')


class Filled(BaseInterface):
    input_spec = OrigAndRawavgInputSpec
    output_spec = OrigAndRawavgOutputSpec

    def __init__(self):
        super(Filled, self).__init__()

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)

        files = ' -i '.join(self.inputs.t1w_files)
        cmd = f"recon-all -subject {self.inputs.subject_id} -i {files} -motioncor {fsthreads}"
        run_cmd_with_timing(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["orig_file"] = Path(f"{self.inputs.subject_dir}/{self.inputs.subject_id}/mri/orig.mgz")
        outputs['rawavg_file'] = Path(f"{self.inputs.subject_dir}/{self.inputs.subject_id}/mri/rawavg.mgz")
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
