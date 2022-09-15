from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, Directory, Str, traits_extension
from nipype import Node, Workflow
from run import run_cmd_with_timing, parse_args
import os
from pathlib import Path
import argparse


def get_freesurfer_threads(threads: int):
    if threads and threads > 1:
        fsthreads = f'-threads {threads} -itkthreads {threads}'
    else:
        fsthreads = ''
    return fsthreads


class BrainmaskInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    need_t1 = traits.BaseCBool(desc='bool', mandatory=True)
    nu_file = File(exists=True, desc="mri/nu.mgz", mandatory=True)
    mask_file = File(exists=True, desc="mri/mask.mgz", mandatory=True)

    T1_file = File(exists=False, desc="mri/T1.mgz", mandatory=True)
    brainmask_file = File(exists=False, desc="mri/brainmask.mgz", mandatory=True)
    norm_file = File(exists=False, desc="mri/norm.mgz", mandatory=True)


class BrainmaskOutputSpec(TraitedSpec):
    brainmask_file = File(exists=True, desc="mri/brainmask.mgz")
    norm_file = File(exists=True, desc="mri/norm.mgz")
    T1_file = File(exists=False, desc="mri/T1.mgz")


class Brainmask(BaseInterface):
    input_spec = BrainmaskInputSpec
    output_spec = BrainmaskOutputSpec

    time = 74 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

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
            cmd = f'cp {self.inputs.norm_file} {self.inputs.brainmask_file}'
            run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["brainmask_file"] = self.inputs.brainmask_file
        outputs["norm_file"] = self.inputs.norm_file
        outputs["T1_file"] = self.inputs.T1_file

        return outputs


class OrigAndRawavgInputSpec(BaseInterfaceInputSpec):
    t1w_files = traits.List(desc='t1w path or t1w paths', mandatory=True)
    subjects_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
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
        outputs["orig_file"] = Path(f"{self.inputs.subjects_dir}/{self.inputs.subject_id}/mri/orig.mgz")
        outputs['rawavg_file'] = Path(f"{self.inputs.subjects_dir}/{self.inputs.subject_id}/mri/rawavg.mgz")
        return outputs


class FilledInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc='subject dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    threads = traits.Int(desc='threads')

    aseg_auto_file = File(exists=True, desc='mri/aseg.auto.mgz', mandatory=True)
    norm_file = File(exists=True, desc='mri/norm.mgz', mandatory=True)
    brainmask_file = File(exists=True, desc='mri/brainmask.mgz', mandatory=True)
    talairach_file = File(exists=True, desc='mri/transforms/talairach.lta', mandatory=True)


class FilledOutputSpec(TraitedSpec):
    aseg_presurf_file = File(exists=True, desc='mri/aseg.presurf.mgz')
    brain_file = File(exists=True, desc='mri/brain.mgz')
    brain_finalsurfs_file = File(exists=True, desc='mri/brain.finalsurfs.mgz')
    wm_file = File(exists=True, desc='mri/wm.mgz')
    wm_filled = File(exists=True, desc='mri/filled.mgz')


class Filled(BaseInterface):
    input_spec = FilledInputSpec
    output_spec = FilledOutputSpec

    def __init__(self):
        super(Filled, self).__init__()

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)

        cmd = f'recon-all -subject {self.inputs.subject_id} -asegmerge {fsthreads}'
        run_cmd_with_timing(cmd)
        cmd = f'recon-all -subject {self.inputs.subject_id} -normalization2 {fsthreads}'
        run_cmd_with_timing(cmd)
        cmd = f'recon-all -subject {self.inputs.subject_id} -maskbfs {fsthreads}'
        run_cmd_with_timing(cmd)
        cmd = f'recon-all -subject {self.inputs.subject_id} -segmentation {fsthreads}'
        run_cmd_with_timing(cmd)
        cmd = f'recon-all -subject {self.inputs.subject_id} -fill {fsthreads}'
        run_cmd_with_timing(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["aseg_presurf_file"] = Path(self.inputs.subjects_dir, self.inputs.subject_id, 'mri/aseg.presurf.mgz')
        outputs["brain_file"] = Path(self.inputs.subjects_dir, self.inputs.subject_id, 'mri/brain.mgz')
        outputs["brain_finalsurfs_file"] = Path(self.inputs.subjects_dir, self.inputs.subject_id, 'mri/brain.finalsurfs.mgz')
        outputs["wm_file"] = Path(self.inputs.subjects_dir, self.inputs.subject_id, 'mri/wm.mgz')
        outputs["wm_filled"] = Path(self.inputs.subjects_dir, self.inputs.subject_id, 'mri/filled.mgz')
        return outputs


class WhitePreaparcInputSpec(BaseInterfaceInputSpec):
    fswhitepreaparc = traits.Bool(desc="True: mris_make_surfaces; \
    False: recon-all -autodetgwstats -white-preaparc -cortex-label", mandatory=True)
    subject = traits.Str(desc="sub-xxx", mandatory=True)
    hemi = traits.Str(desc="?h", mandatory=True)

    # input files of <mris_make_surfaces>
    aseg_presurf = File(exists=True, desc="mri/aseg.presurf.mgz")
    brain_finalsurfs = File(exists=True, desc="mri/brain.finalsurfs.mgz")
    wm_file = File(exists=True, desc="mri/wm.mgz")
    filled_file = File(exists=True, desc="mri/filled.mgz")
    hemi_orig = File(exists=True, desc="surf/?h.orig")

    # input files of <recon-all -autodetgwstats>
    hemi_orig_premesh = File(exists=True, desc="surf/?h.orig.premesh")

    # input files of <recon-all -white-paraparc>
    autodet_gw_stats_hemi_dat = File(exists=True, desc="surf/autodet.gw.stats.?h.dat")

    # input files of <recon-all -cortex-label>
    hemi_white_preaparc = File(exists=True, desc="surf/?h.white.preaparc")


class WhitePreaparcOutputSpec(TraitedSpec):
    # output files of mris_make_surfaces
    hemi_white_preaparc = File(exists=True, desc="surf/?h.white.preaparc")
    hemi_curv = File(exists=True, desc="surf/?h.curv")
    hemi_area = File(exists=True, desc="surf/?h.area")
    hemi_cortex_label = File(exists=True, desc="label/?h.cortex.label")


class WhitePreaparc(BaseInterface):
    input_spec = WhitePreaparcInputSpec
    output_spec = WhitePreaparcOutputSpec

    def __init__(self, output_dir: Path, threads: int):
        super(WhitePreaparc, self).__init__()
        self.output_dir = output_dir
        self.threads = threads
        self.fsthreads = get_freesurfer_threads(threads)

    def _run_interface(self, runtime):
        if not traits_extension.isdefined(self.inputs.brain_finalsurfs):
            self.inputs.brain_finalsurfs = self.output_dir / f"{self.inputs.subject}" / "mri/brain.finalsurfs.mgz"
        if not traits_extension.isdefined(self.inputs.wm_file):
            self.inputs.wm_file = self.output_dir / f"{self.inputs.subject}" / "mri/wm.mgz"
        print("-------------")
        print(f"self.inputs.brain_finalsurfs {self.inputs.brain_finalsurfs}")
        print(f"self.inputs.wm_file {self.inputs.wm_file}")
        print("--------------")

        if self.inputs.fswhitepreaparc:
            time = 130 / 60
            cpu = 1.25
            gpu = 0

            if not traits_extension.isdefined(self.inputs.aseg_presurf):
                self.inputs.aseg_presurf = self.output_dir / f"{self.inputs.subject}" / "mri/aseg.presurf.mgz"
            if not traits_extension.isdefined(self.inputs.filled_file):
                self.inputs.filled_file = self.output_dir / f"{self.inputs.subject}" / "mri/filled.mgz"
            if not traits_extension.isdefined(self.inputs.hemi_orig):
                self.inputs.hemi_orig = self.output_dir / f"{self.inputs.subject}" / "surf" / f"{self.inputs.hemi}.orig"
            print("*" * 10)
            print(f"self.inputs.aseg_presurf {self.inputs.aseg_presurf}")
            print(f"self.inputs.filled_file {self.inputs.filled_file}")
            print(f"self.inputs.hemi_orig {self.inputs.hemi_orig}")
            print("*" * 10)

            cmd = f'mris_make_surfaces -aseg aseg.presurf -white white.preaparc -whiteonly -noaparc -mgz ' \
                  f'-T1 brain.finalsurfs {self.inputs.subject} {self.inputs.hemi} threads {self.threads}'
            run_cmd_with_timing(cmd)
        else:
            # time = ? / 60
            # cpu = ?
            # gpu = 0

            if not traits_extension.isdefined(self.inputs.hemi_orig_premesh):
                self.inputs.hemi_orig_premesh = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.orig.premesh"

            cmd = f'recon-all -subject {self.inputs.subject} -hemi {self.inputs.hemi} -autodetgwstats -white-preaparc -cortex-label ' \
                  f'-no-isrunning {self.fsthreads}'
            run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["hemi_white_preaparc"] = self.inputs.hemi_white_preaparc
        outputs["hemi_curv"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.curv"
        outputs["hemi_area"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.area"
        outputs[
            "hemi_cortex_label"] = self.output_dir / f"{self.inputs.subject}" / f"label/{self.inputs.hemi}.cortex.label"
        return outputs


class InflatedSphereThresholdInputSpec(BaseInterfaceInputSpec):
    hemi = traits.String(mandatory=True, desc='hemi')
    subject = traits.String(mandatory=True, desc='recon')
    white_preaparc_file = File(exists=True, mandatory=True, desc='surf/?h.white.preaparc')
    smoothwm_file = File(mandatory=True, desc='surf/?h.smoothwm')
    inflated_file = File(mandatory=True, desc='surf/?h.inflated')  # Do not set exists=True !!
    sulc_file = File(mandatory=True, desc="surf/?h.sulc")
    threads = traits.Int(desc='threads')


class InflatedSphereThresholdOutputSpec(TraitedSpec):
    smoothwm_file = File(exists=True, mandatory=True, desc='surf/?h.smoothwm')
    inflated_file = File(exists=True, mandatory=True, desc='surf/?h.inflated')  # Do not set exists=True !!
    sulc_file = File(exists=True, mandatory=True, desc="surf/?h.sulc")


class InflatedSphere(BaseInterface):
    input_spec = InflatedSphereThresholdInputSpec
    output_spec = InflatedSphereThresholdOutputSpec

    time = 351 / 60  # 运行时间：分钟
    cpu = 5  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)
        # create nicer inflated surface from topo fixed (not needed, just later for visualization)
        cmd = f"recon-all -subject {self.inputs.subject} -hemi {self.inputs.hemi} -smooth2 -no-isrunning {fsthreads}"
        run_cmd_with_timing(cmd)

        cmd = f"recon-all -subject {self.inputs.subject} -hemi {self.inputs.hemi} -inflate2 -no-isrunning {fsthreads}"
        run_cmd_with_timing(cmd)

        cmd = f"recon-all -subject {self.inputs.subject} -hemi {self.inputs.hemi} -sphere -no-isrunning {fsthreads}"
        run_cmd_with_timing(cmd)
        return runtime

        def _list_outputs(self):
            outputs = self._outputs().get()
            outputs['smoothwm_file'] = self.inputs.smoothwm_file
            outputs['inflated_file'] = self.inputs.inflated_file
            outputs['sulc_file'] = self.inputs.sulc_file
            return outputs


class WhitePialThickness1InputSpec(BaseInterfaceInputSpec):
    subject = traits.Str(desc="sub-xxx", mandatory=True)
    hemi = traits.Str(desc="?h", mandatory=True)


class WhitePialThickness1OutputSpec(TraitedSpec):
    hemi_white = File(exists=True, desc="surf/?h.white")
    hemi_pial_t1 = File(exists=True, desc="surf/?h.pial.T1")

    hemi_pial = File(exists=True, desc="surf/?h.pial")

    hemi_curv = File(exists=True, desc="surf/?h.curv")
    hemi_area = File(exists=True, desc="surf/?h.area")
    hemi_curv_pial = File(exists=True, desc="surf/?h.curv.pial")
    hemi_area_pial = File(exists=True, desc="surf/?h.area.pial")
    hemi_thickness = File(exists=True, desc="surf/?h.thickness")


class WhitePialThickness1(BaseInterface):
    # The two methods (WhitePialThickness1 and WhitePialThickness2) are exacly same.
    input_spec = WhitePialThickness1InputSpec
    output_spec = WhitePialThickness1OutputSpec

    def __init__(self, output_dir: Path, threads: int):
        super(WhitePialThickness1, self).__init__()
        self.output_dir = output_dir
        self.threads = threads
        self.fsthreads = get_freesurfer_threads(threads)

    def _run_interface(self, runtime):
        # must run surfreg first
        # 20-25 min for traditional surface segmentation (each hemi)
        # this creates aparc and creates pial using aparc, also computes jacobian
        # FreeSurfer 7.2
        time = (135 + 120) / 60
        cpu = 6
        gpu = 0
        # FreeSurfer 6.0
        # time = (474 + 462) / 60

        cmd = f"recon-all -subject {self.inputs.subject} -hemi {self.inputs.hemi} -white " \
              f"-no-isrunning {self.fsthreads}"
        run_cmd_with_timing(cmd)
        cmd = f"recon-all -subject {self.inputs.subject} -hemi {self.inputs.hemi} -pial " \
              f"-no-isrunning {self.fsthreads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["hemi_white"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.white"
        outputs["hemi_pial_t1"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.pial.T1"
        outputs["hemi_pial"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.pial"
        outputs["hemi_curv"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.curv"
        outputs["hemi_area"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.area"
        outputs["hemi_curv_pial"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.curv.pial"
        outputs["hemi_area_pial"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.area.pial"
        outputs["hemi_thickness"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.thickness"

        return outputs


class WhitePialThickness2InputSpec(BaseInterfaceInputSpec):
    subject = traits.Str(desc="sub-xxx", mandatory=True)
    hemi = traits.Str(desc="?h", mandatory=True)

    autodet_gw_stats_hemi_dat = File(exists=True, desc="surf/autodet.gw.stats.?h.dat", mandatory=True)
    aseg_presurf = File(exists=True, desc="mri/aseg.presurf.mgz", mandatory=True)
    wm_file = File(exists=True, desc="mri/wm.mgz", mandatory=True)
    brain_finalsurfs = File(exists=True, desc="mri/brain.finalsurfs.mgz", mandatory=True)
    hemi_white_preaparc = File(exists=True, desc="surf/?h.white.preaparc", mandatory=True)
    hemi_white = File(exists=True, desc="surf/?h.white", mandatory=True)
    hemi_cortex_label = File(exists=True, desc="label/?h.cortex.label", mandatory=True)
    hemi_aparc_DKTatlas_mapped_annot = File(exists=True, desc="label/?h.aparc.DKTatlas.mapped.annot", mandatory=True)

    hemi_pial_t1 = File(exists=True, desc="surf/?h.pial.T1", mandatory=True)


class WhitePialThickness2OutputSpec(TraitedSpec):
    hemi_white = File(exists=True, desc="surf/?h.white")
    hemi_pial_t1 = File(exists=True, desc="surf/?h.pial.T1")

    hemi_pial = File(exists=True, desc="surf/?h.pial")

    hemi_curv = File(exists=True, desc="surf/?h.curv")
    hemi_area = File(exists=True, desc="surf/?h.area")
    hemi_curv_pial = File(exists=True, desc="surf/?h.curv.pial")
    hemi_area_pial = File(exists=True, desc="surf/?h.area.pial")
    hemi_thickness = File(exists=True, desc="surf/?h.thickness")


class WhitePialThickness2(BaseInterface):
    # The two methods (WhitePialThickness1 and WhitePialThickness2) are exacly same.
    input_spec = WhitePialThickness1InputSpec
    output_spec = WhitePialThickness1OutputSpec

    def __init__(self, output_dir: Path, threads: int):
        super(WhitePialThickness2, self).__init__()
        self.output_dir = output_dir
        self.threads = threads
        self.fsthreads = get_freesurfer_threads(threads)

    def _run_interface(self, runtime):
        # The two methods below are exacly same.
        # 4 min compute white :
        time = 330 / 60
        cpu = 1
        gpu = 0

        cmd = f"mris_place_surface --adgws-in {self.inputs.autodet_gw_stats_hemi_dat} " \
              f"--seg {self.inputs.aseg_presurf} --wm {self.inputs.wm_file} --invol {self.inputs.brain_finalsurfs} --{self.inputs.hemi} " \
              f"--i {self.inputs.hemi_white_preaparc} --o {self.inputs.hemi_white} --white --nsmooth 0 " \
              f"--rip-label {self.inputs.hemi_cortex_label} --rip-bg --rip-surf {self.inputs.hemi_white_preaparc} " \
              f"--aparc {self.inputs.hemi_aparc_DKTatlas_mapped_annot}"
        run_cmd_with_timing(cmd)
        # 4 min compute pial :
        cmd = f"mris_place_surface --adgws-in {self.inputs.autodet_gw_stats_hemi_dat} --seg {self.inputs.aseg_presurf} " \
              f"--wm {self.inputs.wm_file} --invol {self.inputs.brain_finalsurfs} --{self.inputs.hemi} --i {self.inputs.hemi_white} " \
              f"--o {self.inputs.hemi_pial_t1} --pial --nsmooth 0 --rip-label {self.inputs.hemi_cortexhipamyg_label} " \
              f"--pin-medial-wall {self.inputs.hemi_cortex_label} --aparc {self.inputs.hemi_aparc_DKTatlas_mapped_annot} " \
              f"--repulse-surf {self.inputs.hemi_white} --white-surf {self.inputs.hemi_white}"
        run_cmd_with_timing(cmd)

        # Here insert DoT2Pial  later --> if T2pial is not run, need to softlink pial.T1 to pial!

        cmd = f"cp {self.inputs.hemi_pial_t1} {self.inputs.hemi_pial}"
        run_cmd_with_timing(cmd)

        # these are run automatically in fs7* recon-all and
        # cannot be called directly without -pial flag (or other t2 flags)
        cmd = f"mris_place_surface --curv-map {self.inputs.hemi_white} 2 10 {self.inputs.hemi_curv}"
        run_cmd_with_timing(cmd)
        cmd = f"mris_place_surface --area-map {self.inputs.hemi_white} {self.inputs.hemi_area}"
        run_cmd_with_timing(cmd)
        cmd = f"mris_place_surface --curv-map {self.inputs.hemi_pial} 2 10 {self.inputs.hemi_curv_pial}"
        run_cmd_with_timing(cmd)
        cmd = f"mris_place_surface --area-map {self.inputs.hemi_pial} {self.inputs.hemi_area_pial}"
        run_cmd_with_timing(cmd)
        cmd = f" mris_place_surface --thickness {self.inputs.hemi_white} {self.inputs.hemi_pial} " \
              f"20 5 {self.inputs.hemi_thickness}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["hemi_white"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.white"
        outputs["hemi_pial_t1"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.pial.T1"
        outputs["hemi_pial"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.pial"
        outputs["hemi_curv"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.curv"
        outputs["hemi_area"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.area"
        outputs["hemi_curv_pial"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.curv.pial"
        outputs["hemi_area_pial"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.area.pial"
        outputs["hemi_thickness"] = self.output_dir / f"{self.inputs.subject}" / f"surf/{self.inputs.hemi}.thickness"

        return outputs


class CurvstatsInputSpec(BaseInterfaceInputSpec):
    subject_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    hemi = Str(desc="lh/rh", mandatory=True)
    hemi_smoothwm_file = File(exists=True, desc="surf/?h.smoothwm", mandatory=True)
    hemi_curv_file = File(exists=True, desc="surf/?h.curv", mandatory=True)
    hemi_sulc_file = File(exists=True, desc="surf/?h.sulc", mandatory=True)
    threads = traits.Int(desc='threads')

    hemi_curv_stats_file = File(exists=False, desc="stats/?h.curv.stats", mandatory=True)


class CurvstatsOutputSpec(TraitedSpec):
    hemi_curv_stats_file = File(exists=True, desc="stats/?h.curv.stats")


class Curvstats(BaseInterface):
    input_spec = CurvstatsInputSpec
    output_spec = CurvstatsOutputSpec

    time = 3.1 / 60  # 运行时间：分钟 / 单脑测试时间
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)

        # in FS7 curvstats moves here
        cmd = f"recon-all -subject {self.inputs.subject_id} -hemi {self.inputs.hemi} -curvstats -no-isrunning {fsthreads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["hemi_curv_stats_file"] = self.inputs.hemi_curv_stats_file

        return outputs


class CortribbonInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    threads = traits.Int(desc='threads')
    aseg_presurf_file = File(exists=True, desc="mri/aseg.presurf.mgz", mandatory=True)
    hemi = Str(desc="lh/rh", mandatory=True)
    hemi_white = File(exists=True, desc="surf/?h.white", mandatory=True)
    hemi_pial = File(exists=True, desc="surf/?h.pial", mandatory=True)

    hemi_ribbon = File(exists=False, desc="mri/?h.ribbon.mgz", mandatory=True)
    ribbon = File(exists=False, desc="mri/ribbon.mgz", mandatory=True)


class CortribbonOutputSpec(TraitedSpec):
    hemi_ribbon = File(exists=True, desc="mri/?h.ribbon.mgz")
    ribbon = File(exists=True, desc="mri/ribbon.mgz")


class Cortribbon(BaseInterface):
    input_spec = CortribbonInputSpec
    output_spec = CortribbonOutputSpec

    time = 203 / 60  # 运行时间：分钟 / 单脑测试时间
    cpu = 3.5  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)
        # -cortribbon 4 minutes, ribbon is used in mris_anatomical stats
        # to remove voxels from surface based volumes that should not be cortex
        # anatomical stats can run without ribon, but will omit some surface based measures then
        # wmparc needs ribbon, probably other stuff (aparc to aseg etc).
        # could be stripped but lets run it to have these measures below
        cmd = f"recon-all -subject {self.inputs.subject_id} -cortribbon {fsthreads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["hemi_ribbon"] = self.inputs.hemi_ribbon
        outputs["ribbon"] = self.inputs.ribbon

        return outputs


class ParcstatsInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    threads = traits.Int(desc='threads')
    hemi = Str(desc="lh/rh", mandatory=True)
    hemi_aparc_annot_file = File(exists=True, desc="label/?h.aparc.annot", mandatory=True)
    wm_file = File(exists=True, desc="mri/wm.mgz", mandatory=True)
    ribbon_file = File(exists=True, desc="mri/ribbon.mgz", mandatory=True)
    hemi_white_file = File(exists=True, desc="surf/?h.white", mandatory=True)
    hemi_pial_file = File(exists=True, desc="surf/?h.pial", mandatory=True)
    hemi_thickness_file = File(exists=True, desc="surf/?h.thickness", mandatory=True)

    hemi_aparc_stats_file = File(exists=False, desc="stats/?h.aparc.stats", mandatory=True)
    hemi_aparc_pial_stats_file = File(exists=False, desc="stats/?h.aparc.pial.stats", mandatory=True)
    aparc_annot_ctab_file = File(exists=False, desc="label/aparc.annot.ctab", mandatory=True)


class ParcstatsOutputSpec(TraitedSpec):
    hemi_aparc_stats_file = File(exists=True, desc="stats/?h.aparc.stats")
    hemi_aparc_pial_stats_file = File(exists=True, desc="stats/?h.aparc.pial.stats")
    aparc_annot_ctab_file = File(exists=True, desc="label/aparc.annot.ctab")


class Parcstats(BaseInterface):
    input_spec = ParcstatsInputSpec
    output_spec = ParcstatsOutputSpec

    time = 91 / 60  # 运行时间：分钟 / 单脑测试时间
    cpu = 3  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)

        cmd = f"recon-all -subject {self.inputs.subject_id} -parcstats {fsthreads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["hemi_aparc_stats_file"] = self.inputs.hemi_aparc_stats_file
        outputs["hemi_aparc_pial_stats_file"] = self.inputs.hemi_aparc_pial_stats_file
        outputs["aparc_annot_ctab_file"] = self.inputs.aparc_annot_ctab_file

        return outputs


class PctsurfconInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    threads = traits.Int(desc='threads')
    hemi = Str(desc="lh/rh", mandatory=True)
    rawavg_file = File(exists=True, desc="mri/rawavg.mgz", mandatory=True)
    orig_file = File(exists=True, desc="mri/orig.mgz", mandatory=True)
    hemi_cortex_label_file = File(exists=True, desc="label/?h.cortex.label", mandatory=True)
    hemi_white_file = File(exists=True, desc="surf/?h.white", mandatory=True)

    hemi_wg_pct_mgh_file = File(exists=False, desc="surf/?h.w-g.pct.mgh", mandatory=True)
    hemi_wg_pct_stats_file = File(exists=False, desc="mri/?h.w-g.pct.stats", mandatory=True)


class PctsurfconOutputSpec(TraitedSpec):
    hemi_wg_pct_mgh_file = File(exists=True, desc="surf/?h.w-g.pct.mgh")
    hemi_wg_pct_stats_file = File(exists=True, desc="mri/?h.w-g.pct.stats")


class Pctsurfcon(BaseInterface):
    input_spec = PctsurfconInputSpec
    output_spec = PctsurfconOutputSpec

    time = 9 / 60  # 运行时间：分钟 / 单脑测试时间
    cpu = 2  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)

        cmd = f"recon-all -subject {self.inputs.subject_id} -pctsurfcon {fsthreads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["hemi_wg_pct_mgh_file"] = self.inputs.hemi_wg_pct_mgh_file
        outputs["hemi_wg_pct_stats_file"] = self.inputs.hemi_wg_pct_stats_file

        return outputs


class HyporelabelInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    threads = traits.Int(desc='threads')
    hemi = Str(desc="lh/rh", mandatory=True)
    aseg_presurf_file = File(exists=True, desc="mri/aseg.presurf.mgz", mandatory=True)
    hemi_white_file = File(exists=True, desc="surf/?h.white", mandatory=True)

    aseg_presurf_hypos_file = File(exists=False, desc="mri/aseg.presurf.hypos.mgz", mandatory=True)


class HyporelabelOutputSpec(TraitedSpec):
    aseg_presurf_hypos_file = File(exists=True, desc="mri/aseg.presurf.hypos.mgz")


class Hyporelabel(BaseInterface):
    input_spec = HyporelabelInputSpec
    output_spec = HyporelabelOutputSpec

    time = 12 / 60  # 运行时间：分钟 / 单脑测试时间
    cpu = 2.3  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)

        cmd = f"recon-all -subject {self.inputs.subject_id} -hyporelabel {fsthreads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["aseg_presurf_hypos_file"] = self.inputs.aseg_presurf_hypos_file

        return outputs


class JacobianAvgcurvCortparcThresholdInputSpec(BaseInterfaceInputSpec):
    hemi = traits.String(mandatory=True, desc='hemi')
    subject = traits.String(mandatory=True, desc='recon')
    white_preaparc_file = File(exists=True, mandatory=True, desc='surf/?h.white.preaparc')
    sphere_reg_file = File(exists=True, mandatory=True, desc='surf/?h.sphere.reg')
    jacobian_white_file = File(mandatory=True, desc='surf/?h.jacobian_white')
    avg_curv_file = File(mandatory=True, desc='surf/?h.avg_curv')  # Do not set exists=True !!
    aseg_presurf_file = File(exists=True, mandatory=True, desc="mri/aseg.presurf.mgz")
    cortex_label_file = File(exists=True, mandatory=True, desc="label/?h.cortex.label")
    aparc_annot_file = File(mandatory=True, desc="label/?h.aparc.annot")
    threads = traits.Int(desc='threads')


class JacobianAvgcurvCortparcThresholdOutputSpec(TraitedSpec):
    jacobian_white_file = File(exists=True, mandatory=True, desc='surf/?h.jacobian_white')
    avg_curv_file = File(exists=True, mandatory=True, desc='surf/?h.avg_curv')  # Do not set exists=True !!
    aparc_annot_file = File(exists=True, mandatory=True, desc="surf/?h.aparc.annot")


class JacobianAvgcurvCortparc(BaseInterface):
    input_spec = JacobianAvgcurvCortparcThresholdInputSpec
    output_spec = JacobianAvgcurvCortparcThresholdOutputSpec

    # time = 28 / 60  # 运行时间：分钟
    # cpu = 3  # 最大cpu占用：个
    # gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)
        # create nicer inflated surface from topo fixed (not needed, just later for visualization)
        cmd = f"recon-all -subject {self.inputs.subject} -hemi {self.inputs.hemi} -jacobian_white -avgcurv -cortparc " \
              f"-no-isrunning {fsthreads}"
        run_cmd_with_timing(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['jacobian_white_file'] = self.inputs.jacobian_white_file
        outputs['avg_curv_file'] = self.inputs.avg_curv_file
        outputs['aparc_annot_file'] = self.inputs.aparc_annot_file
        return outputs


class SegstatsInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    threads = traits.Int(desc='threads')
    hemi = Str(desc="lh/rh", mandatory=True)
    brainmask_file = File(exists=True, desc="mri/brainmask.mgz", mandatory=True)
    norm_file = File(exists=True, desc="mri/norm.mgz", mandatory=True)
    aseg_file = File(exists=True, desc="mri/aseg.mgz", mandatory=True)
    aseg_presurf_file = File(exists=True, desc="mri/aseg.presurf.mgz", mandatory=True)
    ribbon_file = File(exists=True, desc="mri/ribbon.mgz", mandatory=True)
    hemi_orig_nofix_file = File(exists=True, desc="surf/?h.orig.nofix", mandatory=True)
    hemi_white_file = File(exists=True, desc="surf/?h.white", mandatory=True)
    hemi_pial_file = File(exists=True, desc="surf/?h.pial", mandatory=True)

    aseg_stats_file = File(exists=False, desc="stats/aseg.stats", mandatory=True)


class SegstatsOutputSpec(TraitedSpec):
    aseg_stats_file = File(exists=True, desc="stats/aseg.stats")


class Segstats(BaseInterface):
    input_spec = SegstatsInputSpec
    output_spec = SegstatsOutputSpec

    time = 34 / 60  # 运行时间：分钟 / 单脑测试时间
    cpu = 8  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)

        cmd = f"recon-all -subject {self.inputs.subject_id} -segstats  {fsthreads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['aseg_stats_file'] = self.inputs.aseg_stats_file

        return outputs


class Aseg7InputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    subject_mri_dir = Directory(exists=True, desc="subject mri dir", mandatory=True)
    threads = traits.Int(desc='threads')

    aseg_presurf_hypos_file = File(exists=False, desc="mri/aseg.presurf.hypos.mgz", mandatory=True)
    # ribbon_file = File(exists=True, desc="mri/ribbon.mgz", mandatory=True)
    lh_cortex_label_file = File(exists=True, desc="label/lh.cortex.label", mandatory=True)
    lh_white_file = File(exists=True, desc="surf/lh.white", mandatory=True)
    lh_pial_file = File(exists=True, desc="surf/lh.pial", mandatory=True)
    rh_cortex_label_file = File(exists=True, desc="label/rh.cortex.label", mandatory=True)
    rh_white_file = File(exists=True, desc="surf/rh.white", mandatory=True)
    rh_pial_file = File(exists=True, desc="surf/rh.pial", mandatory=True)
    lh_aparc_annot_file = File(exists=True, desc="surf/lh.aparc.annot", mandatory=True)
    rh_aparc_annot_file = File(exists=True, desc="surf/rh.aparc.annot", mandatory=True)

    aparc_aseg_file = File(exists=False, desc="mri/aparc+aseg.mgz", mandatory=True)


class Aseg7OutputSpec(TraitedSpec):
    aparc_aseg_file = File(exists=True, desc="mri/aparc+aseg.mgz")


class Aseg7(BaseInterface):
    input_spec = Aseg7InputSpec
    output_spec = Aseg7OutputSpec

    time = 0 / 60  # 运行时间：分钟 / 单脑测试时间
    cpu = 0  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)
        cmd = f'mri_surf2volseg --o aparc+aseg.mgz --label-cortex --i aseg.mgz ' \
              f'--threads {threads} ' \
              f'--lh-annot {self.inputs.lh_aparc_annot_file} 1000 ' \
              f'--lh-cortex-mask {self.inputs.lh_cortex_label_file} --lh-white {self.inputs.lh_white_file} ' \
              f'--lh-pial {self.inputs.lh_pial_file} --rh-annot {self.inputs.rh_aparc_annot_file} ' \
              f'--rh-cortex-mask {self.inputs.rh_cortex_label_file} --rh-white {self.inputs.rh_white_file} ' \
              f'--rh-pial {self.inputs.rh_pial_file} '
        run_cmd_with_timing(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["aparc_aseg_file"] = self.inputs.aparc_aseg_file
        return outputs


class Aseg7ToAsegInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    threads = traits.Int(desc='threads')

    # aseg_presurf_hypos_file = File(exists=True, desc="mri/aseg.presurf.hypos.mgz", mandatory=True)
    # ribbon_file = File(exists=True, desc="mri/ribbon.mgz", mandatory=True)
    lh_cortex_label_file = File(exists=True, desc="label/lh.cortex.label", mandatory=True)
    lh_white_file = File(exists=True, desc="surf/lh.white", mandatory=True)
    lh_pial_file = File(exists=True, desc="surf/lh.pial", mandatory=True)
    rh_cortex_label_file = File(exists=True, desc="label/rh.cortex.label", mandatory=True)
    rh_white_file = File(exists=True, desc="surf/rh.white", mandatory=True)
    rh_pial_file = File(exists=True, desc="surf/rh.pial", mandatory=True)

    aseg_file = File(exists=False, desc="mri/aseg.mgz", mandatory=True)


class Aseg7ToAsegOutputSpec(TraitedSpec):
    aseg_file = File(exists=True, desc="mri/aseg.mgz")


class Aseg7ToAseg(BaseInterface):
    input_spec = Aseg7ToAsegInputSpec
    output_spec = Aseg7ToAsegOutputSpec

    time = 0 / 60  # 运行时间：分钟 / 单脑测试时间
    cpu = 0  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        threads = self.inputs.threads if self.inputs.threads else 0
        fsthreads = get_freesurfer_threads(threads)

        cmd = f"recon-all -subject {self.inputs.subject_id} -hyporelabel -apas2aseg {fsthreads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["aseg_file"] = self.inputs.aseg_file

        return outputs
