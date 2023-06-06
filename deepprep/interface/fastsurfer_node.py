from pathlib import Path
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, Directory, \
    traits, traits_extension, Str
from deepprep.interface.run import run_cmd_with_timing, multipool


class SegmentInputSpec(BaseInterfaceInputSpec):
    python_interpret = File(exists=True, mandatory=True, desc='the python interpret to use')
    eval_py = File(exists=True, mandatory=True, desc="FastSurfer segmentation script")
    subjects_dir = Directory(exists=True, desc='subjects dir', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    # in_file = File(exists=True, mandatory=True, desc='name of file to process. Default: mri/orig.mgz')
    # out_file = File(mandatory=True,
    #                 desc='name under which segmentation will be saved. Default: mri/aparc.DKTatlas+aseg.deep.mgz. '
    #                      'If a separate subfolder is desired (e.g. FS conform, add it to the name: '
    #                      'mri/aparc.DKTatlas+aseg.deep.mgz)')  # Do not set exists=True !!
    # conformed_file = File(desc='Name under which the conformed input image will be saved, in the same directory as '
    #                            'the segmentation (the input image is always conformed first, if it is not already '
    #                            'conformed). The original input image is saved in the output directory as '
    #                            '$id/mri/orig/001.mgz. Default: mri/conform.mgz.')

    network_sagittal_path = File(exists=True, mandatory=True, desc="path to pre-trained weights of sagittal network")
    network_coronal_path = File(exists=True, mandatory=True, desc="pre-trained weights of coronal network")
    network_axial_path = File(exists=True, mandatory=True, desc="pre-trained weights of axial network")

    # aparc_DKTatlas_aseg_deep = File(exists=False, desc="mri/aparc.DKTatlas+aseg.deep.mgz", mandatory=True)
    # aparc_DKTatlas_aseg_orig = File(exists=False, desc="mri/aparc.DKTatlas+aseg.orig.mgz", mandatory=True)


class SegmentOutputSpec(TraitedSpec):
    aparc_DKTatlas_aseg_deep = File(exists=True, desc="mri/aparc.DKTatlas+aseg.deep.mgz")
    aparc_DKTatlas_aseg_orig = File(exists=True, desc="mri/aparc.DKTatlas+aseg.orig.mgz")


class Segment(BaseInterface):
    input_spec = SegmentInputSpec
    output_spec = SegmentOutputSpec

    time = 24 / 60  # 运行时间：分钟
    cpu = 10  # 最大cpu占用：个
    gpu = 9756  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        in_file = subjects_dir / subject_id / 'mri' / 'orig.mgz'
        conformed_file = subjects_dir / subject_id / 'mri' / 'conformed.mgz'

        aparc_DKTatlas_aseg_deep = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'
        aparc_DKTatlas_aseg_orig = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.orig.mgz'

        # if not traits_extension.isdefined(self.inputs.conformed_file):
        #     conformed_file = Path(self.inputs.in_file).parent / 'conformed.mgz'
        # else:
        #     conformed_file = self.inputs.conformed_file
        cmd = f'{self.inputs.python_interpret} {self.inputs.eval_py} ' \
              f'--in_name {in_file} ' \
              f'--out_name {aparc_DKTatlas_aseg_deep} ' \
              f'--conformed_name {conformed_file} ' \
              '--order 1 ' \
              f'--network_sagittal_path {self.inputs.network_sagittal_path} ' \
              f'--network_coronal_path {self.inputs.network_coronal_path} ' \
              f'--network_axial_path {self.inputs.network_axial_path} ' \
              '--batch_size 1 --simple_run --run_viewagg_on check'
        run_cmd_with_timing(cmd)

        cmd = f'mri_convert {aparc_DKTatlas_aseg_deep} {aparc_DKTatlas_aseg_orig}'
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        outputs["aparc_DKTatlas_aseg_deep"] = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'
        outputs["aparc_DKTatlas_aseg_orig"] = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.orig.mgz'

        return outputs

    def create_sub_node(self):
        from interface.create_node_structure import create_Noccseg_node
        node = create_Noccseg_node(self.inputs.subject_id)
        return node


class N4BiasCorrectInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc='subjects dir', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    python_interpret = File(exists=True, desc="default: python3", mandatory=True)
    correct_py = File(exists=True, desc="N4_bias_correct.py", mandatory=True)
    orig_file = File(exists=True, desc="mri/orig.mgz", mandatory=True)
    mask_file = File(exists=True, desc="mri/mask.mgz", mandatory=True)
    # orig_nu_file = File(desc="mri/orig_nu.mgz", mandatory=True)
    threads = traits.Int(desc="threads")


class N4BiasCorrectOutputSpec(TraitedSpec):
    orig_nu_file = File(exists=True, desc="mri/orig_nu.mgz")
    subject_id = Str(desc='subject id')


class N4BiasCorrect(BaseInterface):
    input_spec = N4BiasCorrectInputSpec
    output_spec = N4BiasCorrectOutputSpec

    time = 7 / 60  # per minute
    cpu = 5.5
    gpu = 0

    def _run_interface(self, runtime):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        orig_nu_file = subjects_dir / subject_id / "mri" / "orig_nu.mgz"
        # orig_nu nu correct
        if not traits_extension.isdefined(self.inputs.threads):
            self.inputs.threads = 1

        py = self.inputs.correct_py
        cmd = f"{self.inputs.python_interpret} {py} --in {self.inputs.orig_file} --out {orig_nu_file} " \
              f"--mask {self.inputs.mask_file}  --threads {self.inputs.threads}"
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        outputs = self._outputs().get()
        outputs['orig_nu_file'] = subjects_dir / subject_id / "mri" / "orig_nu.mgz"
        outputs['subject_id'] = subject_id
        return outputs

    def create_sub_node(self):
        from interface.create_node_structure import create_TalairachAndNu_node
        node = create_TalairachAndNu_node(self.inputs.subject_id)
        return node


class TalairachAndNuInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subjects dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    threads = traits.Int(desc="threads")
    orig_nu_file = File(exists=True, desc="mri/orig_nu.mgz", mandatory=True)
    orig_file = File(exists=True, desc="mri/orig.mgz", mandatory=True)
    mni305 = File(exists=True, desc="FREESURFER/average/mni305.cor.mgz", mandatory=True)

    # talairach_lta = File(desc="mri/transforms/talairach.lta")
    # nu_file = File(desc="mri/nu.mgz", mandatory=True)


class TalairachAndNuOutputSpec(TraitedSpec):
    talairach_lta = File(exists=True, desc="mri/transforms/talairach.lta")
    nu_file = File(exists=True, desc="mri/nu.mgz")
    subject_id = Str(desc='subject id')


class TalairachAndNu(BaseInterface):
    input_spec = TalairachAndNuInputSpec
    output_spec = TalairachAndNuOutputSpec

    time = 19 / 60
    cpu = 1
    gpu = 0

    def _run_interface(self, runtime):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        sub_mri_dir = subjects_dir / subject_id / "mri"
        nu_file = subjects_dir / subject_id / 'mri' / 'nu.mgz'

        if self.inputs.threads is None:
            self.inputs.threads = 1
        talairach_auto_xfm = sub_mri_dir / "transforms" / "talairach.auto.xfm"
        talairach_xfm = sub_mri_dir / "transforms" / "talairach.xfm"
        talairach_xfm_lta = sub_mri_dir / "transforms" / "talairach.xfm.lta"

        # talairach.xfm: compute talairach full head (25sec)
        cmd = f'cd {sub_mri_dir} && ' \
              f'talairach_avi --i {self.inputs.orig_nu_file} --xfm {talairach_auto_xfm}'
        run_cmd_with_timing(cmd)
        cmd = f'cp {talairach_auto_xfm} {talairach_xfm}'
        run_cmd_with_timing(cmd)

        # talairach.lta:  convert to lta
        cmd = f"lta_convert --src {self.inputs.orig_file} --trg {self.inputs.mni305} " \
              f"--inxfm {talairach_xfm} --outlta {talairach_xfm_lta} " \
              f"--subject fsaverage --ltavox2vox"
        run_cmd_with_timing(cmd)

        # Since we do not run mri_em_register we sym-link other talairach transform files here
        talairach_skull_lta = sub_mri_dir / "transforms" / "talairach_with_skull.lta"
        talairach_lta = subjects_dir / subject_id / 'mri' / 'transforms' / 'talairach.lta'
        cmd = f"cp {talairach_xfm_lta} {talairach_skull_lta}"
        run_cmd_with_timing(cmd)
        cmd = f"cp {talairach_xfm_lta} {talairach_lta}"
        run_cmd_with_timing(cmd)

        # Add xfm to nu
        cmd = f'mri_add_xform_to_header -c {talairach_xfm} {self.inputs.orig_nu_file} {nu_file}'
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        outputs = self._outputs().get()
        outputs["talairach_lta"] = subjects_dir / subject_id / 'mri' / 'transforms' / 'talairach.lta'
        outputs['nu_file'] = subjects_dir / subject_id / 'mri' / 'nu.mgz'
        outputs['subject_id'] = subject_id
        return outputs

    def create_sub_node(self):
        from interface.create_node_structure import create_Brainmask_node
        node = create_Brainmask_node(self.inputs.subject_id)
        return node


class NoccsegThresholdInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subjects dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    python_interpret = File(exists=True, mandatory=True, desc='the python interpret to use')
    reduce_to_aseg_py = File(exists=True, mandatory=True, desc="reduce to aseg")
    # in_file = File(exists=True, mandatory=True, desc='name of file to process. Default: aparc.DKTatlas+aseg.orig.mgz')

    # mask_file = File(mandatory=True, desc='mri/mask.mgz')
    # aseg_noCCseg_file = File(mandatory=True,
    #                          desc='Default: mri/aseg.auto_noCCseg.mgz'
    #                               'reduce labels to aseg, then create mask (dilate 5, erode 4, largest component), '
    #                               'also mask aseg to remove outliers'
    #                               'output will be uchar (else mri_cc will fail below)')  # Do not set exists=True !!


class NoccsegThresholdOutputSpec(TraitedSpec):
    mask_file = File(exists=True, desc="mask.mgz")
    aseg_noCCseg_file = File(exists=True, desc="aseg.auto_noCCseg.mgz")

    orig_file = File(exists=True, desc='mri/orig.mgz')  # orig_and_rawavg_node outputs
    aparc_DKTatlas_aseg_deep = File(exists=True, desc="mri/aparc.DKTatlas+aseg.deep.mgz")  # segment_node

    subject_id = Str(desc="subject id")


class Noccseg(BaseInterface):
    input_spec = NoccsegThresholdInputSpec
    output_spec = NoccsegThresholdOutputSpec

    time = 21 / 60  # 运行时间：分钟
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        mask_file = subjects_dir / subject_id / 'mri' / 'mask.mgz'
        aseg_noCCseg_file = subjects_dir / subject_id / 'mri' / 'aseg.auto_noCCseg.mgz'
        in_file = subjects_dir / subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
        cmd = f'{self.inputs.python_interpret} {self.inputs.reduce_to_aseg_py} ' \
              f'-i {in_file} ' \
              f'-o {aseg_noCCseg_file} --outmask {mask_file} --fixwm'
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        outputs['mask_file'] = subjects_dir / subject_id / 'mri' / 'mask.mgz'
        outputs['aseg_noCCseg_file'] = subjects_dir / subject_id / 'mri' / 'aseg.auto_noCCseg.mgz'

        outputs["orig_file"] = subjects_dir / subject_id / "mri" / "orig.mgz"  # orig_and_rawavg_node outputs
        outputs["aparc_DKTatlas_aseg_deep"] = subjects_dir / subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"  # segment_node

        outputs["subject_id"] = subject_id
        return outputs

    def create_sub_node(self):
        from interface.create_node_structure import create_N4BiasCorrect_node
        node = create_N4BiasCorrect_node(self.inputs.subject_id)
        return node


class UpdateAsegInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    python_interpret = File(exists=True, desc="python interpret", mandatory=True)
    paint_cc_file = File(exists=True, desc="FastSurfer/recon_surf/paint_cc_into_pred.py", mandatory=True)
    aseg_noCCseg_file = File(exists=True, desc="mri/aseg.auto_noCCseg.mgz", mandatory=True)
    seg_file = File(exists=True, desc="mri/aparc.DKTatlas+aseg.deep.mgz", mandatory=True)

    # aseg_auto_file = File(exists=False, desc="mri/aseg.auto.mgz", mandatory=True)
    # cc_up_file = File(exists=False, desc="mri/transforms/cc_up.lta", mandatory=True)
    # aparc_aseg_file = File(exists=False, desc="mri/aparc.DKTatlas+aseg.deep.withCC.mgz", mandatory=True)


class UpdateAsegOutputSpec(TraitedSpec):
    aseg_auto_file = File(exists=True, desc="mri/aseg.auto.mgz")
    cc_up_file = File(exists=False, desc="mri/transforms/cc_up.lta")
    aparc_aseg_file = File(exists=False, desc="mri/aparc.DKTatlas+aseg.deep.withCC.mgz")

    subject_id = Str(desc='subject id')


class UpdateAseg(BaseInterface):
    input_spec = UpdateAsegInputSpec
    output_spec = UpdateAsegOutputSpec

    time = 21 / 60  # 运行时间：分钟
    cpu = 1.6  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def _run_interface(self, runtime):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        aseg_auto_file = subjects_dir / subject_id / 'mri' / 'aseg.auto.mgz'
        cc_up_file = subjects_dir / subject_id / 'mri' / 'transforms' / 'cc_up.lta'
        aparc_aseg_file = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.withCC.mgz'

        # create aseg.auto including cc segmentation and add cc into aparc.DKTatlas+aseg.deep;
        # 46 sec: (not sure if this is needed), requires norm.mgz
        cmd = f'mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz ' \
              f'-lta {cc_up_file} {self.inputs.subject_id}'
        run_cmd_with_timing(cmd)

        # 0.8s
        cmd = f'{self.inputs.python_interpret} {self.inputs.paint_cc_file} ' \
              f'-in_cc {aseg_auto_file} -in_pred {self.inputs.seg_file} ' \
              f'-out {aparc_aseg_file}'
        run_cmd_with_timing(cmd)

        return runtime

    def _list_outputs(self):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        outputs = self._outputs().get()
        outputs["aseg_auto_file"] = subjects_dir / subject_id / 'mri' / 'aseg.auto.mgz'
        outputs["cc_up_file"] = subjects_dir / subject_id / 'mri' / 'transforms' / 'cc_up.lta'
        outputs["aparc_aseg_file"] = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.withCC.mgz'
        outputs['subject_id'] = subject_id
        return outputs

    def create_sub_node(self):
        from interface.create_node_structure import create_Filled_node
        node = create_Filled_node(self.inputs.subject_id)
        return node


class SampleSegmentationToSurfaceInputSpec(BaseInterfaceInputSpec):
    subjects_dir = Directory(exists=True, desc="subject dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    python_interpret = File(exists=True, desc="python interpret", mandatory=True)
    freesurfer_home = Directory(exists=True, desc="freesurfer_home", mandatory=True)

    lh_DKTatlaslookup_file = File(exists=True, desc="FastSurfer/recon_surf/lh.DKTatlaslookup.txt", mandatory=True)
    rh_DKTatlaslookup_file = File(exists=True, desc="FastSurfer/recon_surf/rh.DKTatlaslookup.txt", mandatory=True)
    smooth_aparc_file = File(exists=True, desc="Fastsurfer/recon_surf/smooth_aparc.py", mandatory=True)
    aparc_aseg_file = File(exists=True, desc="mri/aparc.DKTatlas+aseg.deep.withCC.mgz", mandatory=True)
    lh_white_preaparc_file = File(exists=True, desc="surf/lh.white.preaparc", mandatory=True)
    rh_white_preaparc_file = File(exists=True, desc="surf/rh.white.preaparc", mandatory=True)
    lh_cortex_label_file = File(exists=True, desc="label/lh.cortex.label", mandatory=True)
    rh_cortex_label_file = File(exists=True, desc="label/rh.cortex.label", mandatory=True)

    # lh_aparc_DKTatlas_mapped_prefix_file = File(desc="label/lh.aparc.DKTatlas.mapped.prefix.annot", mandatory=True)
    # rh_aparc_DKTatlas_mapped_prefix_file = File(desc="label/rh.aparc.DKTatlas.mapped.prefix.annot", mandatory=True)
    # lh_aparc_DKTatlas_mapped_file = File(desc="label/lh.aparc.DKTatlas.mapped.annot", mandatory=True)
    # rh_aparc_DKTatlas_mapped_file = File(desc="label/rh.aparc.DKTatlas.mapped.annot", mandatory=True)


class SampleSegmentationToSurfaceOutputSpec(TraitedSpec):
    lh_aparc_DKTatlas_mapped_prefix_file = File(exists=True, desc="label/lh.aparc.DKTatlas.mapped.prefix.annot")
    rh_aparc_DKTatlas_mapped_prefix_file = File(exists=True, desc="label/rh.aparc.DKTatlas.mapped.prefix.annot")
    lh_aparc_DKTatlas_mapped_file = File(exists=True, desc="label/lh.aparc.DKTatlas.mapped.annot")
    rh_aparc_DKTatlas_mapped_file = File(exists=True, desc="label/rh.aparc.DKTatlas.mapped.annot")


class SampleSegmentationToSurface(BaseInterface):
    input_spec = SampleSegmentationToSurfaceInputSpec
    output_spec = SampleSegmentationToSurfaceOutputSpec

    # Pool
    time = 15 / 60  # 运行时间：分钟 / 单脑测试时间
    cpu = 1  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def cmd(self, hemi):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        if hemi == 'lh':
            hemi_DKTatlaslookup_file = self.inputs.lh_DKTatlaslookup_file
            hemi_white_preaparc_file = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
            hemi_aparc_DKTatlas_mapped_prefix_file = subjects_dir / subject_id / 'label' / 'lh.aparc.DKTatlas.mapped.prefix.annot'
            hemi_cortex_label_file = subjects_dir / subject_id / "label" / "lh.cortex.label"
            hemi_aparc_DKTatlas_mapped_file = subjects_dir / subject_id / 'label' / 'lh.aparc.DKTatlas.mapped.annot'
        else:
            hemi_DKTatlaslookup_file = self.inputs.rh_DKTatlaslookup_file
            hemi_white_preaparc_file = subjects_dir / subject_id / "surf" / "rh.white.preaparc"
            hemi_aparc_DKTatlas_mapped_prefix_file = subjects_dir / subject_id / 'label' / 'rh.aparc.DKTatlas.mapped.prefix.annot'
            hemi_cortex_label_file = subjects_dir / subject_id / "label" / "rh.cortex.label"
            hemi_aparc_DKTatlas_mapped_file = subjects_dir / subject_id / 'label' / 'rh.aparc.DKTatlas.mapped.annot'

        cmd = f"mris_sample_parc -ct {self.inputs.freesurfer_home}/average/colortable_desikan_killiany.txt " \
              f"-file {hemi_DKTatlaslookup_file} -projmm 0.6 -f 5  " \
              f"-surf white.preaparc {self.inputs.subject_id} {hemi} " \
              f"aparc.DKTatlas+aseg.orig.mgz aparc.DKTatlas.mapped.prefix.annot"
        run_cmd_with_timing(cmd)
        cmd = f"{self.inputs.python_interpret} {self.inputs.smooth_aparc_file} " \
              f"--insurf {hemi_white_preaparc_file} " \
              f"--inaparc {hemi_aparc_DKTatlas_mapped_prefix_file} " \
              f"--incort {hemi_cortex_label_file} " \
              f"--outaparc {hemi_aparc_DKTatlas_mapped_file}"
        run_cmd_with_timing(cmd)

    def _run_interface(self, runtime):
        # sample input segmentation (aparc.DKTatlas+aseg orig) onto wm surface:
        # map input aparc to surface (requires thickness (and thus pail) to compute projfrac 0.5),
        # here we do projmm which allows us to compute based only on white
        # this is dangerous, as some cortices could be < 0.6 mm, but then there is no volume label probably anyway.
        # Also note that currently we cannot mask non-cortex regions here, should be done in mris_anatomical stats later
        # the smoothing helps
        multipool(self.cmd, Multi_Num=2)

        return runtime

    def _list_outputs(self):
        subjects_dir = Path(self.inputs.subjects_dir)
        subject_id = self.inputs.subject_id
        outputs = self._outputs().get()
        outputs["lh_aparc_DKTatlas_mapped_prefix_file"] = subjects_dir / subject_id / 'label' / 'lh.aparc.DKTatlas.mapped.prefix.annot'
        outputs["rh_aparc_DKTatlas_mapped_prefix_file"] = subjects_dir / subject_id / 'label' / 'rh.aparc.DKTatlas.mapped.prefix.annot'
        outputs["lh_aparc_DKTatlas_mapped_file"] = subjects_dir / subject_id / 'label' / 'lh.aparc.DKTatlas.mapped.annot'
        outputs["rh_aparc_DKTatlas_mapped_file"] = subjects_dir / subject_id / 'label' / 'rh.aparc.DKTatlas.mapped.annot'

        return outputs

    def create_sub_node(self):
        return []