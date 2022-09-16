from pathlib import Path
from nipype import Node, Workflow
from interface.freesurfer import OrigAndRawavg, Brainmask, Filled, WhitePreaparc1, \
    InflatedSphere, JacobianAvgcurvCortparc, WhitePialThickness1, Curvstats, Cortribbon, \
    Parcstats, Pctsurfcon, Hyporelabel, Aseg7ToAseg, Aseg7, Balabels
from interface.fastsurfer import Segment, Noccseg, N4BiasCorrect, TalairachAndNu, UpdateAseg, \
    SampleSegmentationToSurfave
from interface.fastcsr import FastCSR
from interface.featreg_interface import FeatReg



def init_single_structure_wf(t1w_files: list, subjects_dir: Path, subject_id: str,
                             hemi: str,
                             python_interpret: Path,
                             fastsurfer_home: Path,
                             freesurfer_home: Path,
                             fastcsr_home: Path,
                             featreg_home: Path):
    single_structure_wf = Workflow(name=f'single_structure_{subject_id}_wf')

    # orig_and_rawavg_node
    orig_and_rawavg_node = Node(OrigAndRawavg(), name='orig_and_rawavg_node')

    orig_and_rawavg_node.inputs.t1w_files = t1w_files
    orig_and_rawavg_node.inputs.subjects_dir = subjects_dir
    orig_and_rawavg_node.inputs.subject_id = subject_id
    orig_and_rawavg_node.inputs.threads = 8

    # segment_node
    fastsurfer_eval = fastsurfer_home / 'FastSurferCNN' / 'eval.py'  # inference script
    weight_dir = fastsurfer_home / 'checkpoints'  # model checkpoints dir
    network_sagittal_path = weight_dir / "Sagittal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_coronal_path = weight_dir / "Coronal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_axial_path = weight_dir / "Axial_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"

    segment_node = Node(Segment(), f'segment_node')
    segment_node.inputs.python_interpret = python_interpret
    segment_node.inputs.eval_py = fastsurfer_eval
    segment_node.inputs.network_sagittal_path = network_sagittal_path
    segment_node.inputs.network_coronal_path = network_coronal_path
    segment_node.inputs.network_axial_path = network_axial_path

    segment_node.inputs.out_file = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'
    segment_node.inputs.conformed_file = subjects_dir / subject_id / 'mri' / 'conformed.mgz'

    # auto_noccseg_node
    fastsurfer_reduce_to_aseg_py = fastsurfer_home / 'recon_surf' / 'reduce_to_aseg.py'  # inference script

    auto_noccseg_node = Node(Noccseg(), name='auto_noccseg_node')
    auto_noccseg_node.inputs.python_interpret = python_interpret
    auto_noccseg_node.inputs.reduce_to_aseg_py = fastsurfer_reduce_to_aseg_py
    # auto_noccseg_node.inputs.in_file = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'

    auto_noccseg_node.inputs.mask_file = subjects_dir / subject_id / 'mri' / 'mask.mgz'
    auto_noccseg_node.inputs.aseg_noCCseg_file = subjects_dir / subject_id / 'mri' / 'aseg.auto_noCCseg.mgz'

    # N4_bias_correct_node
    correct_py = fastsurfer_home / "recon_surf" / "N4_bias_correct.py"

    N4_bias_correct_node = Node(N4BiasCorrect(), name="N4_bias_correct_node")
    N4_bias_correct_node.inputs.threads = 8
    N4_bias_correct_node.inputs.python_interpret = python_interpret
    N4_bias_correct_node.inputs.correct_py = correct_py
    N4_bias_correct_node.inputs.orig_nu_file = subjects_dir / subject_id / "mri" / "orig_nu.mgz"

    # TalairachAndNu
    talairach_and_nu_node = Node(TalairachAndNu(), name="talairach_and_nu_node")
    talairach_and_nu_node.inputs.subjects_dir = subjects_dir
    talairach_and_nu_node.inputs.subject_id = subject_id
    talairach_and_nu_node.inputs.threads = 8

    talairach_and_nu_node.inputs.mni305 = freesurfer_home / "average" / "mni305.cor.mgz"  # atlas

    talairach_and_nu_node.inputs.talairach_lta = subjects_dir / subject_id / 'mri' / 'transforms' / 'talairach.xfm.lta'
    talairach_and_nu_node.inputs.nu_file = subjects_dir / subject_id / 'mri' / 'nu.mgz'

    # Brainmask
    brainmask_node = Node(Brainmask(), name='brainmask_node')
    brainmask_node.inputs.subjects_dir = subjects_dir
    brainmask_node.inputs.subject_id = subject_id
    brainmask_node.inputs.need_t1 = True
    brainmask_node.inputs.mask_file = subjects_dir / subject_id / 'mri' / 'mask.mgz'

    brainmask_node.inputs.T1_file = subjects_dir / subject_id / 'mri' / 'T1.mgz'
    brainmask_node.inputs.brainmask_file = subjects_dir / subject_id / 'mri' / 'brainmask.mgz'
    brainmask_node.inputs.norm_file = subjects_dir / subject_id / 'mri' / 'norm.mgz'

    # UpdateAseg
    updateaseg_node = Node(UpdateAseg(), name='updateaseg_node')
    updateaseg_node.inputs.subjects_dir = subjects_dir
    updateaseg_node.inputs.subject_id = subject_id
    updateaseg_node.inputs.paint_cc_file = Path("/home/youjia/workspace/DeepPrep/deepprep_pipeline/FastSurfer/recon_surf/paint_cc_into_pred.py")
    updateaseg_node.inputs.python_interpret = python_interpret

    updateaseg_node.inputs.aseg_auto_file = subjects_dir / subject_id / 'mri' / 'aseg.auto.mgz'
    updateaseg_node.inputs.cc_up_file = subjects_dir / subject_id / 'mri' / 'transforms' / 'cc_up.lta'
    updateaseg_node.inputs.aparc_aseg_file = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.withCC.mgz'

    # Filled
    filled_node = Node(Filled(), name='filled_node')
    filled_node.inputs.subjects_dir = subjects_dir
    filled_node.inputs.subject_id = subject_id
    filled_node.inputs.threads = 8

    # FastCSR
    fastcsr_node = Node(FastCSR(), name="fastcsr_node")
    fastcsr_node.inputs.subjects_dir = subjects_dir
    fastcsr_node.inputs.subject_id = subject_id
    fastcsr_node.inputs.python_interpret = python_interpret
    fastcsr_node.inputs.fastcsr_py = fastcsr_home / 'pipeline.py'

    # WhitePreaparc1
    white_preaparc1_node = Node(WhitePreaparc1(), name="white_preaparc1_node")
    white_preaparc1_node.inputs.subjects_dir = subjects_dir
    white_preaparc1_node.inputs.subject_id = subject_id
    white_preaparc1_node.inputs.hemi = hemi

    white_preaparc1_node.inputs.aseg_presurf = subjects_dir / subject_id / "mri/aseg.presurf.mgz"
    white_preaparc1_node.inputs.brain_finalsurfs = subjects_dir / subject_id / "mri/brain.finalsurfs.mgz"
    white_preaparc1_node.inputs.wm_file = subjects_dir / subject_id / "mri/wm.mgz"
    white_preaparc1_node.inputs.filled_file = subjects_dir / subject_id / "mri/filled.mgz"
    white_preaparc1_node.inputs.hemi_orig = subjects_dir / subject_id / f"surf/{hemi}.orig"

    # SampleSegmentationToSurfave
    SampleSegmentationToSurfave_node = Node(SampleSegmentationToSurfave(), name='SampleSegmentationToSurfave_node')
    SampleSegmentationToSurfave_node.inputs.subjects_dir = subjects_dir
    SampleSegmentationToSurfave_node.inputs.subject_id = subject_id
    SampleSegmentationToSurfave_node.inputs.python_interpret = python_interpret
    SampleSegmentationToSurfave_node.inputs.freesurfer_home = freesurfer_home
    SampleSegmentationToSurfave_node.inputs.hemi = hemi

    hemi_DKTatlaslookup_file = fastsurfer_home / 'recon_surf' / f'{hemi}.DKTatlaslookup.txt'
    smooth_aparc_file = fastsurfer_home / 'recon_surf' / 'smooth_aparc.py'
    SampleSegmentationToSurfave_node.inputs.hemi_DKTatlaslookup_file = hemi_DKTatlaslookup_file
    SampleSegmentationToSurfave_node.inputs.smooth_aparc_file = smooth_aparc_file

    SampleSegmentationToSurfave_node.inputs.hemi_aparc_DKTatlas_mapped_prefix_file = subjects_dir / 'label' / f'{hemi}.aparc.DKTatlas.mapped.prefix.annot'
    SampleSegmentationToSurfave_node.inputs.hemi_aparc_DKTatlas_mapped_file = subjects_dir / 'label' / f'{hemi}.aparc.DKTatlas.mapped.annot'

    # InflatedSphere
    inflated_sphere_node = Node(InflatedSphere(), name="inflate_sphere_node")
    inflated_sphere_node.inputs.subjects_dir = subjects_dir
    inflated_sphere_node.inputs.subject_id = subject_id
    inflated_sphere_node.inputs.hemi = hemi
    inflated_sphere_node.inputs.threads = 8

    # FeatReg
    featreg_node = Node(FeatReg(), f'featreg_node')
    featreg_node.inputs.subjects_dir = subjects_dir
    featreg_node.inputs.subject_id = subject_id
    featreg_node.inputs.python_interpret = python_interpret
    featreg_node.inputs.freesurfer_home = freesurfer_home
    featreg_node.inputs.featreg_py = featreg_home / "featreg" / 'predict.py'

    featreg_node.inputs.hemisphere = hemi

    # Jacobian
    JacobianAvgcurvCortparc_node = Node(JacobianAvgcurvCortparc(), name='JacobianAvgcurvCortparc_node')
    JacobianAvgcurvCortparc_node.inputs.subjects_dir = subjects_dir
    JacobianAvgcurvCortparc_node.inputs.subject_id = subject_id
    JacobianAvgcurvCortparc_node.inputs.hemi = hemi
    JacobianAvgcurvCortparc_node.inputs.threads = 8

    JacobianAvgcurvCortparc_node.inputs.jacobian_white_file = subjects_dir / subject_id / "surf" / f"{hemi}.jacobian_white"
    JacobianAvgcurvCortparc_node.inputs.avg_curv_file = subjects_dir / subject_id / "surf" / f"{hemi}.avg_curv"
    JacobianAvgcurvCortparc_node.inputs.aparc_annot_file = subjects_dir / subject_id / "label" / f"{hemi}.aparc.annot"



    # WhitePialThickness1
    white_pial_thickness1_node = Node(WhitePialThickness1(), name='white_pial_thickness1_node')
    white_pial_thickness1_node.inputs.subjects_dir = subjects_dir
    white_pial_thickness1_node.inputs.subject_id = subject_id
    white_pial_thickness1_node.inputs.hemi = hemi
    white_pial_thickness1_node.inputs.threads = 8

    white_pial_thickness1_node.inputs.hemi_cortex_hipamyg_label = subjects_dir / subject_id / "label" / f"{hemi}.cortex+hipamyg.label" ### 测试用

    white_pial_thickness1_node.inputs.hemi_white = subjects_dir / subject_id / "surf" / f"{hemi}.white"


    # Curvstats
    Curvstats_node = Node(Curvstats(), name='Curvstats_node')
    Curvstats_node.inputs.subjects_dir = subjects_dir
    Curvstats_node.inputs.subject_id = subject_id
    Curvstats_node.inputs.hemi = hemi

    # Cortribbon
    Cortribbon_node = Node(Cortribbon(), name='Cortribbon_node')
    Cortribbon_node.inputs.subjects_dir = subjects_dir
    Cortribbon_node.inputs.subject_id = subject_id
    Cortribbon_node.inputs.hemi = hemi
    Cortribbon_node.inputs.threads = 8

    Cortribbon_node.inputs.aseg_presurf_file = subjects_dir / subject_id / 'mri/aseg.presurf.mgz' ### 测试用

    Cortribbon_node.inputs.hemi_ribbon = subjects_dir / subject_id / f'mri/{hemi}.ribbon.mgz'
    Cortribbon_node.inputs.ribbon = subjects_dir / subject_id / 'mri/ribbon.mgz'

    # Parcstats
    Parcstats_node = Node(Parcstats(), name='Parcstats_node')
    Parcstats_node.inputs.subjects_dir = subjects_dir
    Parcstats_node.inputs.subject_id = subject_id
    Parcstats_node.inputs.hemi = hemi
    Parcstats_node.inputs.threads = 8

    Parcstats_node.inputs.hemi_aparc_annot_file = subjects_dir / subject_id / 'label' / f'{hemi}.aparc.annot' ### 测试用
    Parcstats_node.inputs.wm_file = subjects_dir / subject_id / 'mri' / 'wm.mgz' ### 测试用

    Parcstats_node.inputs.hemi_aparc_stats_file = subjects_dir / subject_id / 'stats' / f'{hemi}.aparc.stats'
    Parcstats_node.inputs.hemi_aparc_pial_stats_file = subjects_dir / subject_id / 'stats' / f'{hemi}.aparc.pial.stats'
    Parcstats_node.inputs.aparc_annot_ctab_file = subjects_dir / subject_id / 'label' / 'aparc.annot.ctab'

    # Pctsurfcon
    Pctsurfcon_node = Node(Pctsurfcon(), name='Pctsurfcon_node')
    Pctsurfcon_node.inputs.subjects_dir = subjects_dir
    Pctsurfcon_node.inputs.subject_id = subject_id
    Pctsurfcon_node.inputs.hemi = hemi
    Pctsurfcon_node.inputs.threads = 8


    Pctsurfcon_node.inputs.rawavg_file = subjects_dir / subject_id / 'mri' / 'rawavg.mgz' ### 测试用
    Pctsurfcon_node.inputs.orig_file = subjects_dir / subject_id / 'mri' / 'orig.mgz' ### 测试用
    Pctsurfcon_node.inputs.hemi_cortex_label_file = subjects_dir / subject_id / 'label' / f'{hemi}.cortex.label' ### 测试用

    Pctsurfcon_node.inputs.hemi_wg_pct_mgh_file = subjects_dir / subject_id / 'surf' / f'{hemi}.w-g.pct.mgh'
    Pctsurfcon_node.inputs.hemi_wg_pct_stats_file = subjects_dir / subject_id / 'stats' / f'{hemi}.w-g.pct.stats'

    # Hyporelabel
    Hyporelabel_node = Node(Hyporelabel(), name='Hyporelabel_node')
    Hyporelabel_node.inputs.subjects_dir = subjects_dir
    Hyporelabel_node.inputs.subject_id = subject_id
    Hyporelabel_node.inputs.hemi = hemi
    Hyporelabel_node.inputs.threads = 8

    Hyporelabel_node.inputs.aseg_presurf_file = subjects_dir / subject_id / 'mri' / 'aseg.presurf.mgz' ### 测试用
    Hyporelabel_node.inputs.aseg_presurf_hypos_file = subjects_dir / subject_id / 'mri' / 'aseg.presurf.hypos.mgz' ### 测试用


    # Aseg7ToAseg
    Aseg7ToAseg_node = Node(Aseg7ToAseg(), name='Aseg7_node')
    Aseg7ToAseg_node.inputs.subjects_dir = subjects_dir
    Aseg7ToAseg_node.inputs.subject_id = subject_id
    Aseg7ToAseg_node.inputs.threads = 8

    ##### 测试用 lh + rh
    Aseg7ToAseg_node.inputs.lh_cortex_label_file = subjects_dir / subject_id / 'label' / 'lh.cortex.label'
    Aseg7ToAseg_node.inputs.lh_white_file = subjects_dir / subject_id / 'surf' / 'lh.white'
    Aseg7ToAseg_node.inputs.lh_pial_file = subjects_dir / subject_id / 'surf' / 'lh.pial'
    Aseg7ToAseg_node.inputs.rh_cortex_label_file = subjects_dir / subject_id / 'label' / 'rh.cortex.label'
    Aseg7ToAseg_node.inputs.rh_white_file = subjects_dir / subject_id / 'surf' / 'rh.white'
    Aseg7ToAseg_node.inputs.rh_pial_file = subjects_dir / subject_id / 'surf' / 'rh.pial'

    Aseg7ToAseg_node.inputs.aseg_file = subjects_dir / subject_id / 'mri' / 'aseg.mgz'

    # Aseg7
    Aseg7_node = Node(Aseg7(), name='Aseg7_node')
    Aseg7_node.inputs.subjects_dir = subjects_dir
    Aseg7_node.inputs.subject_id = subject_id
    Aseg7_node.inputs.threads = 8

    Aseg7_node.inputs.subject_mri_dir = subjects_dir / subject_id / 'mri'
    Aseg7_node.inputs.aseg_presurf_hypos_file = subjects_dir / subject_id / 'mri' / 'aseg.presurf.hypos.mgz'
    Aseg7_node.inputs.lh_cortex_label_file = subjects_dir / subject_id / 'label' / 'lh.cortex.label'
    Aseg7_node.inputs.lh_white_file = subjects_dir / subject_id / 'surf' / 'lh.white'
    Aseg7_node.inputs.lh_pial_file = subjects_dir / subject_id / 'surf' / 'lh.pial'
    Aseg7_node.inputs.lh_aparc_annot_file = subjects_dir / subject_id / 'label' / 'lh.aparc.annot'
    Aseg7_node.inputs.rh_cortex_label_file = subjects_dir / subject_id / 'label' / 'rh.cortex.label'
    Aseg7_node.inputs.rh_white_file = subjects_dir / subject_id / 'surf' / 'rh.white'
    Aseg7_node.inputs.rh_pial_file = subjects_dir / subject_id / 'surf' / 'rh.pial'
    Aseg7_node.inputs.rh_aparc_annot_file = subjects_dir / subject_id / 'label' / 'rh.aparc.annot'

    Aseg7_node.inputs.aparc_aseg_file = subjects_dir / subject_id / 'mri' / 'aparc+aseg.mgz'

    # Balabels
    Balabels_node = Node(Balabels(), name='Balabels_node')
    Balabels_node.inputs.subjects_dir = subjects_dir
    Balabels_node.inputs.subject_id = subject_id
    Balabels_node.inputs.hemi = hemi
    Balabels_node.inputs.threads = 8
    Balabels_node.inputs.hemi_sphere_file = subjects_dir / subject_id / 'surf' / f'{hemi}.sphere.reg'

    Balabels_node.inputs.hemi_BA45_exvivo_file = subjects_dir / subject_id / 'label' / f'{hemi}.BA45_exvivo.label'
    Balabels_node.inputs.hemi_BA_exvivo_annot_file = subjects_dir / subject_id / 'label' / f'{hemi}.BA_exvivo.annot'
    Balabels_node.inputs.BA_exvivo_thresh_file = subjects_dir / subject_id / 'label' / 'BA_exvivo.thresh.ctab'
    Balabels_node.inputs.hemi_perirhinal_exvivo_file = subjects_dir / subject_id / 'label' / f'{hemi}.perirhinal_exvivo.label'
    Balabels_node.inputs.hemi_entorhinal_exvivo_file = subjects_dir / subject_id / 'label' / f'{hemi}.entorhinal_exvivo.label'

    # create workflow
    single_structure_wf.connect([
                                 (orig_and_rawavg_node, segment_node, [("orig_file", "in_file"),
                                                                       ]),
                                 (segment_node, auto_noccseg_node, [("aseg_deep_file", "in_file"),
                                                                    ]),
                                 (orig_and_rawavg_node, N4_bias_correct_node, [("orig_file", "orig_file"),
                                                                               ]),
                                 (auto_noccseg_node, N4_bias_correct_node, [("mask_file", "mask_file"),
                                                                            ]),
                                 (orig_and_rawavg_node, talairach_and_nu_node, [("orig_file", "orig_file"),
                                                                                ]),
                                 (N4_bias_correct_node, talairach_and_nu_node, [("orig_nu_file", "orig_nu_file"),
                                                                                ]),
                                 (talairach_and_nu_node, brainmask_node, [("nu_file", "nu_file"),
                                                                        ]),
                                 (segment_node, updateaseg_node, [("aseg_deep_file", "seg_file"),
                                                                  ]),
                                 (auto_noccseg_node, updateaseg_node, [("aseg_noCCseg_file", "aseg_noCCseg_file"),
                                                                          ]),
                                 (updateaseg_node, filled_node, [("aseg_auto_file", "aseg_auto_file"),
                                                                    ]),
                                 (brainmask_node, filled_node, [("brainmask_file", "brainmask_file"), ("norm_file", "norm_file"),
                                                                 ]),
                                 (talairach_and_nu_node, filled_node, [("talairach_lta", "talairach_lta"),
                                                                        ]),
                                 (orig_and_rawavg_node, fastcsr_node, [("orig_file", "orig_file"),
                                                                       ]),
                                 (brainmask_node, fastcsr_node, [("brainmask_file", "brainmask_file"),
                                                                ]),
                                 (filled_node, fastcsr_node, [("aseg_presurf_file", "aseg_presurf_file"), ("wm_filled", "filled_file"),
                                                              ("brain_finalsurfs_file", "brain_finalsurfs_file"), ("wm_file", "wm_file"),
                                                                 ]),
                                 (filled_node, white_preaparc1_node, [("aseg_presurf_file", "aseg_presurf"), ("brain_finalsurfs_file", "brain_finalsurfs"),
                                                                      ("wm_file", "wm_file"), ("wm_filled", "filled_file"),
                                                                        ]),
                                 (fastcsr_node, white_preaparc1_node, [("lh_orig_file", "hemi_orig"),
                                                                        ]),
                                 (updateaseg_node, SampleSegmentationToSurfave_node, [("aparc_aseg_file", "aparc_aseg_file"),
                                                                                        ]),
                                 (white_preaparc1_node, SampleSegmentationToSurfave_node, [("hemi_white_preaparc", "hemi_white_preaparc_file"),
                                                                                            ("hemi_cortex_label", "hemi_cortex_label_file"),
                                                                                            ]),
                                 (white_preaparc1_node, inflated_sphere_node, [("hemi_white_preaparc", "white_preaparc_file"),
                                                                                ]),
                                 (white_preaparc1_node, featreg_node, [("hemi_curv", "curv_file"),
                                                                       ]),
                                 (inflated_sphere_node, featreg_node, [("sulc_file", "sulc_file"), ("hemi_sphere", "sphere_file"),
                                                                      ]),
                                 (white_preaparc1_node, JacobianAvgcurvCortparc_node, [("hemi_white_preaparc", "white_preaparc_file"),
                                                                                       ("hemi_cortex_label", "cortex_label_file"),
                                                                                        ]),
                                 (filled_node, JacobianAvgcurvCortparc_node, [("aseg_presurf_file", "aseg_presurf_file"),
                                                                                ]),
                                 (featreg_node, JacobianAvgcurvCortparc_node, [("sphere_reg_file", "sphere_reg_file"),
                                                                                ]),
                                 (filled_node, white_pial_thickness1_node, [("aseg_presurf_file", "aseg_presurf"), ("brain_finalsurfs_file", "brain_finalsurfs"),
                                                                               ]),
                                 (white_preaparc1_node, white_pial_thickness1_node, [("hemi_white_preaparc", "hemi_white_preaparc"),
                                                                                     ("hemi_cortex_label", "hemi_cortex_label"),
                                                                                    ]),
                                 (SampleSegmentationToSurfave_node, white_pial_thickness1_node, [("hemi_aparc_DKTatlas_mapped_file", "hemi_aparc_DKTatlas_mapped_annot"),
                                                                                                ]),
                                 (JacobianAvgcurvCortparc_node, white_pial_thickness1_node, [("aparc_annot_file", "hemi_aparc_annot"),
                                                                                            ]),
                                 (white_pial_thickness1_node, Cortribbon_node, [("hemi_white", "hemi_white"), ("hemi_pial", "hemi_pial"),
                                                                                 ]),
                                 (Cortribbon_node, Parcstats_node, [("ribbon", "ribbon_file"),
                                                                     ]),
                                 (white_pial_thickness1_node, Parcstats_node, [("hemi_white", "hemi_white_file"), ("hemi_pial", "hemi_pial_file"),
                                                                               ("hemi_thickness", "hemi_thickness_file"),
                                                                               ]),
                                 (white_pial_thickness1_node, Pctsurfcon_node, [("hemi_white", "hemi_white_file"),
                                                                                ]),
                                 (white_pial_thickness1_node, Hyporelabel_node, [("hemi_white", "hemi_white_file"),
                                                                                ]),

                                 ])

    return single_structure_wf


def pipeline():
    t1w_files = [
        f'/mnt/ngshare/ProjData/SurfRecon/V001/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    ]
    pwd = Path.cwd()
    python_interpret = Path('/home/youjia/anaconda3/envs/3.8/bin/python3')
    fastsurfer_home = pwd / "FastSurfer"
    freesurfer_home = Path('/usr/local/freesurfer')
    fastcsr_home = pwd.parent / "deepprep_pipeline/FastCSR"
    featreg_home = pwd.parent / "deepprep_pipeline/FeatReg"

    subjects_dir = Path('/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon')
    subject_id = 'sub-001'

    hemi = 'lh'

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    wf = init_single_structure_wf(t1w_files, subjects_dir, subject_id, hemi, python_interpret, fastsurfer_home,
                                  freesurfer_home, fastcsr_home, featreg_home)
    wf.base_dir = subjects_dir
    # wf.write_graph(graph2use='flat', simple_form=False)
    wf.run()


if __name__ == '__main__':
    import os


    def set_envrion(threads: int = 1):
        # FreeSurfer recon-all env
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
        os.environ['FREESURFER'] = '/usr/local/freesurfer'
        os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
        os.environ['PATH'] = '/usr/local/freesurfer/bin:/usr/local/freesurfer/mni/bin:/usr/local/freesurfer/tktools:' + \
                             '/usr/local/freesurfer/fsfast/bin:' + os.environ['PATH']
        os.environ['MINC_BIN_DIR'] = '/usr/local/freesurfer/mni/bin'
        os.environ['MINC_LIB_DIR'] = '/usr/local/freesurfer/mni/lib'
        os.environ['PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'
        os.environ['MNI_PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'
        # FreeSurfer fsfast env
        os.environ['FSF_OUTPUT_FORMAT'] = 'nii.gz'
        os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
        # FSL
        os.environ['PATH'] = '/usr/local/fsl/bin:' + os.environ['PATH']
        # ANFI
        os.environ['PATH'] = '/home/anning/abin:' + os.environ['PATH']
        # ANTs
        os.environ['PATH'] = '/usr/local/ANTs/bin:' + os.environ['PATH']
        # Convert3D
        os.environ['PATH'] = '/usr/local/c3d-1.1.0-Linux-x86_64/bin:' + os.environ['PATH']


    set_envrion()
    pipeline()
