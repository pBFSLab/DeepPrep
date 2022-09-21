from pathlib import Path
from nipype import Node, Workflow
from interface.freesurfer import OrigAndRawavg, Brainmask, Filled, WhitePreaparc1, \
    InflatedSphere, JacobianAvgcurvCortparc, WhitePialThickness1, Curvstats, Cortribbon, \
    Parcstats, Pctsurfcon, Hyporelabel, Aseg7ToAseg, Aseg7, BalabelsMult, Segstats
from interface.fastsurfer import Segment, Noccseg, N4BiasCorrect, TalairachAndNu, UpdateAseg, \
    SampleSegmentationToSurfave
from interface.fastcsr import FastCSR
from interface.featreg_interface import FeatReg



def init_single_structure_wf(t1w_files: list, subjects_dir: Path, subject_id: str,
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
    # updateaseg_node.inputs.paint_cc_file = Path("/home/youjia/workspace/DeepPrep/deepprep_pipeline/FastSurfer/recon_surf/paint_cc_into_pred.py")
    updateaseg_node.inputs.paint_cc_file = Path("/home/anning/workspace/DeepPrep/deepprep_pipeline/FastSurfer/recon_surf/paint_cc_into_pred.py")
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

    # white_preaparc1_node.inputs.aseg_presurf = subjects_dir / subject_id / "mri/aseg.presurf.mgz"
    # white_preaparc1_node.inputs.brain_finalsurfs = subjects_dir / subject_id / "mri/brain.finalsurfs.mgz"
    # white_preaparc1_node.inputs.wm_file = subjects_dir / subject_id / "mri/wm.mgz"
    # white_preaparc1_node.inputs.filled_file = subjects_dir / subject_id / "mri/filled.mgz"
    # white_preaparc1_node.inputs.lh_orig = subjects_dir / subject_id / f"surf/lh.orig"
    # white_preaparc1_node.inputs.rh_orig = subjects_dir / subject_id / f"surf/rh.orig"

    # SampleSegmentationToSurfave
    SampleSegmentationToSurfave_node = Node(SampleSegmentationToSurfave(), name='SampleSegmentationToSurfave_node')
    SampleSegmentationToSurfave_node.inputs.subjects_dir = subjects_dir
    SampleSegmentationToSurfave_node.inputs.subject_id = subject_id
    SampleSegmentationToSurfave_node.inputs.python_interpret = python_interpret
    SampleSegmentationToSurfave_node.inputs.freesurfer_home = freesurfer_home

    lh_DKTatlaslookup_file = fastsurfer_home / 'recon_surf' / f'lh.DKTatlaslookup.txt'
    rh_DKTatlaslookup_file = fastsurfer_home / 'recon_surf' / f'rh.DKTatlaslookup.txt'
    smooth_aparc_file = fastsurfer_home / 'recon_surf' / 'smooth_aparc.py'
    SampleSegmentationToSurfave_node.inputs.lh_DKTatlaslookup_file = lh_DKTatlaslookup_file
    SampleSegmentationToSurfave_node.inputs.rh_DKTatlaslookup_file = rh_DKTatlaslookup_file
    SampleSegmentationToSurfave_node.inputs.smooth_aparc_file = smooth_aparc_file

    # SampleSegmentationToSurfave_node.inputs.lh_white_preaparc_file = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
    # SampleSegmentationToSurfave_node.inputs.rh_white_preaparc_file = subjects_dir / subject_id / "surf" / "rh.white.preaparc"
    # SampleSegmentationToSurfave_node.inputs.lh_cortex_label_file = subjects_dir / subject_id / "label" / "lh.cortex.label"
    # SampleSegmentationToSurfave_node.inputs.rh_cortex_label_file = subjects_dir / subject_id / "label" / "rh.cortex.label"
    #
    SampleSegmentationToSurfave_node.inputs.lh_aparc_DKTatlas_mapped_prefix_file = subjects_dir / subject_id / 'label' / 'lh.aparc.DKTatlas.mapped.prefix.annot'
    SampleSegmentationToSurfave_node.inputs.rh_aparc_DKTatlas_mapped_prefix_file = subjects_dir / subject_id / 'label' / 'rh.aparc.DKTatlas.mapped.prefix.annot'
    SampleSegmentationToSurfave_node.inputs.lh_aparc_DKTatlas_mapped_file = subjects_dir / subject_id / 'label' / 'lh.aparc.DKTatlas.mapped.annot'
    SampleSegmentationToSurfave_node.inputs.rh_aparc_DKTatlas_mapped_file = subjects_dir / subject_id / 'label' / 'rh.aparc.DKTatlas.mapped.annot'

    # InflatedSphere
    inflated_sphere_node = Node(InflatedSphere(), name="inflate_sphere_node")
    inflated_sphere_node.inputs.subjects_dir = subjects_dir
    inflated_sphere_node.inputs.subject_id = subject_id
    inflated_sphere_node.inputs.threads = 8

    # FeatReg
    featreg_node = Node(FeatReg(), f'featreg_node')
    featreg_node.inputs.subjects_dir = subjects_dir
    featreg_node.inputs.subject_id = subject_id
    featreg_node.inputs.python_interpret = python_interpret
    featreg_node.inputs.freesurfer_home = freesurfer_home
    featreg_node.inputs.featreg_py = featreg_home / "featreg" / 'predict.py'

    # Jacobian
    JacobianAvgcurvCortparc_node = Node(JacobianAvgcurvCortparc(), name='JacobianAvgcurvCortparc_node')
    JacobianAvgcurvCortparc_node.inputs.subjects_dir = subjects_dir
    JacobianAvgcurvCortparc_node.inputs.subject_id = subject_id
    JacobianAvgcurvCortparc_node.inputs.threads = 8

    JacobianAvgcurvCortparc_node.inputs.lh_jacobian_white = subjects_dir / subject_id / "surf" / f"lh.jacobian_white"
    JacobianAvgcurvCortparc_node.inputs.rh_jacobian_white = subjects_dir / subject_id / "surf" / f"rh.jacobian_white"
    JacobianAvgcurvCortparc_node.inputs.lh_avg_curv = subjects_dir / subject_id / "surf" / f"lh.avg_curv"
    JacobianAvgcurvCortparc_node.inputs.rh_avg_curv = subjects_dir / subject_id / "surf" / f"rh.avg_curv"
    JacobianAvgcurvCortparc_node.inputs.lh_aparc_annot = subjects_dir / subject_id / "label" / f"lh.aparc.annot"
    JacobianAvgcurvCortparc_node.inputs.rh_aparc_annot = subjects_dir / subject_id / "label" / f"rh.aparc.annot"

    # WhitePialThickness1
    white_pial_thickness1_node = Node(WhitePialThickness1(), name='white_pial_thickness1_node')
    white_pial_thickness1_node.inputs.subjects_dir = subjects_dir
    white_pial_thickness1_node.inputs.subject_id = subject_id
    white_pial_thickness1_node.inputs.threads = 8

    white_pial_thickness1_node.inputs.lh_cortex_hipamyg_label = subjects_dir / subject_id / "label" / f"lh.cortex+hipamyg.label" ### 测试用
    white_pial_thickness1_node.inputs.rh_cortex_hipamyg_label = subjects_dir / subject_id / "label" / f"rh.cortex+hipamyg.label" ### 测试用

    white_pial_thickness1_node.inputs.lh_white = subjects_dir / subject_id / "surf" / f"lh.white"
    white_pial_thickness1_node.inputs.rh_white = subjects_dir / subject_id / "surf" / f"rh.white"

    # Curvstats
    Curvstats_node = Node(Curvstats(), name='Curvstats_node')
    Curvstats_node.inputs.subjects_dir = subjects_dir
    Curvstats_node.inputs.subject_id = subject_id

    # Cortribbon
    Cortribbon_node = Node(Cortribbon(), name='Cortribbon_node')
    Cortribbon_node.inputs.subjects_dir = subjects_dir
    Cortribbon_node.inputs.subject_id = subject_id
    Cortribbon_node.inputs.threads = 8

    # Cortribbon_node.inputs.aseg_presurf_file = subjects_dir / subject_id / 'mri/aseg.presurf.mgz' ### 测试用

    Cortribbon_node.inputs.lh_ribbon = subjects_dir / subject_id / f'mri/lh.ribbon.mgz'
    Cortribbon_node.inputs.rh_ribbon = subjects_dir / subject_id / f'mri/rh.ribbon.mgz'
    Cortribbon_node.inputs.ribbon = subjects_dir / subject_id / 'mri/ribbon.mgz'

    # Parcstats
    Parcstats_node = Node(Parcstats(), name='Parcstats_node')
    Parcstats_node.inputs.subjects_dir = subjects_dir
    Parcstats_node.inputs.subject_id = subject_id
    Parcstats_node.inputs.threads = 8

    # Parcstats_node.inputs.hemi_aparc_annot_file = subjects_dir / subject_id / 'label' / f'{hemi}.aparc.annot' ### 测试用
    # Parcstats_node.inputs.wm_file = subjects_dir / subject_id / 'mri' / 'wm.mgz' ### 测试用

    Parcstats_node.inputs.lh_aparc_stats = subjects_dir / subject_id / 'stats' / f'lh.aparc.stats'
    Parcstats_node.inputs.rh_aparc_stats = subjects_dir / subject_id / 'stats' / f'rh.aparc.stats'
    Parcstats_node.inputs.lh_aparc_pial_stats = subjects_dir / subject_id / 'stats' / f'lh.aparc.pial.stats'
    Parcstats_node.inputs.rh_aparc_pial_stats = subjects_dir / subject_id / 'stats' / f'rh.aparc.pial.stats'
    Parcstats_node.inputs.aparc_annot_ctab = subjects_dir / subject_id / 'label' / 'aparc.annot.ctab'
    Parcstats_node.inputs.aparc_annot_ctab = subjects_dir / subject_id / 'label' / 'aparc.annot.ctab'

    # Pctsurfcon
    Pctsurfcon_node = Node(Pctsurfcon(), name='Pctsurfcon_node')
    Pctsurfcon_node.inputs.subjects_dir = subjects_dir
    Pctsurfcon_node.inputs.subject_id = subject_id
    Pctsurfcon_node.inputs.threads = 8
    #
    # Pctsurfcon_node.inputs.rawavg_file = subjects_dir / subject_id / 'mri' / 'rawavg.mgz' ### 测试用
    # Pctsurfcon_node.inputs.orig_file = subjects_dir / subject_id / 'mri' / 'orig.mgz' ### 测试用
    # Pctsurfcon_node.inputs.hemi_cortex_label_file = subjects_dir / subject_id / 'label' / f'{hemi}.cortex.label' ### 测试用
    #
    # Pctsurfcon_node.inputs.hemi_wg_pct_mgh_file = subjects_dir / subject_id / 'surf' / f'{hemi}.w-g.pct.mgh'
    # Pctsurfcon_node.inputs.hemi_wg_pct_stats_file = subjects_dir / subject_id / 'stats' / f'{hemi}.w-g.pct.stats'

    # Hyporelabel
    Hyporelabel_node = Node(Hyporelabel(), name='Hyporelabel_node')
    Hyporelabel_node.inputs.subjects_dir = subjects_dir
    Hyporelabel_node.inputs.subject_id = subject_id
    Hyporelabel_node.inputs.threads = 8

    # Hyporelabel_node.inputs.aseg_presurf_file = subjects_dir / subject_id / 'mri' / 'aseg.presurf.mgz' ### 测试用
    Hyporelabel_node.inputs.aseg_presurf_hypos = subjects_dir / subject_id / 'mri' / 'aseg.presurf.hypos.mgz' ### 测试用


    # Aseg7ToAseg
    Aseg7ToAseg_node = Node(Aseg7ToAseg(), name='Aseg7ToAseg_node')
    Aseg7ToAseg_node.inputs.subjects_dir = subjects_dir
    Aseg7ToAseg_node.inputs.subject_id = subject_id
    Aseg7ToAseg_node.inputs.threads = 8

    # ##### 测试用 lh + rh
    # Aseg7ToAseg_node.inputs.lh_cortex_label_file = subjects_dir / subject_id / 'label' / 'lh.cortex.label'
    # Aseg7ToAseg_node.inputs.lh_white_file = subjects_dir / subject_id / 'surf' / 'lh.white'
    # Aseg7ToAseg_node.inputs.lh_pial_file = subjects_dir / subject_id / 'surf' / 'lh.pial'
    # Aseg7ToAseg_node.inputs.rh_cortex_label_file = subjects_dir / subject_id / 'label' / 'rh.cortex.label'
    # Aseg7ToAseg_node.inputs.rh_white_file = subjects_dir / subject_id / 'surf' / 'rh.white'
    # Aseg7ToAseg_node.inputs.rh_pial_file = subjects_dir / subject_id / 'surf' / 'rh.pial'

    Aseg7ToAseg_node.inputs.aseg_file = subjects_dir / subject_id / 'mri' / 'aseg.mgz'

    # Aseg7
    Aseg7_node = Node(Aseg7(), name='Aseg7_node')
    Aseg7_node.inputs.subjects_dir = subjects_dir
    Aseg7_node.inputs.subject_id = subject_id
    Aseg7_node.inputs.threads = 8

    Aseg7_node.inputs.subject_mri_dir = subjects_dir / subject_id / 'mri'
    Aseg7_node.inputs.aseg_presurf_hypos = subjects_dir / subject_id / 'mri' / 'aseg.presurf.hypos.mgz'

    # Aseg7_node.inputs.lh_cortex_label_file = subjects_dir / subject_id / 'label' / 'lh.cortex.label'
    # Aseg7_node.inputs.lh_white_file = subjects_dir / subject_id / 'surf' / 'lh.white'
    # Aseg7_node.inputs.lh_pial_file = subjects_dir / subject_id / 'surf' / 'lh.pial'
    # Aseg7_node.inputs.lh_aparc_annot_file = subjects_dir / subject_id / 'label' / 'lh.aparc.annot'
    # Aseg7_node.inputs.rh_cortex_label_file = subjects_dir / subject_id / 'label' / 'rh.cortex.label'
    # Aseg7_node.inputs.rh_white_file = subjects_dir / subject_id / 'surf' / 'rh.white'
    # Aseg7_node.inputs.rh_pial_file = subjects_dir / subject_id / 'surf' / 'rh.pial'
    # Aseg7_node.inputs.rh_aparc_annot_file = subjects_dir / subject_id / 'label' / 'rh.aparc.annot'

    Aseg7_node.inputs.aparc_aseg = subjects_dir / subject_id / 'mri' / 'aparc+aseg.mgz'

    # Segstats
    Segstats_node = Node(Segstats(), name='Segstats_node')
    Segstats_node.inputs.subjects_dir = subjects_dir
    Segstats_node.inputs.subject_id = subject_id
    Segstats_node.inputs.threads = 8

    # Balabels
    BalabelsMult_node = Node(BalabelsMult(), name='BalabelsMult_node')
    BalabelsMult_node.inputs.subjects_dir = subjects_dir
    BalabelsMult_node.inputs.subject_id = subject_id
    BalabelsMult_node.inputs.threads = 8

    # BalabelsMult_node.inputs.lh_sphere_reg = subjects_dir / subject_id / 'surf' / f'lh.sphere.reg'
    # BalabelsMult_node.inputs.rh_sphere_reg = subjects_dir / subject_id / 'surf' / f'rh.sphere.reg'
    
    BalabelsMult_node.inputs.freesurfer_dir = os.environ['FREESURFER']
    BalabelsMult_node.inputs.fsaverage_label_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon/fsaverage6/label')
    
    BalabelsMult_node.inputs.lh_BA45_exvivo = subjects_dir / subject_id / 'label' / f'lh.BA45_exvivo.label'
    BalabelsMult_node.inputs.rh_BA45_exvivo = subjects_dir / subject_id / 'label' / f'rh.BA45_exvivo.label'
    # BalabelsMult_node.inputs.lh_BA_exvivo_annot = subjects_dir / subject_id / 'label' / f'lh.BA_exvivo.annot'
    # BalabelsMult_node.inputs.rh_BA_exvivo_annot = subjects_dir / subject_id / 'label' / f'rh.BA_exvivo.annot'
    BalabelsMult_node.inputs.BA_exvivo_thresh = subjects_dir / subject_id / 'label' / 'BA_exvivo.thresh.ctab'
    BalabelsMult_node.inputs.lh_perirhinal_exvivo = subjects_dir / subject_id / 'label' / f'lh.perirhinal_exvivo.label'
    BalabelsMult_node.inputs.rh_perirhinal_exvivo = subjects_dir / subject_id / 'label' / f'rh.perirhinal_exvivo.label'
    BalabelsMult_node.inputs.lh_entorhinal_exvivo = subjects_dir / subject_id / 'label' / f'lh.entorhinal_exvivo.label'
    BalabelsMult_node.inputs.rh_entorhinal_exvivo = subjects_dir / subject_id / 'label' / f'rh.entorhinal_exvivo.label'


    # ############################### part 2 ###############################
    # updateaseg_node.inputs.aseg_noCCseg_file = subjects_dir / subject_id / 'mri' / 'aseg.auto_noCCseg.mgz'
    # updateaseg_node.inputs.seg_file = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'
    # updateaseg_node.outputs.aparc_aseg_file = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.withCC.mgz'
    # white_preaparc1_node.inputs.aseg_presurf = subjects_dir / subject_id / 'mri' / 'aseg.presurf.mgz'
    # white_preaparc1_node.inputs.brain_finalsurfs = subjects_dir / subject_id / 'mri' / 'brain.finalsurfs.mgz'
    # white_preaparc1_node.inputs.wm_file = subjects_dir / subject_id / 'mri' / 'wm.mgz'
    # white_preaparc1_node.inputs.filled_file = subjects_dir / subject_id / 'mri' / 'filled.mgz'
    # white_preaparc1_node.inputs.lh_orig = subjects_dir / subject_id / 'surf' / 'lh.orig'
    # white_preaparc1_node.inputs.rh_orig = subjects_dir / subject_id / 'surf' / 'rh.orig'
    # white_preaparc1_node.outputs.lh_white_preaparc = subjects_dir / subject_id / 'surf' / 'lh.white.preaparc'
    # white_preaparc1_node.outputs.rh_white_preaparc = subjects_dir / subject_id / 'surf' / 'rh.white.preaparc'
    # white_preaparc1_node.outputs.lh_cortex_label = subjects_dir / subject_id / 'label' / 'lh.cortex.label'
    # white_preaparc1_node.outputs.rh_cortex_label = subjects_dir / subject_id / 'label' / 'rh.cortex.label'
    # ############################### part 2 ###############################
    #
    # ############################### part 3 ###############################
    # filled_node.inputs.aseg_auto_file = subjects_dir / subject_id / 'mri/aseg.auto.mgz'
    # filled_node.inputs.norm_file = subjects_dir / subject_id / 'mri/norm.mgz'
    # filled_node.inputs.brainmask_file = subjects_dir / subject_id / 'mri/brainmask.mgz'
    # filled_node.inputs.talairach_lta = subjects_dir / subject_id / 'mri/transforms/talairach.lta'
    #
    # lh_white_preaparc = subjects_dir / subject_id / "surf" / f"lh.white.preaparc"
    # rh_white_preaparc = subjects_dir / subject_id / "surf" / f"rh.white.preaparc"
    # lh_sphere_reg = subjects_dir / subject_id / "surf" / f"lh.sphere.reg"
    # rh_sphere_reg = subjects_dir / subject_id / "surf" / f"rh.sphere.reg"
    # lh_jacobian_white = subjects_dir / subject_id / "surf" / f"lh.jacobian_white"
    # rh_jacobian_white = subjects_dir / subject_id / "surf" / f"rh.jacobian_white"
    # lh_avg_curv = subjects_dir / subject_id / "surf" / f"lh.avg_curv"
    # rh_avg_curv = subjects_dir / subject_id / "surf" / f"rh.avg_curv"
    # aseg_presurf_dir = subjects_dir / subject_id / "mri" / "aseg.presurf.mgz"
    # lh_cortex_label = subjects_dir / subject_id / "label" / f"lh.cortex.label"
    # rh_cortex_label = subjects_dir / subject_id / "label" / f"rh.cortex.label"
    # lh_aparc_annot = subjects_dir / subject_id / "label" / f"lh.aparc.annot"
    # rh_aparc_annot = subjects_dir / subject_id / "label" / f"rh.aparc.annot"
    #
    # JacobianAvgcurvCortparc_node.inputs.subjects_dir = subjects_dir
    # JacobianAvgcurvCortparc_node.inputs.subject_id = subject_id
    # JacobianAvgcurvCortparc_node.inputs.lh_white_preaparc = lh_white_preaparc
    # JacobianAvgcurvCortparc_node.inputs.rh_white_preaparc = rh_white_preaparc
    # JacobianAvgcurvCortparc_node.inputs.lh_sphere_reg = lh_sphere_reg
    # JacobianAvgcurvCortparc_node.inputs.rh_sphere_reg = rh_sphere_reg
    # JacobianAvgcurvCortparc_node.inputs.lh_jacobian_white = lh_jacobian_white
    # JacobianAvgcurvCortparc_node.inputs.rh_jacobian_white = rh_jacobian_white
    # JacobianAvgcurvCortparc_node.inputs.lh_avg_curv = lh_avg_curv
    # JacobianAvgcurvCortparc_node.inputs.rh_avg_curv = rh_avg_curv
    # JacobianAvgcurvCortparc_node.inputs.aseg_presurf_file = aseg_presurf_dir
    # JacobianAvgcurvCortparc_node.inputs.lh_cortex_label = lh_cortex_label
    # JacobianAvgcurvCortparc_node.inputs.rh_cortex_label = rh_cortex_label
    #
    # JacobianAvgcurvCortparc_node.inputs.lh_aparc_annot = lh_aparc_annot
    # JacobianAvgcurvCortparc_node.inputs.rh_aparc_annot = rh_aparc_annot
    # JacobianAvgcurvCortparc_node.inputs.threads = 8
    #
    # ############################### part 3 ###############################
    #
    # ############################### part -1 ###############################
    # featreg_node.inputs.lh_sulc = Path(subjects_dir) / subject_id / f'surf/lh.sulc'
    # featreg_node.inputs.rh_sulc = Path(subjects_dir) / subject_id / f'surf/rh.sulc'
    # featreg_node.inputs.lh_curv = Path(subjects_dir) / subject_id / f'surf/lh.curv'
    # featreg_node.inputs.rh_curv = Path(subjects_dir) / subject_id / f'surf/rh.curv'
    # featreg_node.inputs.lh_sphere = Path(subjects_dir) / subject_id / f'surf/lh.sphere'
    # featreg_node.inputs.rh_sphere = Path(subjects_dir) / subject_id / f'surf/rh.sphere'
    # ############################### part -1 ###############################



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
                                 (fastcsr_node, white_preaparc1_node, [("lh_orig_file", "lh_orig"), ("rh_orig_file", "rh_orig"),
                                                                        ]),

                                 (updateaseg_node, SampleSegmentationToSurfave_node, [("aparc_aseg_file", "aparc_aseg_file"),
                                                                                        ]),
                                 (white_preaparc1_node, SampleSegmentationToSurfave_node, [("lh_white_preaparc", "lh_white_preaparc_file"), ("rh_white_preaparc", "rh_white_preaparc_file"),
                                                                                            ("lh_cortex_label", "lh_cortex_label_file"), ("rh_cortex_label", "rh_cortex_label_file"),
                                                                                            ]),
                                 (white_preaparc1_node, inflated_sphere_node, [("lh_white_preaparc", "lh_white_preaparc_file"), ("rh_white_preaparc", "rh_white_preaparc_file"),
                                                                                ]),
                                 (white_preaparc1_node, featreg_node, [("lh_curv", "lh_curv"), ("rh_curv", "rh_curv"),
                                                                       ]),
                                 (inflated_sphere_node, featreg_node, [("lh_sulc", "lh_sulc"), ("rh_sulc", "rh_sulc"),
                                                                       ("lh_sphere", "lh_sphere"), ("rh_sphere", "rh_sphere"),
                                                                      ]),

                                 (white_preaparc1_node, JacobianAvgcurvCortparc_node, [("lh_white_preaparc", "lh_white_preaparc"), ("rh_white_preaparc", "rh_white_preaparc"),
                                                                                       ("lh_cortex_label", "lh_cortex_label"), ("rh_cortex_label", "rh_cortex_label"),
                                                                                        ]),
                                 (filled_node, JacobianAvgcurvCortparc_node, [("aseg_presurf_file", "aseg_presurf_file"),
                                                                                ]),
                                 (featreg_node, JacobianAvgcurvCortparc_node, [("lh_sphere_reg", "lh_sphere_reg"), ("rh_sphere_reg", "rh_sphere_reg"),
                                                                                ]),
                                 (filled_node, white_pial_thickness1_node, [("aseg_presurf_file", "aseg_presurf"), ("brain_finalsurfs_file", "brain_finalsurfs"),
                                                                            ("wm_file", "wm_file"),
                                                                               ]),
                                 (white_preaparc1_node, white_pial_thickness1_node, [("lh_white_preaparc", "lh_white_preaparc"), ("rh_white_preaparc", "rh_white_preaparc"),
                                                                                     ("lh_cortex_label", "lh_cortex_label"), ("rh_cortex_label", "rh_cortex_label"),
                                                                                    ]),
                                 (SampleSegmentationToSurfave_node, white_pial_thickness1_node, [("lh_aparc_DKTatlas_mapped_file", "lh_aparc_DKTatlas_mapped_annot"),
                                                                                                 ("rh_aparc_DKTatlas_mapped_file", "rh_aparc_DKTatlas_mapped_annot"),
                                                                                                ]),
                                 (JacobianAvgcurvCortparc_node, white_pial_thickness1_node, [("lh_aparc_annot", "lh_aparc_annot"), ("rh_aparc_annot", "rh_aparc_annot"),
                                                                                            ]),
                                 (inflated_sphere_node, Curvstats_node, [("lh_smoothwm", "lh_smoothwm"), ("rh_smoothwm", "rh_smoothwm"),
                                                                         ("lh_sulc", "lh_sulc"), ("rh_sulc", "rh_sulc"),
                                                                         # ("lh_curv", "lh_curv"), ("rh_curv", "rh_curv"),
                                                                          ]),
                                 (white_pial_thickness1_node, Curvstats_node, [("lh_curv", "lh_curv"), ("rh_curv", "rh_curv"),
                                                                               ]),
                                 (filled_node, Cortribbon_node, [("aseg_presurf_file", "aseg_presurf_file"),
                                                                ]),
                                 (white_pial_thickness1_node, Cortribbon_node, [("lh_white", "lh_white"), ("rh_white", "rh_white"),
                                                                                ("lh_pial", "lh_pial"), ("rh_pial", "rh_pial"),
                                                                                ]),
                                 (Cortribbon_node, Parcstats_node, [("ribbon", "ribbon_file"),
                                                                     ]),
                                 (filled_node, Parcstats_node, [("wm_file", "wm_file"),
                                                               ]),
                                 (JacobianAvgcurvCortparc_node, Parcstats_node, [("lh_aparc_annot", "lh_aparc_annot"), ("rh_aparc_annot", "rh_aparc_annot"),
                                                                                ]),
                                 (white_pial_thickness1_node, Parcstats_node, [("lh_white", "lh_white"), ("rh_white", "rh_white"),
                                                                               ("lh_pial", "lh_pial"), ("rh_pial", "rh_pial"),
                                                                               ("lh_thickness", "lh_thickness"), ("rh_thickness", "rh_thickness"),
                                                                               ]),
                                 (orig_and_rawavg_node, Pctsurfcon_node, [("orig_file", "orig_file"), ("rawavg_file", "rawavg_file"),
                                                                         ]),
                                 (white_preaparc1_node, Pctsurfcon_node, [("lh_cortex_label", "lh_cortex_label"), ("rh_cortex_label", "rh_cortex_label"),
                                                                         ]),
                                 (white_pial_thickness1_node, Pctsurfcon_node, [("lh_white", "lh_white"), ("rh_white", "rh_white"),
                                                                                ]),
                                 (filled_node, Hyporelabel_node, [("aseg_presurf_file", "aseg_presurf")
                                                                 ]),
                                 (white_pial_thickness1_node, Hyporelabel_node, [("lh_white", "lh_white"), ("rh_white", "rh_white"),
                                                                                ]),
                                 (white_pial_thickness1_node, Aseg7ToAseg_node, [("lh_white", "lh_white"), ("rh_white", "rh_white"),
                                                                                 ("lh_pial", "lh_pial"), ("rh_pial", "rh_pial"),
                                                                                ]),
                                 (white_preaparc1_node, Aseg7ToAseg_node, [("lh_cortex_label", "lh_cortex_label"), ("rh_cortex_label", "rh_cortex_label"),
                                                                          ]),
                                 (white_pial_thickness1_node, Aseg7_node, [("lh_white", "lh_white"), ("rh_white", "rh_white"),
                                                                            ("lh_pial", "lh_pial"), ("rh_pial", "rh_pial"),
                                                                            ]),
                                 (white_preaparc1_node, Aseg7_node, [("lh_cortex_label", "lh_cortex_label"), ("rh_cortex_label", "rh_cortex_label"),
                                                                    ]),
                                 (JacobianAvgcurvCortparc_node, Aseg7_node, [("lh_aparc_annot", "lh_aparc_annot"), ("rh_aparc_annot", "rh_aparc_annot"),
                                                                            ]),

                                 (brainmask_node, Segstats_node, [("brainmask_file", "brainmask_file"), ("norm_file", "norm_file"),
                                                                    ]),
                                 (Aseg7ToAseg_node, Segstats_node, [("aseg_file", "aseg_file"),
                                                                    ]),
                                 (filled_node, Segstats_node, [("aseg_presurf_file", "aseg_presurf"),
                                                                    ]),
                                 (white_pial_thickness1_node, Segstats_node, [("lh_white", "lh_white"), ("rh_white", "rh_white"),
                                                                              ("lh_pial", "lh_pial"), ("rh_pial", "rh_pial"),
                                                                              ]),
                                 (fastcsr_node, Segstats_node, [("lh_orig_premesh_file", "lh_orig_premesh"), ("rh_orig_premesh_file", "rh_orig_premesh"),
                                                                  ]),
                                 (Cortribbon_node, Segstats_node, [("ribbon", "ribbon_file"),
                                                                    ]),
                                 (featreg_node, BalabelsMult_node, [("lh_sphere_reg", "lh_sphere_reg"), ("rh_sphere_reg", "rh_sphere_reg"),
                                                                    ])
                                ])

    return single_structure_wf


def pipeline():
    # t1w_files = [
    #     f'/mnt/ngshare/ProjData/SurfRecon/V001/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    # ]
    # pwd = Path.cwd()
    # python_interpret = Path('/home/youjia/anaconda3/envs/3.8/bin/python3')
    # fastsurfer_home = pwd / "FastSurfer"
    # freesurfer_home = Path('/usr/local/freesurfer')
    # fastcsr_home = pwd.parent / "deepprep_pipeline/FastCSR"
    # featreg_home = pwd.parent / "deepprep_pipeline/FeatReg"
    #
    # subjects_dir = Path('/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon')
    # subject_id = 'sub-170'
    #
    # os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    #
    # wf = init_single_structure_wf(t1w_files, subjects_dir, subject_id, python_interpret, fastsurfer_home,
    #                               freesurfer_home, fastcsr_home, featreg_home)
    # wf.base_dir = subjects_dir
    # # wf.write_graph(graph2use='flat', simple_form=False)
    # wf.run()

    t1w_files = [
        f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
    ]
    pwd = Path.cwd()
    python_interpret = Path('/home/anning/miniconda3/envs/3.8/bin/python3')
    fastsurfer_home = pwd / "FastSurfer"
    freesurfer_home = Path('/usr/local/freesurfer')
    fastcsr_home = pwd.parent / "deepprep_pipeline/FastCSR"
    featreg_home = pwd.parent / "deepprep_pipeline/FeatReg"

    subjects_dir = Path('/mnt/ngshare/Data_Mirror/pipeline_test')
    subject_id = 'sub-MSC01'

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    wf = init_single_structure_wf(t1w_files, subjects_dir, subject_id, python_interpret, fastsurfer_home,
                                  freesurfer_home, fastcsr_home, featreg_home)
    wf.base_dir = f'/mnt/ngshare/Data_Mirror/pipeline_test'
    wf.write_graph(graph2use='flat', simple_form=False)
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
