from interface.freesurfer_node import *
from interface.bold_node import *
from interface.fastcsr_node import *
from interface.fastsurfer_node import *
from interface.featreg_node import *
from interface.node_source import Source

from nipype import Node

"""环境变量
subjects_dir = Path(os.environ['SUBJECTS_DIR'])
bold_preprocess_dir = Path(os.environ['BOLD_PREPROCESS_DIR'])
workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
fastsurfer_home = Path(os.environ['FASTSURFER_HOME'])
freesurfer_home = Path(os.environ['FREESURFER_HOME'])
fastcsr_home = Path(os.environ['FASTCSR_HOME'])
featreg_home = Path(os.environ['FEATREG_HOME'])
python_interpret = sys.executable
"""


def create_origandrawavg_node(subject_id: str, t1w_files: list):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = os.environ['WORKFLOW_CACHED_DIR']

    origandrawavg_node = Node(OrigAndRawavg(), f'{subject_id}_origandrawavg_node')
    origandrawavg_node.inputs.t1w_files = t1w_files
    origandrawavg_node.inputs.subjects_dir = subjects_dir
    origandrawavg_node.inputs.subject_id = subject_id
    origandrawavg_node.inputs.threads = 1

    origandrawavg_node.base_dir = workflow_cached_dir
    origandrawavg_node.source = Source(CPU_n=1)

    return origandrawavg_node


def create_Segment_node(subject_id: str):

    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    fastsurfer_home = Path(os.environ['FASTSURFER_HOME'])
    workflow_cached_dir = os.environ['WORKFLOW_CACHED_DIR']
    python_interpret = sys.executable

    fastsurfer_eval = fastsurfer_home / 'FastSurferCNN' / 'eval.py'  # inference script
    weight_dir = fastsurfer_home / 'checkpoints'  # model checkpoints dir

    network_sagittal_path = weight_dir / "Sagittal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_coronal_path = weight_dir / "Coronal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_axial_path = weight_dir / "Axial_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"

    segment_node = Node(Segment(), f'{subject_id}_segment_node')
    segment_node.inputs.subjects_dir = subjects_dir
    segment_node.inputs.subject_id = subject_id
    segment_node.inputs.python_interpret = python_interpret
    segment_node.inputs.eval_py = fastsurfer_eval
    segment_node.inputs.network_sagittal_path = network_sagittal_path
    segment_node.inputs.network_coronal_path = network_coronal_path
    segment_node.inputs.network_axial_path = network_axial_path

    segment_node.base_dir = workflow_cached_dir
    segment_node.source = Source(CPU_n=1, GPU_MB=8000)

    return segment_node


def create_Noccseg_node(subject_id: str):
    fastsurfer_home = Path(os.environ['FASTSURFER_HOME'])
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable

    reduce_to_aseg_py = fastsurfer_home / 'recon_surf' / 'reduce_to_aseg.py'

    noccseg_node = Node(Noccseg(), f'{subject_id}_noccseg_node')
    noccseg_node.inputs.python_interpret = python_interpret
    noccseg_node.inputs.reduce_to_aseg_py = reduce_to_aseg_py
    noccseg_node.inputs.subject_id = subject_id
    noccseg_node.inputs.subjects_dir = subjects_dir

    noccseg_node.base_dir = workflow_cached_dir
    noccseg_node.source = Source()
    return noccseg_node


def create_InflatedSphere_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    lh_white_preaparc_file = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
    rh_white_preaparc_file = subjects_dir / subject_id / "surf" / "rh.white.preaparc"

    Inflated_Sphere_node = Node(InflatedSphere(), f'{subject_id}_Inflated_Sphere_node')
    Inflated_Sphere_node.inputs.threads = 8
    Inflated_Sphere_node.inputs.subjects_dir = subjects_dir
    Inflated_Sphere_node.inputs.subject_id = subject_id
    Inflated_Sphere_node.inputs.lh_white_preaparc_file = lh_white_preaparc_file
    Inflated_Sphere_node.inputs.rh_white_preaparc_file = rh_white_preaparc_file
    Inflated_Sphere_node.base_dir = workflow_cached_dir
    Inflated_Sphere_node.source = Source()

    return Inflated_Sphere_node


def create_FeatReg_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    featreg_home = Path(os.environ["FEATREG_HOME"])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    python_interpret = sys.executable
    featreg_py = featreg_home / "featreg" / 'predict.py'  # inference script

    featreg_node = Node(FeatReg(), f'{subject_id}_featreg_node')
    featreg_node.inputs.featreg_py = featreg_py
    featreg_node.inputs.python_interpret = python_interpret

    featreg_node.inputs.subjects_dir = subjects_dir
    featreg_node.inputs.subject_id = subject_id
    featreg_node.inputs.freesurfer_home = '/usr/local/freesurfer'
    featreg_node.inputs.lh_sulc = Path(subjects_dir) / subject_id / f'surf/lh.sulc'
    featreg_node.inputs.rh_sulc = Path(subjects_dir) / subject_id / f'surf/rh.sulc'
    featreg_node.inputs.lh_curv = Path(subjects_dir) / subject_id / f'surf/lh.curv'
    featreg_node.inputs.rh_curv = Path(subjects_dir) / subject_id / f'surf/rh.curv'
    featreg_node.inputs.lh_sphere = Path(subjects_dir) / subject_id / f'surf/lh.sphere'
    featreg_node.inputs.rh_sphere = Path(subjects_dir) / subject_id / f'surf/rh.sphere'
    featreg_node.base_dir = workflow_cached_dir
    featreg_node.source = Source()

    return featreg_node


def create_JacobianAvgcurvCortparc_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    lh_white_preaparc = subjects_dir / subject_id / "surf" / f"lh.white.preaparc"
    rh_white_preaparc = subjects_dir / subject_id / "surf" / f"rh.white.preaparc"
    lh_sphere_reg = subjects_dir / subject_id / "surf" / f"lh.sphere.reg"
    rh_sphere_reg = subjects_dir / subject_id / "surf" / f"rh.sphere.reg"
    lh_jacobian_white = subjects_dir / subject_id / "surf" / f"lh.jacobian_white"
    rh_jacobian_white = subjects_dir / subject_id / "surf" / f"rh.jacobian_white"
    lh_avg_curv = subjects_dir / subject_id / "surf" / f"lh.avg_curv"
    rh_avg_curv = subjects_dir / subject_id / "surf" / f"rh.avg_curv"
    aseg_presurf_dir = subjects_dir / subject_id / "mri" / "aseg.presurf.mgz"
    lh_cortex_label = subjects_dir / subject_id / "label" / f"lh.cortex.label"
    rh_cortex_label = subjects_dir / subject_id / "label" / f"rh.cortex.label"
    lh_aparc_annot = subjects_dir / subject_id / "label" / f"lh.aparc.annot"
    rh_aparc_annot = subjects_dir / subject_id / "label" / f"rh.aparc.annot"

    JacobianAvgcurvCortparc_node = Node(JacobianAvgcurvCortparc(), f'{subject_id}_JacobianAvgcurvCortparc_node')
    JacobianAvgcurvCortparc_node.inputs.subjects_dir = subjects_dir
    JacobianAvgcurvCortparc_node.inputs.subject_id = subject_id
    JacobianAvgcurvCortparc_node.inputs.lh_white_preaparc = lh_white_preaparc
    JacobianAvgcurvCortparc_node.inputs.rh_white_preaparc = rh_white_preaparc
    JacobianAvgcurvCortparc_node.inputs.lh_sphere_reg = lh_sphere_reg
    JacobianAvgcurvCortparc_node.inputs.rh_sphere_reg = rh_sphere_reg
    JacobianAvgcurvCortparc_node.inputs.lh_jacobian_white = lh_jacobian_white
    JacobianAvgcurvCortparc_node.inputs.rh_jacobian_white = rh_jacobian_white
    JacobianAvgcurvCortparc_node.inputs.lh_avg_curv = lh_avg_curv
    JacobianAvgcurvCortparc_node.inputs.rh_avg_curv = rh_avg_curv
    JacobianAvgcurvCortparc_node.inputs.aseg_presurf_file = aseg_presurf_dir
    JacobianAvgcurvCortparc_node.inputs.lh_cortex_label = lh_cortex_label
    JacobianAvgcurvCortparc_node.inputs.rh_cortex_label = rh_cortex_label

    JacobianAvgcurvCortparc_node.inputs.lh_aparc_annot = lh_aparc_annot
    JacobianAvgcurvCortparc_node.inputs.rh_aparc_annot = rh_aparc_annot
    JacobianAvgcurvCortparc_node.inputs.threads = 8
    JacobianAvgcurvCortparc_node.base_dir = workflow_cached_dir
    JacobianAvgcurvCortparc_node.source = Source()

    return JacobianAvgcurvCortparc_node


def create_WhitePialThickness1_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    threads = 8

    white_pial_thickness1 = Node(WhitePialThickness1(), name=f'{subject_id}_white_pial_thickness1')
    white_pial_thickness1.inputs.subjects_dir = subjects_dir
    white_pial_thickness1.inputs.subject_id = subject_id
    white_pial_thickness1.inputs.threads = threads

    white_pial_thickness1.inputs.lh_white_preaparc = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
    white_pial_thickness1.inputs.rh_white_preaparc = subjects_dir / subject_id / "surf" / "rh.white.preaparc"
    white_pial_thickness1.inputs.aseg_presurf = subjects_dir / subject_id / "mri" / "aseg.presurf.mgz"
    white_pial_thickness1.inputs.brain_finalsurfs = subjects_dir / subject_id / "mri" / "brain.finalsurfs.mgz"
    white_pial_thickness1.inputs.wm_file = subjects_dir / subject_id / "mri" / "wm.mgz"
    white_pial_thickness1.inputs.lh_aparc_annot = subjects_dir / subject_id / "label" / "lh.aparc.annot"
    white_pial_thickness1.inputs.rh_aparc_annot = subjects_dir / subject_id / "label" / "rh.aparc.annot"
    white_pial_thickness1.inputs.lh_cortex_hipamyg_label = subjects_dir / subject_id / "label" / "lh.cortex+hipamyg.label"
    white_pial_thickness1.inputs.rh_cortex_hipamyg_label = subjects_dir / subject_id / "label" / "rh.cortex+hipamyg.label"
    white_pial_thickness1.inputs.lh_cortex_label = subjects_dir / subject_id / "label" / "lh.cortex.label"
    white_pial_thickness1.inputs.rh_cortex_label = subjects_dir / subject_id / "label" / "rh.cortex.label"

    white_pial_thickness1.inputs.lh_white = subjects_dir / subject_id / "surf" / "lh.white"
    white_pial_thickness1.inputs.rh_white = subjects_dir / subject_id / "surf" / "rh.white"
    white_pial_thickness1.base_dir = workflow_cached_dir
    white_pial_thickness1.source = Source()

    return white_pial_thickness1


def create_Curvstats_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    subject_surf_dir = subjects_dir / "surf"
    threads = 8

    Curvstats_node = Node(Curvstats(), name=f'{subject_id}_Curvstats_node')
    Curvstats_node.inputs.subjects_dir = subjects_dir
    Curvstats_node.inputs.subject_id = subject_id

    Curvstats_node.inputs.lh_smoothwm = subject_surf_dir / f'lh.smoothwm'
    Curvstats_node.inputs.rh_smoothwm = subject_surf_dir / f'rh.smoothwm'
    Curvstats_node.inputs.lh_curv = subject_surf_dir / f'lh.curv'
    Curvstats_node.inputs.rh_curv = subject_surf_dir / f'rh.curv'
    Curvstats_node.inputs.lh_sulc = subject_surf_dir / f'lh.sulc'
    Curvstats_node.inputs.rh_sulc = subject_surf_dir / f'rh.sulc'
    Curvstats_node.inputs.threads = threads
    Curvstats_node.base_dir = workflow_cached_dir
    Curvstats_node.source = Source()

    return Curvstats_node


def create_BalabelsMult_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    threads = 10

    BalabelsMult_node = Node(BalabelsMult(), name=f'{subject_id}_BalabelsMult_node')
    BalabelsMult_node.inputs.subjects_dir = subjects_dir
    BalabelsMult_node.inputs.subject_id = subject_id
    BalabelsMult_node.inputs.threads = threads
    BalabelsMult_node.inputs.freesurfer_dir = os.environ['FREESURFER_HOME']

    BalabelsMult_node.inputs.lh_sphere_reg = subject_surf_dir / f'lh.sphere.reg'
    BalabelsMult_node.inputs.rh_sphere_reg = subject_surf_dir / f'rh.sphere.reg'
    BalabelsMult_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    BalabelsMult_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    BalabelsMult_node.inputs.fsaverage_label_dir = Path(os.environ['FREESURFER_HOME']) / "subjects/fsaverage/label"
    BalabelsMult_node.base_dir = workflow_cached_dir
    BalabelsMult_node.source = Source()

    return BalabelsMult_node


def create_Cortribbon_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    threads = 8

    Cortribbon_node = Node(Cortribbon(), name=f'{subject_id}_Cortribbon_node')
    Cortribbon_node.inputs.subjects_dir = subjects_dir
    Cortribbon_node.inputs.subject_id = subject_id
    Cortribbon_node.inputs.threads = threads

    Cortribbon_node.inputs.aseg_presurf_file = subject_mri_dir / 'aseg.presurf.mgz'
    Cortribbon_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Cortribbon_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    Cortribbon_node.inputs.lh_pial = subject_surf_dir / f'lh.pial'
    Cortribbon_node.inputs.rh_pial = subject_surf_dir / f'rh.pial'

    Cortribbon_node.inputs.lh_ribbon = subject_mri_dir / f'lh.ribbon.mgz'
    Cortribbon_node.inputs.rh_ribbon = subject_mri_dir / f'rh.ribbon.mgz'
    Cortribbon_node.inputs.ribbon = subject_mri_dir / 'ribbon.mgz'
    Cortribbon_node.base_dir = workflow_cached_dir
    Cortribbon_node.source = Source()

    return Cortribbon_node


def create_Parcstats_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    threads = 8

    Parcstats_node = Node(Parcstats(), name=f'{subject_id}_Parcstats_node')
    Parcstats_node.inputs.subjects_dir = subjects_dir
    Parcstats_node.inputs.subject_id = subject_id
    Parcstats_node.inputs.threads = threads

    Parcstats_node.inputs.lh_aparc_annot = subject_label_dir / f'lh.aparc.annot'
    Parcstats_node.inputs.rh_aparc_annot = subject_label_dir / f'rh.aparc.annot'
    Parcstats_node.inputs.wm_file = subject_mri_dir / 'wm.mgz'
    Parcstats_node.inputs.ribbon_file = subject_mri_dir / 'ribbon.mgz'
    Parcstats_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Parcstats_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    Parcstats_node.inputs.lh_pial = subject_surf_dir / f'lh.pial'
    Parcstats_node.inputs.rh_pial = subject_surf_dir / f'rh.pial'
    Parcstats_node.inputs.lh_thickness = subject_surf_dir / f'lh.thickness'
    Parcstats_node.inputs.rh_thickness = subject_surf_dir / f'rh.thickness'
    Parcstats_node.base_dir = workflow_cached_dir
    Parcstats_node.source = Source()

    return Parcstats_node


def create_Aseg7_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    threads = 8

    Aseg7_node = Node(Aseg7(), name=f'{subject_id}_Aseg7_node')
    Aseg7_node.inputs.subjects_dir = subjects_dir
    Aseg7_node.inputs.subject_id = subject_id
    Aseg7_node.inputs.threads = threads
    Aseg7_node.inputs.aseg_presurf_hypos = subject_mri_dir / 'aseg.presurf.hypos.mgz'
    Aseg7_node.inputs.aseg_file = subject_mri_dir / 'aseg.mgz'
    Aseg7_node.inputs.lh_cortex_label = subject_label_dir / 'lh.cortex.label'
    Aseg7_node.inputs.lh_white = subject_surf_dir / 'lh.white'
    Aseg7_node.inputs.lh_pial = subject_surf_dir / 'lh.pial'
    Aseg7_node.inputs.lh_aparc_annot = subject_label_dir / 'lh.aparc.annot'
    Aseg7_node.inputs.rh_cortex_label = subject_label_dir / 'rh.cortex.label'
    Aseg7_node.inputs.rh_white = subject_surf_dir / 'rh.white'
    Aseg7_node.inputs.rh_pial = subject_surf_dir / 'rh.pial'
    Aseg7_node.inputs.rh_aparc_annot = subject_label_dir / 'rh.aparc.annot'
    Aseg7_node.inputs.aparc_aseg = subject_mri_dir / 'aparc+aseg.mgz'
    Aseg7_node.base_dir = workflow_cached_dir
    Aseg7_node.source = Source()

    return Aseg7_node


def create_N4BiasCorrect_node(subject_id: str):
    fastsurfer_home = Path(os.environ['FASTSURFER_HOME'])
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable
    sub_mri_dir = subjects_dir / subject_id / "mri"
    correct_py = fastsurfer_home / "recon_surf" / "N4_bias_correct.py"

    orig_file = sub_mri_dir / "orig.mgz"
    mask_file = sub_mri_dir / "mask.mgz"

    N4_bias_correct_node = Node(N4BiasCorrect(), name=f'{subject_id}_N4_bias_correct_node')
    N4_bias_correct_node.inputs.subject_id = subject_id
    N4_bias_correct_node.inputs.subjects_dir = subjects_dir
    N4_bias_correct_node.inputs.python_interpret = python_interpret
    N4_bias_correct_node.inputs.correct_py = correct_py
    N4_bias_correct_node.inputs.orig_file = orig_file
    N4_bias_correct_node.inputs.mask_file = mask_file
    N4_bias_correct_node.inputs.threads = 8

    N4_bias_correct_node.base_dir = workflow_cached_dir
    N4_bias_correct_node.source = Source()

    return N4_bias_correct_node


def create_TalairachAndNu_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    sub_mri_dir = subjects_dir / subject_id / "mri"
    orig_nu_file = sub_mri_dir / "orig_nu.mgz"
    orig_file = sub_mri_dir / "orig.mgz"
    freesurfer_home = Path(os.environ['FREESURFER_HOME'])
    mni305 = freesurfer_home / "average" / "mni305.cor.mgz"

    talairach_and_nu_node = Node(TalairachAndNu(), name=f'{subject_id}_talairach_and_nu_node')
    talairach_and_nu_node.inputs.subjects_dir = subjects_dir
    talairach_and_nu_node.inputs.subject_id = subject_id
    talairach_and_nu_node.inputs.threads = 8
    talairach_and_nu_node.inputs.mni305 = mni305
    talairach_and_nu_node.inputs.orig_nu_file = orig_nu_file
    talairach_and_nu_node.inputs.orig_file = orig_file

    talairach_and_nu_node.base_dir = workflow_cached_dir
    talairach_and_nu_node.source = Source()

    return talairach_and_nu_node


def create_Brainmask_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    brainmask_node = Node(Brainmask(), name=f'{subject_id}_brainmask_node')
    brainmask_node.inputs.subjects_dir = subjects_dir
    brainmask_node.inputs.subject_id = subject_id
    brainmask_node.inputs.need_t1 = True
    brainmask_node.inputs.nu_file = subjects_dir / subject_id / 'mri' / 'nu.mgz'
    brainmask_node.inputs.mask_file = subjects_dir / subject_id / 'mri' / 'mask.mgz'

    brainmask_node.base_dir = workflow_cached_dir
    brainmask_node.source = Source()

    return brainmask_node


def create_UpdateAseg_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    paint_cc_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / 'paint_cc_into_pred.py'
    updateaseg_node = Node(UpdateAseg(), name=f'{subject_id}_updateaseg_node')
    updateaseg_node.inputs.subjects_dir = subjects_dir
    updateaseg_node.inputs.subject_id = subject_id
    updateaseg_node.inputs.paint_cc_file = paint_cc_file
    updateaseg_node.inputs.python_interpret = python_interpret
    updateaseg_node.inputs.seg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'
    updateaseg_node.inputs.aseg_noCCseg_file = subject_mri_dir / 'aseg.auto_noCCseg.mgz'

    updateaseg_node.base_dir = workflow_cached_dir
    updateaseg_node.source = Source()

    return updateaseg_node


def create_Filled_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    filled_node = Node(Filled(), name=f'{subject_id}_filled_node')
    filled_node.inputs.subjects_dir = subjects_dir
    filled_node.inputs.subject_id = subject_id
    filled_node.inputs.threads = 8
    filled_node.inputs.aseg_auto_file = subjects_dir / subject_id / 'mri/aseg.auto.mgz'
    filled_node.inputs.norm_file = subjects_dir / subject_id / 'mri/norm.mgz'
    filled_node.inputs.brainmask_file = subjects_dir / subject_id / 'mri/brainmask.mgz'
    filled_node.inputs.talairach_lta = subjects_dir / subject_id / 'mri/transforms/talairach.lta'

    filled_node.base_dir = workflow_cached_dir
    filled_node.source = Source()

    return filled_node


def create_FastCSR_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    fastcsr_home = Path(os.environ['FASTCSR_HOME'])
    fastcsr_py = fastcsr_home / 'pipeline.py'  # inference script

    fastcsr_node = Node(FastCSR(), name=f'{subject_id}_fastcsr_node')
    fastcsr_node.inputs.python_interpret = python_interpret
    fastcsr_node.inputs.fastcsr_py = fastcsr_py
    fastcsr_node.inputs.subjects_dir = subjects_dir
    fastcsr_node.inputs.subject_id = subject_id
    fastcsr_node.inputs.orig_file = Path(subjects_dir) / subject_id / 'mri/orig.mgz'
    fastcsr_node.inputs.filled_file = Path(subjects_dir) / subject_id / 'mri/filled.mgz'
    fastcsr_node.inputs.aseg_presurf_file = Path(subjects_dir) / subject_id / 'mri/aseg.presurf.mgz'
    fastcsr_node.inputs.brainmask_file = Path(subjects_dir) / subject_id / 'mri/brainmask.mgz'
    fastcsr_node.inputs.wm_file = Path(subjects_dir) / subject_id / 'mri/wm.mgz'
    fastcsr_node.inputs.brain_finalsurfs_file = Path(subjects_dir) / subject_id / 'mri/brain.finalsurfs.mgz'

    fastcsr_node.base_dir = workflow_cached_dir
    fastcsr_node.source = Source()

    return fastcsr_node


def create_WhitePreaparc1_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    white_preaparc1 = Node(WhitePreaparc1(), name=f'{subject_id}_white_preaparc1_node')
    white_preaparc1.inputs.subjects_dir = subjects_dir
    white_preaparc1.inputs.subject_id = subject_id
    white_preaparc1.inputs.threads = 8

    white_preaparc1.base_dir = workflow_cached_dir
    white_preaparc1.source = Source()

    return white_preaparc1


def create_SampleSegmentationToSurfave_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable
    freesurfer_home = Path(os.environ['FREESURFER_HOME'])
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    smooth_aparc_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / 'smooth_aparc.py'
    lh_DKTatlaslookup_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / f'lh.DKTatlaslookup.txt'
    rh_DKTatlaslookup_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / f'rh.DKTatlaslookup.txt'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    SampleSegmentationToSurfave_node = Node(SampleSegmentationToSurfave(), name=f'{subject_id}_SampleSegmentationToSurfave_node')
    SampleSegmentationToSurfave_node.inputs.subjects_dir = subjects_dir
    SampleSegmentationToSurfave_node.inputs.subject_id = subject_id
    SampleSegmentationToSurfave_node.inputs.python_interpret = python_interpret
    SampleSegmentationToSurfave_node.inputs.freesurfer_home = freesurfer_home
    SampleSegmentationToSurfave_node.inputs.lh_DKTatlaslookup_file = lh_DKTatlaslookup_file
    SampleSegmentationToSurfave_node.inputs.rh_DKTatlaslookup_file = rh_DKTatlaslookup_file
    SampleSegmentationToSurfave_node.inputs.aparc_aseg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.withCC.mgz'
    SampleSegmentationToSurfave_node.inputs.smooth_aparc_file = smooth_aparc_file
    SampleSegmentationToSurfave_node.inputs.lh_white_preaparc_file = subject_surf_dir / f'lh.white.preaparc'
    SampleSegmentationToSurfave_node.inputs.rh_white_preaparc_file = subject_surf_dir / f'rh.white.preaparc'
    SampleSegmentationToSurfave_node.inputs.lh_cortex_label_file = subject_label_dir / f'lh.cortex.label'
    SampleSegmentationToSurfave_node.inputs.rh_cortex_label_file = subject_label_dir / f'rh.cortex.label'

    SampleSegmentationToSurfave_node.base_dir = workflow_cached_dir
    SampleSegmentationToSurfave_node.source = Source()

    return SampleSegmentationToSurfave_node


def creat_VxmRegistraion_node(subject_id: str, vxm_model_path: Path, atlas_type: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])

    VxmRegistraion_node = Node(VxmRegistraion(), name=f'{subject_id}_VxmRegistraion_node')
    VxmRegistraion_node.inputs.subject_id = subject_id
    VxmRegistraion_node.inputs.data_path = data_path
    VxmRegistraion_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    VxmRegistraion_node.inputs.subjects_dir = subjects_dir
    VxmRegistraion_node.inputs.model_file = vxm_model_path / atlas_type / 'model.h5'
    VxmRegistraion_node.inputs.vxm_model_path = vxm_model_path
    VxmRegistraion_node.inputs.atlas_type = atlas_type

    VxmRegistraion_node.base_dir = workflow_cached_dir
    VxmRegistraion_node.source = Source()

    return VxmRegistraion_node


def creat_BoldSkipReorient_node(subject_id: str, task: str):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])

    BoldSkipReorient_node = Node(BoldSkipReorient(), name=f'{subject_id}_BoldSkipReorient_node')
    BoldSkipReorient_node.inputs.subject_ids = subject_id
    BoldSkipReorient_node.inputs.data_path = data_path
    BoldSkipReorient_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    BoldSkipReorient_node.inputs.task = task

    BoldSkipReorient_node.base_dir = workflow_cached_dir
    BoldSkipReorient_node.source = Source()

    return BoldSkipReorient_node


def creat_Stc_node(subject_id: str, task: str):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])

    Stc_node = Node(Stc(), name=f'{subject_id}_stc_node')
    Stc_node.inputs.subject_id = subject_id
    Stc_node.inputs.task = task
    Stc_node.inputs.data_path = data_path
    Stc_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    Stc_node.inputs.base_dir = workflow_cached_dir
    Stc_node.source = Source()

    return Stc_node


def creat_MkTemplate_node(subject_id: str, task: str):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])

    MkTemplate_node = Node(MkTemplate(), name=f'{subject_id}_MkTemplate_node')
    MkTemplate_node.inputs.subject_id = subject_id
    MkTemplate_node.inputs.task = task
    MkTemplate_node.inputs.data_path = data_path
    MkTemplate_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    MkTemplate_node.base_dir = workflow_cached_dir
    MkTemplate_node.source = Source()

    return MkTemplate_node


def creat_MotionCorrection_node(subject_id: str, task: str):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])

    MotionCorrection_node = Node(MotionCorrection(), name=f'{subject_id}_MotionCorrection_node')
    MotionCorrection_node.inputs.subject_id = subject_id
    MotionCorrection_node.inputs.task = task
    MotionCorrection_node.inputs.data_path = data_path
    MotionCorrection_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    MotionCorrection_node.base_dir = workflow_cached_dir
    MotionCorrection_node.source = Source()

    return MotionCorrection_node


def creat_Register_node(subject_id: str, task: str):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])

    Register_node = Node(Register(), name=f'{subject_id}_register_node')
    Register_node.inputs.subject_id = subject_id
    Register_node.inputs.task = task
    Register_node.inputs.data_path = data_path
    Register_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    Register_node.base_dir = workflow_cached_dir
    Register_node.source = Source()

    return Register_node


def creat_Mkbrainmask_node(subject_id: str, task: str):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])

    Mkbrainmask_node = Node(MkBrainmask(), name=f'{subject_id}_mkbrainmask_node')
    Mkbrainmask_node.inputs.subject_id = subject_id
    Mkbrainmask_node.inputs.subjects_dir = subjects_dir
    Mkbrainmask_node.inputs.task = task
    Mkbrainmask_node.inputs.data_path = data_path
    Mkbrainmask_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    Mkbrainmask_node.base_dir = workflow_cached_dir
    Mkbrainmask_node.source = Source()

    return Mkbrainmask_node


def creat_RestGauss_node(subject_id: str, task: str):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])

    RestGauss_node = Node(RestGauss(), name=f'{subject_id}_RestGauss_node')
    RestGauss_node.inputs.subject_id = subject_id
    RestGauss_node.inputs.subjects_dir = subjects_dir
    RestGauss_node.inputs.data_path = data_path
    RestGauss_node.inputs.task = task
    RestGauss_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    RestGauss_node.base_dir = workflow_cached_dir
    RestGauss_node.source = Source()

    return RestGauss_node


def creat_RestBandpass_node(subject_id: str, task: str):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])

    RestBandpass_node = Node(RestBandpass(), name=f'{subject_id}_RestBandpass_node')
    RestBandpass_node.inputs.subject_id = subject_id
    RestBandpass_node.inputs.data_path = data_path
    RestBandpass_node.inputs.task = task
    RestBandpass_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    RestBandpass_node.base_dir = workflow_cached_dir
    RestBandpass_node.source = Source()

    return RestBandpass_node


def creat_RestRegression_node(subject_id: str, task: str):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])

    RestRegression_node = Node(RestRegression(), name=f'{subject_id}_RestRegression_node')
    RestRegression_node.inputs.subject_id = subject_id
    RestRegression_node.inputs.subjects_dir = subjects_dir
    RestRegression_node.inputs.data_path = data_path
    RestRegression_node.inputs.task = task
    RestRegression_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    RestRegression_node.base_dir = workflow_cached_dir
    RestRegression_node.source = Source()

    return RestRegression_node


def creat_Smooth_node(subject_id: str, task: str, preprocess_method: str, mni152_brain_mask: Path):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])

    Smooth_node = Node(Smooth(), name=f'{subject_id}_Smooth_node')
    Smooth_node.inputs.subject_id = subject_id
    Smooth_node.inputs.task = task
    Smooth_node.inputs.data_path = data_path
    Smooth_node.inputs.preprocess_method = preprocess_method
    Smooth_node.inputs.MNI152_T1_2mm_brain_mask = mni152_brain_mask
    Smooth_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    Smooth_node.inputs.base_dir = workflow_cached_dir
    Smooth_node.source = Source()

    return Smooth_node


def creat_VxmRegNormMNI152_node(subject_id: str, task: str, preprocess_method: str, atlas_type: str,
                                vxm_model_path: Path, resource_dir: Path):
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    derivative_deepprep_path = Path(os.environ['BOLD_PROPROCESS_DIR'])
    data_path = Path(os.environ['BIDS_DIR'])
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])

    VxmRegNormMNI152_node = Node(VxmRegNormMNI152(), name=f'{subject_id}_VxmRegNormMNI152_node')
    VxmRegNormMNI152_node.inputs.subjects_dir = subjects_dir
    VxmRegNormMNI152_node.inputs.subject_id = subject_id
    VxmRegNormMNI152_node.inputs.atlas_type = atlas_type
    VxmRegNormMNI152_node.inputs.task = task
    VxmRegNormMNI152_node.inputs.data_path = data_path
    VxmRegNormMNI152_node.inputs.preprocess_method = preprocess_method
    VxmRegNormMNI152_node.inputs.vxm_model_path = vxm_model_path
    VxmRegNormMNI152_node.inputs.resource_dir = resource_dir
    VxmRegNormMNI152_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    VxmRegNormMNI152_node.base_dir = workflow_cached_dir
    VxmRegNormMNI152_node.source = Source()

    return VxmRegNormMNI152_node


def create_node_t():
    from interface.run import set_envrion
    set_envrion()

    pwd = Path.cwd()
    pwd = pwd.parent
    fastsurfer_home = pwd / "FastSurfer"
    freesurfer_home = Path('/usr/local/freesurfer720')
    fastcsr_home = pwd / "FastCSR"
    featreg_home = pwd / "FeatReg"

    bids_data_dir_test = '/mnt/ngshare/DeepPrep_workflow_test/UKB_BIDS'
    subjects_dir_test = '/mnt/ngshare/DeepPrep_workflow_test/UKB_Recon'
    bold_preprocess_dir_test = '/mnt/ngshare/DeepPrep_workflow_test/UKB_BoldPreprocess'
    workflow_cached_dir_test = '/mnt/ngshare/DeepPrep_workflow_test/UKB_Workflow'

    subject_id_test = 'sub-1000037-ses-02'
    t1w_files = ['/mnt/ngshare/DeepPrep_workflow_test/UKB_BIDS/sub-1000037/ses-02/anat/sub-1000037_ses-02_T1w.nii.gz']

    os.environ['SUBJECTS_DIR'] = str(subjects_dir_test)
    os.environ['BOLD_PREPROCESS_DIR'] = str(bold_preprocess_dir_test)
    os.environ['WORKFLOW_CACHED_DIR'] = str(workflow_cached_dir_test)
    os.environ['FASTSURFER_HOME'] = str(fastsurfer_home)
    os.environ['FREESURFER_HOME'] = str(freesurfer_home)
    os.environ['FASTCSR_HOME'] = str(fastcsr_home)
    os.environ['FEATREG_HOME'] = str(featreg_home)

    # 测试
    node = create_origandrawavg_node(subject_id=subject_id_test, t1w_files=t1w_files)
    node.run()
    sub_node = node.interface.create_sub_node()
    sub_node.run()
    sub_node.interface.create_sub_node()

    node = create_Segment_node(subject_id=subject_id_test)
    node.run()
    sub_node = node.interface.create_sub_node()
    sub_node.run()
    sub_node.interface.create_sub_node()


if __name__ == '__main__':
    create_node_t()  # 测试
