from interface.freesurfer_node_hard import *
from interface.fastcsr_node import *
from interface.fastsurfer_node_hard import *
from interface.featreg_node import *
from interface.node_source import Source
import sys
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

THREAD = 1


def create_OrigAndRawavg_node(subject_id: str, t1w_files: list):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = os.environ['WORKFLOW_CACHED_DIR']

    origandrawavg_node = Node(OrigAndRawavg(), f'{subject_id}_recon_OrigAndRawavg_node')
    origandrawavg_node.inputs.t1w_files = t1w_files
    origandrawavg_node.inputs.subjects_dir = subjects_dir
    origandrawavg_node.inputs.subject_id = subject_id
    origandrawavg_node.inputs.threads = THREAD

    origandrawavg_node.base_dir = workflow_cached_dir
    origandrawavg_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=500)

    origandrawavg_node.interface.recon_only = os.environ['RECON_ONLY']

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

    segment_node = Node(Segment(), f'{subject_id}_recon_Segment_node')
    segment_node.inputs.subjects_dir = subjects_dir
    segment_node.inputs.subject_id = subject_id
    segment_node.inputs.python_interpret = python_interpret
    segment_node.inputs.eval_py = fastsurfer_eval
    segment_node.inputs.network_sagittal_path = network_sagittal_path
    segment_node.inputs.network_coronal_path = network_coronal_path
    segment_node.inputs.network_axial_path = network_axial_path

    segment_node.base_dir = workflow_cached_dir
    segment_node.source = Source(CPU_n=0, GPU_MB=8500, RAM_MB=7500)

    segment_node.interface.recon_only = os.environ['RECON_ONLY']

    return segment_node


def create_Noccseg_node(subject_id: str):
    fastsurfer_home = Path(os.environ['FASTSURFER_HOME'])
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable

    reduce_to_aseg_py = fastsurfer_home / 'recon_surf' / 'reduce_to_aseg.py'

    noccseg_node = Node(Noccseg(), f'{subject_id}_recon_Noccseg_node')
    noccseg_node.inputs.python_interpret = python_interpret
    noccseg_node.inputs.reduce_to_aseg_py = reduce_to_aseg_py
    noccseg_node.inputs.subject_id = subject_id
    noccseg_node.inputs.subjects_dir = subjects_dir

    noccseg_node.base_dir = workflow_cached_dir
    noccseg_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=500)

    noccseg_node.interface.recon_only = os.environ['RECON_ONLY']

    return noccseg_node


def create_N4BiasCorrect_node(subject_id: str):
    fastsurfer_home = Path(os.environ['FASTSURFER_HOME'])
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable
    sub_mri_dir = subjects_dir / subject_id / "mri"
    correct_py = fastsurfer_home / "recon_surf" / "N4_bias_correct.py"

    orig_file = sub_mri_dir / "orig.mgz"
    mask_file = sub_mri_dir / "mask.mgz"

    N4_bias_correct_node = Node(N4BiasCorrect(), name=f'{subject_id}_recon_N4BiasCorrect_node')
    N4_bias_correct_node.inputs.subject_id = subject_id
    N4_bias_correct_node.inputs.subjects_dir = subjects_dir
    N4_bias_correct_node.inputs.python_interpret = python_interpret
    N4_bias_correct_node.inputs.correct_py = correct_py
    N4_bias_correct_node.inputs.orig_file = orig_file
    N4_bias_correct_node.inputs.threads = THREAD

    N4_bias_correct_node.base_dir = workflow_cached_dir
    N4_bias_correct_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=500)

    N4_bias_correct_node.interface.recon_only = os.environ['RECON_ONLY']

    return N4_bias_correct_node


def create_TalairachAndNu_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    sub_mri_dir = subjects_dir / subject_id / "mri"
    orig_nu_file = sub_mri_dir / "orig_nu.mgz"
    orig_file = sub_mri_dir / "orig.mgz"
    freesurfer_home = Path(os.environ['FREESURFER_HOME'])
    mni305 = freesurfer_home / "average" / "mni305.cor.mgz"

    talairach_and_nu_node = Node(TalairachAndNu(), name=f'{subject_id}_recon_TalairachAndNu_node')
    talairach_and_nu_node.inputs.subjects_dir = subjects_dir
    talairach_and_nu_node.inputs.subject_id = subject_id
    talairach_and_nu_node.inputs.threads = THREAD
    talairach_and_nu_node.inputs.mni305 = mni305
    talairach_and_nu_node.inputs.orig_nu_file = orig_nu_file
    talairach_and_nu_node.inputs.orig_file = orig_file

    talairach_and_nu_node.base_dir = workflow_cached_dir
    talairach_and_nu_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=500)

    talairach_and_nu_node.interface.recon_only = os.environ['RECON_ONLY']

    return talairach_and_nu_node


def create_Brainmask_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    atlas_type = os.environ['DEEPPREP_ATLAS_TYPE']
    task = os.environ['DEEPPREP_TASK']
    preprocess_method = os.environ['DEEPPREP_PREPROCESS_METHOD']

    brainmask_node = Node(Brainmask(), name=f'{subject_id}_recon_Brainmask_node')
    brainmask_node.inputs.subjects_dir = subjects_dir
    brainmask_node.inputs.subject_id = subject_id
    brainmask_node.inputs.need_t1 = True
    brainmask_node.inputs.nu_file = subjects_dir / subject_id / 'mri' / 'nu.mgz'
    brainmask_node.inputs.mask_file = subjects_dir / subject_id / 'mri' / 'mask.mgz'

    brainmask_node.base_dir = workflow_cached_dir
    brainmask_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=1000)

    brainmask_node.interface.atlas_type = atlas_type
    brainmask_node.interface.task = task
    brainmask_node.interface.preprocess_method = preprocess_method
    brainmask_node.interface.recon_only = os.environ['RECON_ONLY']

    return brainmask_node


def create_UpdateAseg_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    fastsurfer_home = Path(os.environ['FASTSURFER_HOME'])
    python_interpret = sys.executable
    subject_mri_dir = subjects_dir / subject_id / 'mri'

    paint_cc_file = fastsurfer_home / 'recon_surf' / 'paint_cc_into_pred.py'
    updateaseg_node = Node(UpdateAseg(), name=f'{subject_id}_recon_UpdateAseg_node')
    updateaseg_node.inputs.subjects_dir = subjects_dir
    updateaseg_node.inputs.subject_id = subject_id
    updateaseg_node.inputs.paint_cc_file = paint_cc_file
    updateaseg_node.inputs.python_interpret = python_interpret
    updateaseg_node.inputs.seg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'
    updateaseg_node.inputs.aseg_noCCseg_file = subject_mri_dir / 'aseg.auto_noCCseg.mgz'

    updateaseg_node.base_dir = workflow_cached_dir
    updateaseg_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=500)

    updateaseg_node.interface.recon_only = os.environ['RECON_ONLY']

    return updateaseg_node


def create_Filled_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    filled_node = Node(Filled(), name=f'{subject_id}_recon_Filled_node')
    filled_node.inputs.subjects_dir = subjects_dir
    filled_node.inputs.subject_id = subject_id
    filled_node.inputs.threads = THREAD
    filled_node.inputs.aseg_auto_file = subjects_dir / subject_id / 'mri/aseg.auto.mgz'
    filled_node.inputs.norm_file = subjects_dir / subject_id / 'mri/norm.mgz'
    filled_node.inputs.brainmask_file = subjects_dir / subject_id / 'mri/brainmask.mgz'
    filled_node.inputs.talairach_lta = subjects_dir / subject_id / 'mri/transforms/talairach.lta'

    filled_node.base_dir = workflow_cached_dir
    filled_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=500)

    filled_node.interface.recon_only = os.environ['RECON_ONLY']

    return filled_node


def create_FastCSR_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    fastcsr_home = Path(os.environ['FASTCSR_HOME'])
    fastcsr_py = fastcsr_home / 'pipeline.py'  # inference script

    fastcsr_node = Node(FastCSR(), name=f'{subject_id}_recon_FastCSR_node')
    fastcsr_node.inputs.python_interpret = python_interpret
    fastcsr_node.inputs.fastcsr_py = fastcsr_py
    fastcsr_node.inputs.parallel_scheduling = 'off'
    fastcsr_node.inputs.subjects_dir = subjects_dir
    fastcsr_node.inputs.subject_id = subject_id
    fastcsr_node.inputs.orig_file = Path(subjects_dir) / subject_id / 'mri/orig.mgz'
    fastcsr_node.inputs.filled_file = Path(subjects_dir) / subject_id / 'mri/filled.mgz'
    fastcsr_node.inputs.aseg_presurf_file = Path(subjects_dir) / subject_id / 'mri/aseg.presurf.mgz'
    fastcsr_node.inputs.brainmask_file = Path(subjects_dir) / subject_id / 'mri/brainmask.mgz'
    fastcsr_node.inputs.wm_file = Path(subjects_dir) / subject_id / 'mri/wm.mgz'
    fastcsr_node.inputs.brain_finalsurfs_file = Path(subjects_dir) / subject_id / 'mri/brain.finalsurfs.mgz'

    fastcsr_node.base_dir = workflow_cached_dir
    fastcsr_node.source = Source(CPU_n=0, GPU_MB=7000, RAM_MB=6500)

    fastcsr_node.interface.recon_only = os.environ['RECON_ONLY']

    return fastcsr_node


def create_WhitePreaparc1_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    atlas_type = os.environ['DEEPPREP_ATLAS_TYPE']
    task = os.environ['DEEPPREP_TASK']
    preprocess_method = os.environ['DEEPPREP_PREPROCESS_METHOD']

    white_preaparc1 = Node(WhitePreaparc1(), name=f'{subject_id}_recon_WhitePreaparc1_node')
    white_preaparc1.inputs.subjects_dir = subjects_dir
    white_preaparc1.inputs.subject_id = subject_id
    white_preaparc1.inputs.threads = THREAD

    white_preaparc1.base_dir = workflow_cached_dir
    white_preaparc1.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=1500)

    white_preaparc1.interface.atlas_type = atlas_type
    white_preaparc1.interface.task = task
    white_preaparc1.interface.preprocess_method = preprocess_method
    white_preaparc1.interface.recon_only = os.environ['RECON_ONLY']

    return white_preaparc1


def create_SampleSegmentationToSurface_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable
    freesurfer_home = Path(os.environ['FREESURFER_HOME'])
    fastsurfer_home = Path(os.environ['FASTSURFER_HOME'])

    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    smooth_aparc_file = fastsurfer_home / 'recon_surf' / 'smooth_aparc.py'
    lh_DKTatlaslookup_file = fastsurfer_home / 'recon_surf' / f'lh.DKTatlaslookup.txt'
    rh_DKTatlaslookup_file = fastsurfer_home / 'recon_surf' / f'rh.DKTatlaslookup.txt'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    SampleSegmentationToSurfave_node = Node(SampleSegmentationToSurface(),
                                            name=f'{subject_id}_recon_SampleSegmentationToSurface_node')
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
    SampleSegmentationToSurfave_node.source = Source(CPU_n=2, GPU_MB=0, RAM_MB=4000)

    SampleSegmentationToSurfave_node.interface.recon_only = os.environ['RECON_ONLY']

    return SampleSegmentationToSurfave_node


def create_InflatedSphere_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    lh_white_preaparc_file = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
    rh_white_preaparc_file = subjects_dir / subject_id / "surf" / "rh.white.preaparc"

    Inflated_Sphere_node = Node(InflatedSphere(), f'{subject_id}_recon_InflatedSphere_node')
    Inflated_Sphere_node.inputs.threads = THREAD
    Inflated_Sphere_node.inputs.subjects_dir = subjects_dir
    Inflated_Sphere_node.inputs.subject_id = subject_id
    Inflated_Sphere_node.inputs.lh_white_preaparc_file = lh_white_preaparc_file
    Inflated_Sphere_node.inputs.rh_white_preaparc_file = rh_white_preaparc_file

    Inflated_Sphere_node.base_dir = workflow_cached_dir
    Inflated_Sphere_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=500)

    Inflated_Sphere_node.interface.recon_only = os.environ['RECON_ONLY']

    return Inflated_Sphere_node


def create_FeatReg_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    featreg_home = Path(os.environ["FEATREG_HOME"])
    freesurfer_home = Path(os.environ['FREESURFER_HOME'])

    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    device = os.environ['DEEPPREP_DEVICES']

    python_interpret = sys.executable
    featreg_py = featreg_home / "featreg" / 'predict.py'  # inference script

    featreg_node = Node(FeatReg(), f'{subject_id}_recon_FeatReg_node')
    featreg_node.inputs.featreg_py = featreg_py
    featreg_node.inputs.python_interpret = python_interpret
    featreg_node.inputs.device = device

    featreg_node.inputs.subjects_dir = subjects_dir
    featreg_node.inputs.subject_id = subject_id
    featreg_node.inputs.freesurfer_home = freesurfer_home
    featreg_node.inputs.lh_sulc = Path(subjects_dir) / subject_id / f'surf/lh.sulc'
    featreg_node.inputs.rh_sulc = Path(subjects_dir) / subject_id / f'surf/rh.sulc'
    featreg_node.inputs.lh_curv = Path(subjects_dir) / subject_id / f'surf/lh.curv'
    featreg_node.inputs.rh_curv = Path(subjects_dir) / subject_id / f'surf/rh.curv'
    featreg_node.inputs.lh_sphere = Path(subjects_dir) / subject_id / f'surf/lh.sphere'
    featreg_node.inputs.rh_sphere = Path(subjects_dir) / subject_id / f'surf/rh.sphere'

    featreg_node.base_dir = workflow_cached_dir
    featreg_node.source = Source(CPU_n=0, GPU_MB=7000, RAM_MB=10000)

    featreg_node.interface.recon_only = os.environ['RECON_ONLY']

    return featreg_node


def create_JacobianAvgcurvCortparc_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    JacobianAvgcurvCortparc_node = Node(JacobianAvgcurvCortparc(), f'{subject_id}_JacobianAvgcurvCortparc_node')
    JacobianAvgcurvCortparc_node.inputs.subjects_dir = subjects_dir
    JacobianAvgcurvCortparc_node.inputs.subject_id = subject_id
    JacobianAvgcurvCortparc_node.inputs.threads = THREAD

    JacobianAvgcurvCortparc_node.base_dir = workflow_cached_dir
    JacobianAvgcurvCortparc_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=500)

    JacobianAvgcurvCortparc_node.interface.recon_only = os.environ['RECON_ONLY']

    return JacobianAvgcurvCortparc_node


def create_WhitePialThickness1_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    threads = 1

    white_pial_thickness1 = Node(WhitePialThickness1(), name=f'{subject_id}_recon_WhitePialThickness1_node')
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
    white_pial_thickness1.inputs.lh_cortex_label = subjects_dir / subject_id / "label" / "lh.cortex.label"
    white_pial_thickness1.inputs.rh_cortex_label = subjects_dir / subject_id / "label" / "rh.cortex.label"

    white_pial_thickness1.base_dir = workflow_cached_dir
    white_pial_thickness1.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=1500)

    white_pial_thickness1.interface.recon_only = os.environ['RECON_ONLY']

    return white_pial_thickness1


def create_Curvstats_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    threads = 8

    Curvstats_node = Node(Curvstats(), name=f'{subject_id}_recon_Curvstats_node')
    Curvstats_node.inputs.subjects_dir = subjects_dir
    Curvstats_node.inputs.subject_id = subject_id
    subject_surf_dir = subjects_dir / subject_id / "surf"

    Curvstats_node.inputs.lh_smoothwm = subject_surf_dir / f'lh.smoothwm'
    Curvstats_node.inputs.rh_smoothwm = subject_surf_dir / f'rh.smoothwm'
    Curvstats_node.inputs.lh_curv = subject_surf_dir / f'lh.curv'
    Curvstats_node.inputs.rh_curv = subject_surf_dir / f'rh.curv'
    Curvstats_node.inputs.lh_sulc = subject_surf_dir / f'lh.sulc'
    Curvstats_node.inputs.rh_sulc = subject_surf_dir / f'rh.sulc'
    Curvstats_node.inputs.threads = threads

    Curvstats_node.base_dir = workflow_cached_dir
    Curvstats_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=250)

    Curvstats_node.interface.recon_only = os.environ['RECON_ONLY']

    return Curvstats_node


def create_BalabelsMult_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    threads = 1

    BalabelsMult_node = Node(BalabelsMult(), name=f'{subject_id}_recon_BalabelsMult_node')
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
    BalabelsMult_node.source = Source(CPU_n=2, GPU_MB=0, RAM_MB=1500)

    BalabelsMult_node.interface.recon_only = os.environ['RECON_ONLY']

    return BalabelsMult_node


def create_Cortribbon_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    threads = 1

    Cortribbon_node = Node(Cortribbon(), name=f'{subject_id}_recon_Cortribbon_node')
    Cortribbon_node.inputs.subjects_dir = subjects_dir
    Cortribbon_node.inputs.subject_id = subject_id
    Cortribbon_node.inputs.threads = threads

    Cortribbon_node.inputs.aseg_presurf_file = subject_mri_dir / 'aseg.presurf.mgz'
    Cortribbon_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Cortribbon_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    Cortribbon_node.inputs.lh_pial = subject_surf_dir / f'lh.pial'
    Cortribbon_node.inputs.rh_pial = subject_surf_dir / f'rh.pial'

    Cortribbon_node.base_dir = workflow_cached_dir
    Cortribbon_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=1000)

    Cortribbon_node.interface.recon_only = os.environ['RECON_ONLY']

    return Cortribbon_node


def create_Parcstats_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    threads = 1

    Parcstats_node = Node(Parcstats(), name=f'{subject_id}_recon_Parcstats_node')
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
    Parcstats_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=500)

    Parcstats_node.interface.recon_only = os.environ['RECON_ONLY']

    return Parcstats_node


def create_Aseg7_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    atlas_type = os.environ['DEEPPREP_ATLAS_TYPE']
    task = os.environ['DEEPPREP_TASK']
    preprocess_method = os.environ['DEEPPREP_PREPROCESS_METHOD']

    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    threads = 1

    Aseg7_node = Node(Aseg7(), name=f'{subject_id}_recon_Aseg7_node')
    Aseg7_node.inputs.subjects_dir = subjects_dir
    Aseg7_node.inputs.subject_id = subject_id
    Aseg7_node.inputs.threads = threads
    Aseg7_node.inputs.aseg_file = subject_mri_dir / 'aseg.mgz'
    Aseg7_node.inputs.lh_cortex_label = subject_label_dir / 'lh.cortex.label'
    Aseg7_node.inputs.lh_white = subject_surf_dir / 'lh.white'
    Aseg7_node.inputs.lh_pial = subject_surf_dir / 'lh.pial'
    Aseg7_node.inputs.lh_aparc_annot = subject_label_dir / 'lh.aparc.annot'
    Aseg7_node.inputs.rh_cortex_label = subject_label_dir / 'rh.cortex.label'
    Aseg7_node.inputs.rh_white = subject_surf_dir / 'rh.white'
    Aseg7_node.inputs.rh_pial = subject_surf_dir / 'rh.pial'
    Aseg7_node.inputs.rh_aparc_annot = subject_label_dir / 'rh.aparc.annot'
    Aseg7_node.base_dir = workflow_cached_dir
    Aseg7_node.source = Source(CPU_n=1, GPU_MB=0, RAM_MB=800)

    Aseg7_node.interface.atlas_type = atlas_type
    Aseg7_node.interface.task = task
    Aseg7_node.interface.preprocess_method = preprocess_method

    Aseg7_node.interface.recon_only = os.environ['RECON_ONLY']

    return Aseg7_node


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
    subjects_dir_test = Path('/mnt/ngshare/DeepPrep_workflow_test/UKB_Recon')
    bold_preprocess_dir_test = Path('/mnt/ngshare/DeepPrep_workflow_test/UKB_BoldPreprocess')
    workflow_cached_dir_test = '/mnt/ngshare/DeepPrep_workflow_test/UKB_WorkflowfsT1'
    vxm_model_path_test = '//model/voxelmorph'
    mni152_brain_mask_test = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    resource_dir_test = '//resource'

    if not subjects_dir_test.exists():
        subjects_dir_test.mkdir(parents=True, exist_ok=True)

    if not bold_preprocess_dir_test.exists():
        bold_preprocess_dir_test.mkdir(parents=True, exist_ok=True)

    os.environ['SUBJECTS_DIR'] = str(subjects_dir_test)
    os.environ['BOLD_PREPROCESS_DIR'] = str(bold_preprocess_dir_test)
    os.environ['WORKFLOW_CACHED_DIR'] = str(workflow_cached_dir_test)
    os.environ['FASTSURFER_HOME'] = str(fastsurfer_home)
    os.environ['FREESURFER_HOME'] = str(freesurfer_home)
    os.environ['FASTCSR_HOME'] = str(fastcsr_home)
    os.environ['FEATREG_HOME'] = str(featreg_home)
    os.environ['BIDS_DIR'] = bids_data_dir_test
    os.environ['VXM_MODEL_PATH'] = str(vxm_model_path_test)
    os.environ['MNI152_BRAIN_MASK'] = str(mni152_brain_mask_test)
    os.environ['RESOURCE_DIR'] = str(resource_dir_test)
    os.environ['DEEPPREP_DEVICES'] = 'cuda'

    atlas_type_test = 'MNI152_T1_2mm'
    task_test = 'rest'
    preprocess_method_test = 'task'

    os.environ['DEEPPREP_ATLAS_TYPE'] = atlas_type_test
    os.environ['DEEPPREP_TASK'] = task_test
    os.environ['DEEPPREP_PREPROCESS_METHOD'] = preprocess_method_test

    os.environ['RECON_ONLY'] = 'True'
    os.environ['BOLD_ONLY'] = 'False'

    subject_id_test = 'sub-R07renuorignu'
    t1w_files = ['/mnt/ngshare/DeepPrep_workflow_test/UKB_BIDS/sub-R07/ses-01/anat/sub-R07_ses-01_T1w.nii.gz']
    # t1w_files = ['/mnt/ngshare/DeepPrep_workflow_test/sub-R07T1_ses-01_T1w.nii.gz']

    # 测试
    node = create_OrigAndRawavg_node(subject_id=subject_id_test, t1w_files=t1w_files)
    node.run()

    node = create_N4BiasCorrect_node(subject_id=subject_id_test)
    node.run()

    node = create_TalairachAndNu_node(subject_id=subject_id_test)
    node.run()

    node = create_Segment_node(subject_id=subject_id_test)
    node.run()

    node = create_Noccseg_node(subject_id=subject_id_test)
    node.run()

    node = create_Brainmask_node(subject_id=subject_id_test)
    node.run()

    node = create_UpdateAseg_node(subject_id=subject_id_test)
    node.run()

    node = create_Filled_node(subject_id=subject_id_test)
    node.run()

    node = create_FastCSR_node(subject_id=subject_id_test)
    node.run()

    node = create_WhitePreaparc1_node(subject_id=subject_id_test)
    node.run()

    node = create_InflatedSphere_node(subject_id=subject_id_test)
    node.run()

    node = create_FeatReg_node(subject_id=subject_id_test)
    node.run()

    node = create_JacobianAvgcurvCortparc_node(subject_id=subject_id_test)
    node.run()

    node = create_WhitePialThickness1_node(subject_id=subject_id_test)
    node.run()

    exit()

    node = create_Curvstats_node(subject_id=subject_id_test)
    node.run()

    node = create_Aseg7_node(subject_id=subject_id_test)
    node.run()
    sub_node = node.interface.create_sub_node()
    sub_node.run()
    exit()


if __name__ == '__main__':
    create_node_t()  # 测试
