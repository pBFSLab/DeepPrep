from nipype import Node

from interface.freesurfer_node import *
from interface.bold_node import *
from interface.fastcsr_node import *
from interface.fastsurfer_node import *
from interface.run import Source

def create_origandrawavg_node(subject_id: str, t1w_files: list):

    # t1w_files = [
    #     f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
    # ]
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])

    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    origandrawavg_node = Node(OrigAndRawavg(), f'{subject_id}_origandrawavg_node')
    origandrawavg_node.inputs.t1w_files = t1w_files
    origandrawavg_node.inputs.subjects_dir = subjects_dir
    origandrawavg_node.inputs.subject_id = subject_id
    origandrawavg_node.inputs.threads = 1
    origandrawavg_node.base_dir = workflow_cached_dir
    origandrawavg_node.source = Source()

    return origandrawavg_node


def create_Segment_node(subject_id: str):

    fastsurfer_home = os.environ['FASTSURFER_HOME']
    fastsurfer_eval = fastsurfer_home / 'FastSurferCNN' / 'eval.py'  # inference script
    weight_dir = fastsurfer_home / 'checkpoints'  # model checkpoints dir

    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

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
    segment_node.source = Source()
    return segment_node


def create_Noccseg_node(subject_id: str):
    fastsurfer_home = os.environ['FASTSURFER_HOME']
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable
    reduce_to_aseg_py = fastsurfer_home / 'recon_surf' / 'reduce_to_aseg.py'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)



    noccseg_node = Node(Noccseg(), f'noccseg_node')
    noccseg_node.inputs.python_interpret = python_interpret
    noccseg_node.inputs.reduce_to_aseg_py = reduce_to_aseg_py
    noccseg_node.inputs.subject_id = subject_id
    noccseg_node.inputs.subjects_dir = subjects_dir

    noccseg_node.base_dir = workflow_cached_dir
    noccseg_node.source = Source()
    return noccseg_node


def create_N4BiasCorrect_node(subject_id: str):
    fastsurfer_home = os.environ['FASTSURFER_HOME']
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])
    python_interpret = sys.executable
    sub_mri_dir = subjects_dir / subject_id / "mri"
    correct_py = fastsurfer_home / "recon_surf" / "N4_bias_correct.py"

    orig_file = sub_mri_dir / "orig.mgz"
    mask_file = sub_mri_dir / "mask.mgz"


    N4_bias_correct_node = Node(N4BiasCorrect(), name="N4_bias_correct_node")
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

    talairach_and_nu_node = Node(TalairachAndNu(), name="talairach_and_nu_node")
    talairach_and_nu_node.inputs.subjects_dir = subjects_dir
    talairach_and_nu_node.inputs.subject_id = subject_id
    talairach_and_nu_node.inputs.threads = 8
    talairach_and_nu_node.inputs.mni305 = mni305
    talairach_and_nu_node.inputs.orig_nu_file = orig_nu_file
    talairach_and_nu_node.inputs.orig_file = orig_file

    talairach_and_nu_node.base_dir = workflow_cached_dir
    talairach_and_nu_node.source = Source()

    return  talairach_and_nu_node

def create_Brainmask_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    workflow_cached_dir = Path(os.environ['WORKFLOW_CACHED_DIR'])

    brainmask_node = Node(Brainmask(), name='brainmask_node')
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
    updateaseg_node = Node(UpdateAseg(), name='updateaseg_node')
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


    filled_node = Node(Filled(), name='filled_node')
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

    fastcsr_node = Node(FastCSR(), f'fastcsr_node')
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

    white_preaparc1 = Node(WhitePreaparc1(), name="white_preaparc1_node")
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

    SampleSegmentationToSurfave_node = Node(SampleSegmentationToSurfave(), name='SampleSegmentationToSurfave_node')
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