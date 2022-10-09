from nipype import Node

from interface.freesurfer_node import *
from interface.bold_node import *
from interface.fastcsr_node import *
from interface.fastsurfer_node import *
from interface.featreg_node import *
from interface.node_source import Source

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


def create_fastcsr_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    fastcsr_home = Path(os.environ['FASTCSR_HOME'])
    workflow_cached_dir = os.environ['WORKFLOW_CACHED_DIR']
    python_interpret = sys.executable

    fastcsr_py = fastcsr_home / 'pipeline.py'  # inference script

    fastcsr_node = Node(FastCSR(), f'{subject_id}_fastcsr_node')
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
    return fastcsr_node


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

    segment_node.inputs.base_dir = workflow_cached_dir
    segment_node.inputs.fastsurfer_home = fastsurfer_home

    segment_node.source = Source(CPU_n=1, GPU_MB=8000)
    return segment_node


def creat_Noccseg_node(subject_id: str, subjects_dir: Path, base_dir: Path, python_interpret: Path,
                       fastsurfer_home: Path):
    reduce_to_aseg_py = fastsurfer_home / 'recon_surf' / 'reduce_to_aseg.py'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    noccseg_node = Node(Noccseg(), f'noccseg_node')
    noccseg_node.inputs.python_interpret = python_interpret
    noccseg_node.inputs.reduce_to_aseg_py = reduce_to_aseg_py
    noccseg_node.inputs.in_file = subjects_dir / subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"

    noccseg_node.inputs.mask_file = subjects_dir / subject_id / 'mri/mask.mgz'
    noccseg_node.inputs.aseg_noCCseg_file = subjects_dir / subject_id / 'mri/aseg.auto_noCCseg.mgz'

    noccseg_node.inputs.base_dir = base_dir
    return noccseg_node


def creat_VxmRegistraion_node(subject_id: str, vxm_model_path: Path, atlas_type: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    derivative_deepprep_path = os.environ['BOLD_PREPROCESS_DIR']
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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
    derivative_deepprep_path = Path(os.environ['BOLD_PREPROCESS_DIR'])
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

    subject_id_test = 'sub-1000896'
    t1w_files = ['/mnt/ngshare/DeepPrep_workflow_test/UKB_BIDS/sub-1000037/ses-02/anat/sub-1000037_ses-02_T1w.nii.gz']

    os.environ['SUBJECTS_DIR'] = str(subjects_dir_test)
    os.environ['BOLD_PREPROCESS_DIR'] = str(bold_preprocess_dir_test)
    os.environ['WORKFLOW_CACHED_DIR'] = str(workflow_cached_dir_test)
    os.environ['FASTSURFER_HOME'] = str(fastsurfer_home)
    os.environ['FREESURFER_HOME'] = str(freesurfer_home)
    os.environ['FASTCSR_HOME'] = str(fastcsr_home)
    os.environ['FEATREG_HOME'] = str(featreg_home)
    os.environ['BIDS_DIR'] = bids_data_dir_test

    # 测试
    vxm_model_path_test = Path('/home/zhenyu/workspace/DeepPrep/deepprep_pipeline/model/voxelmorph')
    atlas_type_test = 'MNI152_T1_2mm'


    node = creat_VxmRegistraion_node(subject_id=subject_id_test, vxm_model_path=vxm_model_path_test, atlas_type=atlas_type_test)
    node.run()
    sub_node = node.interface.create_sub_node()
    sub_node.run()


if __name__ == '__main__':
    create_node_t()  # 测试
