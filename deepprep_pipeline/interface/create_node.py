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
    subjects_dir = os.environ['SUBJECTS_DIR']
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


def create_Noccseg_node(subject_id: str, subjects_dir: Path, base_dir: Path, python_interpret: Path,
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


######################## Second Half ######################
def create_InflatedSphere_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    lh_white_preaparc_file = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
    rh_white_preaparc_file = subjects_dir / subject_id / "surf" / "rh.white.preaparc"

    Inflated_Sphere_node = Node(InflatedSphere(), f'Inflated_Sphere_node')
    Inflated_Sphere_node.inputs.threads = 8
    Inflated_Sphere_node.inputs.subjects_dir = subjects_dir
    Inflated_Sphere_node.inputs.subject_id = subject_id
    Inflated_Sphere_node.inputs.lh_white_preaparc_file = lh_white_preaparc_file
    Inflated_Sphere_node.inputs.rh_white_preaparc_file = rh_white_preaparc_file

    return Inflated_Sphere_node


def create_FeatReg_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    featreg_home = Path(os.environ["FEATREG_HOME"])

    python_interpret = sys.executable
    featreg_py = featreg_home / "featreg" / 'predict.py'  # inference script

    featreg_node = Node(FeatReg(), f'featreg_node')
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

    return featreg_node


def create_JacobianAvgcurvCortparc_node(subject_id: str):
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])

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

    JacobianAvgcurvCortparc_node = Node(JacobianAvgcurvCortparc(), f'JacobianAvgcurvCortparc_node')
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

    return JacobianAvgcurvCortparc_node

def create_WhitePialThickness1_node(subject_id: str):

    threads = 8
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])

    white_pial_thickness1 = Node(WhitePialThickness1(), name="white_pial_thickness1")
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

    return white_pial_thickness1


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
