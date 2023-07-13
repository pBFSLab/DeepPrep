from deepprep.interface.boldpost_node import *
from deepprep.interface.node_source import Source

from nipype import Node

"""环境变量
subjects_dir = Path(settings.SUBJECTS_DIR)
bold_preprocess_dir = Path(settings.BOLD_PREPROCESS_DIR)
workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
fastsurfer_home = Path(settings.FASTSURFER_HOME)
freesurfer_home = Path(settings.FREESURFER_HOME)
fastcsr_home = Path(settings.FASTCSR_HOME)
featreg_home = Path(settings.FEATREG_HOME)
python_interpret = sys.executable
"""


def create_RestGauss_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    derivative_deepprep_path = Path(settings.BOLD_PREPROCESS_DIR)
    data_path = Path(settings.BIDS_DIR)
    subjects_dir = Path(settings.SUBJECTS_DIR)

    RestGauss_node = Node(RestGauss(), name=f'{subject_id}_fMRI_RestGauss_node')
    RestGauss_node.inputs.subject_id = subject_id
    RestGauss_node.inputs.subjects_dir = subjects_dir
    RestGauss_node.inputs.data_path = data_path
    RestGauss_node.inputs.task = task
    RestGauss_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    RestGauss_node.inputs.atlas_type = atlas_type
    RestGauss_node.inputs.preprocess_method = preprocess_method

    RestGauss_node.base_dir = workflow_cached_dir / subject_id
    RestGauss_node.source = Source(CPU_n=0, GPU_MB=0, RAM_MB=2000, IO_write_MB=0, IO_read_MB=0)

    return RestGauss_node


def create_RestBandpass_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    derivative_deepprep_path = Path(settings.BOLD_PREPROCESS_DIR)
    data_path = Path(settings.BIDS_DIR)

    RestBandpass_node = Node(RestBandpass(), name=f'{subject_id}_fMRI_RestBandpass_node')
    RestBandpass_node.inputs.subject_id = subject_id
    RestBandpass_node.inputs.data_path = data_path
    RestBandpass_node.inputs.task = task
    RestBandpass_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    RestBandpass_node.inputs.atlas_type = atlas_type
    RestBandpass_node.inputs.preprocess_method = preprocess_method

    RestBandpass_node.base_dir = workflow_cached_dir / subject_id
    RestBandpass_node.source = Source(CPU_n=0, GPU_MB=0, RAM_MB=3000, IO_write_MB=0, IO_read_MB=0)

    return RestBandpass_node


def create_RestRegression_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    derivative_deepprep_path = Path(settings.BOLD_PREPROCESS_DIR)
    data_path = Path(settings.BIDS_DIR)
    subjects_dir = Path(settings.SUBJECTS_DIR)

    RestRegression_node = Node(RestRegression(), name=f'{subject_id}_fMRI_RestRegression_node')
    RestRegression_node.inputs.subject_id = subject_id
    RestRegression_node.inputs.subjects_dir = subjects_dir
    RestRegression_node.inputs.data_path = data_path
    RestRegression_node.inputs.task = task
    RestRegression_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    RestRegression_node.inputs.atlas_type = atlas_type
    RestRegression_node.inputs.preprocess_method = preprocess_method

    RestRegression_node.base_dir = workflow_cached_dir / subject_id
    RestRegression_node.source = Source(CPU_n=0, GPU_MB=0, RAM_MB=4000, IO_write_MB=20, IO_read_MB=40)

    return RestRegression_node


def create_Smooth_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    derivative_deepprep_path = Path(settings.BOLD_PREPROCESS_DIR)
    data_path = Path(settings.BIDS_DIR)
    mni152_brain_mask = Path(settings.BRAIN_MASK)

    Smooth_node = Node(Smooth(), name=f'{subject_id}_Smooth_node')
    Smooth_node.inputs.subject_id = subject_id
    Smooth_node.inputs.task = task
    Smooth_node.inputs.data_path = data_path
    Smooth_node.inputs.atlas_type = atlas_type
    Smooth_node.inputs.preprocess_method = preprocess_method
    Smooth_node.inputs.MNI152_T1_2mm_brain_mask = mni152_brain_mask
    Smooth_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    Smooth_node.base_dir = workflow_cached_dir / subject_id
    Smooth_node.source = Source(CPU_n=0, GPU_MB=0, RAM_MB=7500, IO_write_MB=0, IO_read_MB=0)

    return Smooth_node


def create_node_t(settings):
    from interface.run import set_envrion
    set_envrion(threads=8)

    pwd = Path.cwd()
    pwd = pwd.parent
    fastsurfer_home = pwd / "FastSurfer"
    freesurfer_home = Path('/usr/local/freesurfer720')
    fastcsr_home = pwd / "FastCSR"
    featreg_home = pwd / "FeatReg"

    bids_data_dir_test = '/mnt/ngshare/temp/UKB'
    subjects_dir_test = Path('/mnt/ngshare/temp/UKB_Recon')
    bold_preprocess_dir_test = Path('/mnt/ngshare/temp/UKB_BoldPreprocess')
    workflow_cached_dir_test = '/mnt/ngshare/temp/UKB_Workflow'
    vxm_model_path_test = '/home/anning/workspace/DeepPrep/deepprep/model/voxelmorph'
    mni152_brain_mask_test = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    resource_dir_test = '/home/anning/workspace/DeepPrep/deepprep/resource'

    if not subjects_dir_test.exists():
        subjects_dir_test.mkdir(parents=True, exist_ok=True)

    if not bold_preprocess_dir_test.exists():
        bold_preprocess_dir_test.mkdir(parents=True, exist_ok=True)

    subject_id_test = 'sub-1000525'

    # t1w_files = ['/mnt/ngshare/DeepPrep_workflow_test/UKB_BIDS/sub-1000037/ses-02/anat/sub-1000037_ses-02_T1w.nii.gz']

    settings.SUBJECTS_DIR = str(subjects_dir_test)
    settings.BOLD_PREPROCESS_DIR = str(bold_preprocess_dir_test)
    settings.WORKFLOW_CACHED_DIR = str(workflow_cached_dir_test)
    settings.FASTSURFER_HOME = str(fastsurfer_home)
    settings.FREESURFER_HOME = str(freesurfer_home)
    settings.FASTCSR_HOME = str(fastcsr_home)
    settings.FEATREG_HOME = str(featreg_home)
    settings.BIDS_DIR = bids_data_dir_test
    settings.VXM_MODEL_PATH = str(vxm_model_path_test)
    settings.BRAIN_MASK = str(mni152_brain_mask_test)
    settings.RESOURCE_DIR = str(resource_dir_test)
    settings.DEVICE = 'cuda'

    atlas_type_test = 'MNI152_T1_2mm'
    task_test = 'rest'
    preprocess_method_test = 'task'

    settings.FMRI.ATLAS_SPACE = atlas_type_test
    settings.FMRI.TASK = task_test
    settings.FMRI.PREPROCESS_TYPE = preprocess_method_test

    settings.RECON_ONLY = 'False'
    settings.BOLD_ONLY = 'False'

    print('#####################################################7#####################################################')
    node = create_RestGauss_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                                 preprocess_method=preprocess_method_test, settings=settings)
    node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()
    print('#####################################################8#####################################################')
    node = create_RestBandpass_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                                    preprocess_method=preprocess_method_test, settings=settings)
    node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()
    print('#####################################################6#####################################################')

    node = create_RestRegression_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                                      preprocess_method=preprocess_method_test, settings=settings)
    node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()
    print('####################################################11####################################################')
    node = create_Smooth_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                              preprocess_method=preprocess_method_test, settings=settings)
    node.run()


if __name__ == '__main__':
    from config import settings as settings_main
    create_node_t(settings_main)  # 测试
