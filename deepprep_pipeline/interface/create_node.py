from nipype import Node

from interface.freesurfer_node import *
from interface.bold_node import *
from interface.fastcsr_node import *
from interface.fastsurfer_node import *


def create_origandrawavg_node(subject_id: str, subjects_dir: Path, bold_result_dir: Path, base_dir: Path,
                              bids_dir: Path, t1w_files: list):

    subject_id = subject_id
    # t1w_files = [
    #     f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
    # ]
    origandrawavg_node = Node(OrigAndRawavg(), f'{subject_id}_origandrawavg_node')
    origandrawavg_node.inputs.t1w_files = t1w_files
    origandrawavg_node.inputs.subjects_dir = subjects_dir
    origandrawavg_node.inputs.subject_id = subject_id
    origandrawavg_node.inputs.threads = 1
    origandrawavg_node.base_dir = base_dir
    return origandrawavg_node


def create_fastcsr_node(subject_id: str, subjects_dir: Path, bold_result_dir: Path, base_dir: Path,
                        python_interpret: Path,
                        fastcsr_home: Path):
    # fastcsr_home = pwd.parent / "FastCSR"
    fastcsr_py = fastcsr_home / 'pipeline.py'  # inference script

    fastcsr_node = Node(FastCSR(), f'{subject_id}_fastcsr_node')
    fastcsr_node.inputs.python_interpret = python_interpret
    fastcsr_node.inputs.fastcsr_py = fastcsr_py

    subjects_dir = '/mnt/ngshare/Data_Mirror/pipeline_test'
    subject_id = 'sub-MSC01'

    os.environ['SUBJECTS_DIR'] = subjects_dir

    fastcsr_node.inputs.subjects_dir = subjects_dir
    fastcsr_node.inputs.subject_id = subject_id
    fastcsr_node.inputs.orig_file = Path(subjects_dir) / subject_id / 'mri/orig.mgz'
    fastcsr_node.inputs.filled_file = Path(subjects_dir) / subject_id / 'mri/filled.mgz'
    fastcsr_node.inputs.aseg_presurf_file = Path(subjects_dir) / subject_id / 'mri/aseg.presurf.mgz'
    fastcsr_node.inputs.brainmask_file = Path(subjects_dir) / subject_id / 'mri/brainmask.mgz'
    fastcsr_node.inputs.wm_file = Path(subjects_dir) / subject_id / 'mri/wm.mgz'
    fastcsr_node.inputs.brain_finalsurfs_file = Path(subjects_dir) / subject_id / 'mri/brain.finalsurfs.mgz'
    return fastcsr_node


def creat_Segment_node(subject_id: str, subjects_dir: Path, python_interpret: Path, fastsurfer_home: Path):

    fastsurfer_eval = fastsurfer_home / 'FastSurferCNN' / 'eval.py'  # inference script
    weight_dir = fastsurfer_home / 'checkpoints'  # model checkpoints dir


    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    network_sagittal_path = weight_dir / "Sagittal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_coronal_path = weight_dir / "Coronal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_axial_path = weight_dir / "Axial_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"

    segment_node = Node(Segment(), f'{subject_id}_segment_node')
    segment_node.inputs.python_interpret = python_interpret
    segment_node.inputs.in_file = subjects_dir / subject_id / "mri" / "orig.mgz"
    segment_node.inputs.eval_py = fastsurfer_eval
    segment_node.inputs.network_sagittal_path = network_sagittal_path
    segment_node.inputs.network_coronal_path = network_coronal_path
    segment_node.inputs.network_axial_path = network_axial_path

    segment_node.inputs.aparc_DKTatlas_aseg_deep = subjects_dir / subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    segment_node.inputs.aparc_DKTatlas_aseg_orig = subjects_dir / subject_id / "mri" / "aparc.DKTatlas+aseg.orig.mgz"

    segment_node.inputs.conformed_file = subjects_dir / subject_id / "mri" / "conformed.mgz"
    return segment_node


def creat