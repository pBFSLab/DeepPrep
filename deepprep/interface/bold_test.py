# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : NingAnMe <ninganme0317@gmail.com>
# @Author : Cong Lin       @Email : lincong <lincong8722@gmail.com>
# @Author : Youjia Zhang   @Email : youjia <ireneyou33@gmail.com>
# @Author : Zhenyu Sun     @Email : Kid-sunzhenyu <sun25939789@gmail.com>

from bold import BoldSkipReorient, MotionCorrection, Stc, Register, MkBrainmask, VxmRegistraion, RestGauss, \
    RestBandpass, RestRegression, VxmRegNormMNI152, Smooth, MkTemplate
from pathlib import Path
from nipype import Node
import os
from run import set_envrion


def BoldSkipReorient_test():
    # threads = 8
    task = 'motor'
    subject_id = 'sub-MSC01'
    subj = 'MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path

    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'

    BoldSkipReorient_node = Node(BoldSkipReorient(), name='BoldSkipReorient_node')
    BoldSkipReorient_node.inputs.subject_id = subject_id
    BoldSkipReorient_node.inputs.subj = subj
    BoldSkipReorient_node.inputs.data_path = data_path
    BoldSkipReorient_node.inputs.task = task
    BoldSkipReorient_node.inputs.deepprep_subj_path = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id
    BoldSkipReorient_node.inputs.preprocess_dir = preprocess_dir
    BoldSkipReorient_node.run()


def MotionCorrection_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/home/lincong/pack/temp/DeepPrep/MSC')
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'
    # preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'
    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    MotionCorrection_node = Node(MotionCorrection(), name='MotionCorrection_node')
    MotionCorrection_node.inputs.preprocess_dir = preprocess_dir
    MotionCorrection_node.inputs.subject_id = subject_id
    MotionCorrection_node.run()


def Stc_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/home/lincong/pack/temp/DeepPrep/MSC')
    subjects_dir = Path('/home/lincong/pack/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'


    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    # preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    # data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')
    # preprocess_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/{subject_id}/tmp/task-{task}')

    Stc_node = Node(Stc(), name='stc_node')
    Stc_node.inputs.subject_id = subject_id
    Stc_node.inputs.preprocess_dir = preprocess_dir
    Stc_node.run()


def Register_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    Register_node = Node(Register(), name='register_node')
    Register_node.inputs.subject_id = subject_id
    Register_node.inputs.preprocess_dir = preprocess_dir
    Register_node.run()


def MkBrainmask_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    Mkbrainmask_node = Node(MkBrainmask(), name='mkbrainmask_node')
    Mkbrainmask_node.inputs.subject_id = subject_id
    Mkbrainmask_node.inputs.subjects_dir = subjects_dir
    Mkbrainmask_node.inputs.preprocess_dir = preprocess_dir

    Mkbrainmask_node.run()


def VxmRegistraion_test():
    task = 'motor'
    subject_id = 'sub-0025427'
    data_path = Path(f'/mnt/ngshare/deepprep_test/HNU')
    # derivative_deepprep_path = data_path / 'derivatives' / 'deepprep'
    # tmpdir = derivative_deepprep_path / subject_id / 'tmp'
    # subjects_dir = derivative_deepprep_path / 'Recon'

    # data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    # data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    # preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'
    preprocess_dir = data_path / subject_id / 'tmp' / f'task-{task}'
    # subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/Recon')
    subjects_dir = data_path
    # derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'
    # derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_bold_test'
    derivative_deepprep_path = Path(f'/mnt/ngshare/deepprep_test/HNU')
    tmpdir = derivative_deepprep_path / subject_id / 'tmp'


    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    atlas_type = 'MNI152_T1_2mm'
    VxmRegistraion_node = Node(VxmRegistraion(), name='VxmRegistraion_node')
    VxmRegistraion_node.inputs.subject_id = subject_id
    VxmRegistraion_node.inputs.norm = subjects_dir / subject_id / 'mri' / 'norm.mgz'
    # VxmRegistraion_node.inputs.model_file = Path(__file__).parent.parent / 'model' / 'voxelmorph' / atlas_type / 'model.h5'
    VxmRegistraion_node.inputs.model_file = Path(__file__).parent.parent / 'model' / 'voxelmorph' / atlas_type / 'model.h5'
    VxmRegistraion_node.inputs.model_path = Path(__file__).parent.parent / 'model' / 'voxelmorph' / atlas_type
    VxmRegistraion_node.inputs.atlas_type = atlas_type
    VxmRegistraion_node.inputs.data_path = data_path
    VxmRegistraion_node.inputs.preprocess_dir = preprocess_dir
    VxmRegistraion_node.inputs.deepprep_subj_path = derivative_deepprep_path / subject_id
    VxmRegistraion_node.inputs.vxm_warp = tmpdir / 'warp.nii.gz'
    VxmRegistraion_node.inputs.vxm_warped = tmpdir / 'warped.nii.gz'
    VxmRegistraion_node.inputs.trf = tmpdir / f'{subject_id}_affine.mat'
    VxmRegistraion_node.inputs.warp = tmpdir / f'{subject_id}_warp.nii.gz'
    VxmRegistraion_node.inputs.warped = tmpdir / f'{subject_id}_warped.nii.gz'
    VxmRegistraion_node.inputs.npz = tmpdir / 'vxminput.npz'
    VxmRegistraion_node.run()


def RestGauss_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    # data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    # subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    # os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    # preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/Recon')
    preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)


    RestGauss_node = Node(RestGauss(), f'RestGauss_node')
    RestGauss_node.inputs.subject_id = subject_id
    RestGauss_node.inputs.preprocess_dir = preprocess_dir
    RestGauss_node.run()


def RestBandpass_test():
    task = 'rest'
    subject_id = 'sub-MSC01'
    subj = 'MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/Recon')
    preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    RestBandpass_node = Node(RestBandpass(), name='RestBandpass_node')
    RestBandpass_node.inputs.subject_id = subject_id
    RestBandpass_node.inputs.data_path = data_path
    RestBandpass_node.inputs.preprocess_dir = preprocess_dir
    RestBandpass_node.inputs.subj = subj
    RestBandpass_node.inputs.task = task
    RestBandpass_node.run()


def RestRegression_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    subj = 'MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/Recon')
    deepprep_subj_path = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest') / subject_id
    preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    deepprep_subj_path = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test') / subject_id
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    bold_dir = preprocess_dir / subject_id / 'bold'
    fcmri_dir = preprocess_dir / subject_id / 'fcmri'
    RestRegression_node = Node(RestRegression(), f'RestRegression_node')
    RestRegression_node.inputs.subjects_dir = subjects_dir
    RestRegression_node.inputs.subject_id = subject_id
    RestRegression_node.inputs.preprocess_dir = preprocess_dir
    RestRegression_node.inputs.bold_dir = bold_dir
    RestRegression_node.inputs.data_path = data_path
    RestRegression_node.inputs.deepprep_subj_path = deepprep_subj_path
    RestRegression_node.inputs.task = task
    RestRegression_node.inputs.subj = subj
    RestRegression_node.inputs.fcmri_dir = fcmri_dir

    RestRegression_node.run()

def VxmRegNormMNI152_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    subj = 'MSC01'
    preprocess_method = 'task'

    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep'
    deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
    subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'
    deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/Recon')
    preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    deepprep_subj_path = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test') / subject_id
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)


    VxmRegNormMNI152_node = Node(VxmRegNormMNI152(), name='VxmRegNormMNI152_node')
    VxmRegNormMNI152_node.inputs.subject_id = subject_id
    VxmRegNormMNI152_node.inputs.preprocess_dir = preprocess_dir
    VxmRegNormMNI152_node.inputs.subj = subj
    VxmRegNormMNI152_node.inputs.task = task
    VxmRegNormMNI152_node.inputs.data_path = data_path
    VxmRegNormMNI152_node.inputs.deepprep_subj_path = deepprep_subj_path
    VxmRegNormMNI152_node.inputs.preprocess_method = preprocess_method
    VxmRegNormMNI152_node.inputs.norm = derivative_deepprep_path / 'Recon' / f'sub-{subj}' / 'mri' / 'norm.mgz'
    VxmRegNormMNI152_node.run()

def Smooth_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    subj = 'MSC01'
    preprocess_method = 'rest'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep'
    deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
    subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')

    preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'


    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/Recon')
    preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'
    deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'
    deepprep_subj_path = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test') / subject_id
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)


    # os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    Smooth_node = Node(Smooth(), name='Smooth_node')
    Smooth_node.inputs.subject_id = subject_id
    Smooth_node.inputs.subj = subj
    Smooth_node.inputs.task = task
    Smooth_node.inputs.preprocess_dir = preprocess_dir
    Smooth_node.inputs.data_path = data_path
    Smooth_node.inputs.deepprep_subj_path = deepprep_subj_path
    Smooth_node.inputs.preprocess_method = preprocess_method
    Smooth_node.inputs.MNI152_T1_2mm_brain_mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'

    Smooth_node.run()

def MkTemplate_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_wftest' / subject_id / 'tmp' / f'task-{task}'
    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    preprocess_dir = data_path / 'derivatives' / 'deepprep_bold_test' / subject_id / 'tmp' / f'task-{task}'

    MkTemplate_node = Node(MkTemplate(), name='MkTemplate_node')
    MkTemplate_node.inputs.subject_id = subject_id
    MkTemplate_node.inputs.preprocess_dir = preprocess_dir
    MkTemplate_node.run()


if __name__ == '__main__':
    set_envrion()

    # BoldSkipReorient_test()

    # Stc_test()

    # MkTemplate_test()

    # MotionCorrection_test()

    # Register_test()

    # MkBrainmask_test()

    ######## rest #########

    VxmRegistraion_test()

    # RestGauss_test()

    # RestBandpass_test()

    # RestRegression_test()

    # VxmRegNormMNI152_test()

    # Smooth_test()


