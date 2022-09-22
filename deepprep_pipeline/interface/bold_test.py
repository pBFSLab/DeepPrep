from bold import BoldSkipReorient, MotionCorrection, Stc, Register, MkBrainmask
from pathlib import Path
from nipype import Node
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

    # set threads
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads)


def BoldSkipReorient_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'
    preprocess_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/{subject_id}/tmp/task-{task}')
    runs = sorted([d.name for d in (preprocess_dir / subject_id / 'bold').iterdir() if d.is_dir()])

    for run in runs:
        BoldSkipReorient_node = Node(BoldSkipReorient(), name='BoldSkipReorient_node')
        BoldSkipReorient_node.inputs.subject_id = subject_id
        BoldSkipReorient_node.inputs.preprocess_dir = preprocess_dir
        BoldSkipReorient_node.inputs.bold = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld{run}_rest.nii.gz'

        BoldSkipReorient_node.inputs.skip_bold = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld{run}_rest_skip.nii.gz'
        BoldSkipReorient_node.inputs.reorient_skip_bold = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld{run}_rest_reorient_skip.nii.gz'
        BoldSkipReorient_node.run()


def MotionCorrection_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'
    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')
    preprocess_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/{subject_id}/tmp/task-{task}')

    runs = sorted([d.name for d in (preprocess_dir / subject_id / 'bold').iterdir() if d.is_dir()])
    MotionCorrection_node = Node(MotionCorrection(), name='MotionCorrection_node')
    MotionCorrection_node.inputs.preprocess_dir = preprocess_dir
    MotionCorrection_node.inputs.subject_id = subject_id
    for run in runs:
        MotionCorrection_node.inputs.skip_faln = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip_faln.nii.gz'
        MotionCorrection_node.inputs.skip_faln_mc = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
    MotionCorrection_node.run()


def Stc_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'

    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    preprocess_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/{subject_id}/tmp/task-{task}')

    runs = sorted([d.name for d in (preprocess_dir / subject_id / 'bold').iterdir() if d.is_dir()])
    Stc_node = Node(Stc(), f'stc_node')
    Stc_node.inputs.subject_id = subject_id
    Stc_node.inputs.preprocess_dir = preprocess_dir
    for run in runs:
        Stc_node.inputs.skip = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip.nii.gz'
        Stc_node.inputs.faln = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip_faln.nii.gz'
    Stc_node.run()


def Register_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'


    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    preprocess_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep_wftest/{subject_id}/tmp/task-{task}')

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    runs = sorted([d.name for d in (preprocess_dir / subject_id / 'bold').iterdir() if d.is_dir()])
    for run in runs:
        Register_node = Node(Register(), f'register_node')
        Register_node.inputs.subject_id = subject_id
        Register_node.inputs.preprocess_dir = preprocess_dir
        Register_node.inputs.mov = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
        Register_node.inputs.reg = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip_faln_mc.register.dat'
        Register_node.run()


def MkBrainmask_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    subjects_dir = Path('/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'
    runs = sorted([d.name for d in (preprocess_dir / subject_id / 'bold').iterdir() if d.is_dir()])
    for run in runs:
        Mkbrainmask_node = Node(MkBrainmask(), f'mkbrainmask_node')
        Mkbrainmask_node.inputs.subject_id = subject_id
        Mkbrainmask_node.inputs.preprocess_dir = preprocess_dir
        Mkbrainmask_node.inputs.seg = subjects_dir / subject_id / 'mri/aparc+aseg.mgz'
        Mkbrainmask_node.inputs.targ = subjects_dir / subject_id / 'mri/brainmask.mgz'
        Mkbrainmask_node.inputs.func = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}.func.aseg.nii'
        Mkbrainmask_node.inputs.mov = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'

        Mkbrainmask_node.inputs.reg = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip_faln_mc.register.dat'
        Mkbrainmask_node.inputs.wm = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}.func.wm.nii.gz'
        Mkbrainmask_node.inputs.vent = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}.func.ventricles.nii.gz'
        Mkbrainmask_node.inputs.mask = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}.brainmask.nii.gz'
        Mkbrainmask_node.inputs.binmask = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}.brainmask.bin.nii.gz'
        Mkbrainmask_node.run()


if __name__ == '__main__':
    set_envrion()

    # BoldSkipReorient_test()

    # MotionCorrection_test()

    # Stc_test()

    Register_test()
