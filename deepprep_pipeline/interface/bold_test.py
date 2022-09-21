from bold import BoldSkipReorient, MotionCorrection, Stc, Register, MkBrainmask, VxmRegistraion
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
    runs = sorted([d.name for d in (preprocess_dir / subject_id / 'bold').iterdir() if d.is_dir()])
    Stc_node = Node(Stc(), name='stc_node')
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
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'
    runs = sorted([d.name for d in (preprocess_dir / subject_id / 'bold').iterdir() if d.is_dir()])
    for run in runs:
        Register_node = Node(Register(), name='register_node')
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
        Mkbrainmask_node = Node(MkBrainmask(), name='mkbrainmask_node')
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

def VxmRegistraion_test():
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/DATA/lincong/temp/DeepPrep/MSC')
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep'
    tmpdir = derivative_deepprep_path / subject_id / 'tmp'
    freesurfer_subjects_path = derivative_deepprep_path / 'Recon'
    os.environ['SUBJECTS_DIR'] = str(freesurfer_subjects_path)
    atlas_type = 'MNI152_T1_2mm'
    VxmRegistraion_node = Node(VxmRegistraion(), name='VxmRegistraion_node')
    VxmRegistraion_node.inputs.subject_id = subject_id
    VxmRegistraion_node.inputs.norm = freesurfer_subjects_path / subject_id / 'mri' / 'norm.mgz'
    VxmRegistraion_node.inputs.model_file = Path(__file__).parent.parent / 'model' / 'voxelmorph' / atlas_type / 'model.h5'
    VxmRegistraion_node.inputs.atlas_type = atlas_type
    VxmRegistraion_node.inputs.atlas = Path(__file__).parent.parent / 'model' / 'voxelmorph' / atlas_type / 'MNI152_T1_2mm_brain.nii.gz'
    VxmRegistraion_node.inputs.vxm_atlas = Path(__file__).parent.parent / 'model' / 'voxelmorph' / atlas_type / 'MNI152_T1_2mm_brain_vxm.nii.gz'
    VxmRegistraion_node.inputs.vxm_atlas_npz = Path(__file__).parent.parent / 'model' / 'voxelmorph' / atlas_type / 'MNI152_T1_2mm_brain_vxm.npz'
    VxmRegistraion_node.inputs.vxm2atlas_trf = Path(__file__).parent.parent / 'model' / 'voxelmorph' / atlas_type / 'MNI152_T1_2mm_vxm2atlas.mat'

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
    data_path = Path(f'/mnt/DATA/lincong/mnt/DATA/lincong/temp/DeepPrep/MSC')
    subjects_dir = Path('/mnt/DATA/lincong/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'
    RestGauss_node = Node(RestGauss(), f'RestGauss_node')
    RestGauss_node.inputs.fcmri = preprocess_dir / subject_id / 'fcmri'
    runs = sorted([d.name for d in (preprocess_dir / subject_id / 'bold').iterdir() if d.is_dir()])
    for run in runs:
        RestGauss_node.inputs.subject_id = subject_id
        RestGauss_node.inputs.preprocess_dir = preprocess_dir
        RestGauss_node.inputs.mc = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz'
        RestGauss_node.run()


def RestBandpass_test():
    task = 'motor'
    subject_id = 'sub-MSC01'
    data_path = Path(f'/mnt/DATA/lincong/mnt/DATA/lincong/temp/DeepPrep/MSC')
    subjects_dir = Path('/mnt/DATA/lincong/mnt/DATA/lincong/temp/DeepPrep/MSC/derivatives/deepprep/Recon')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    preprocess_dir = data_path / 'derivatives' / 'deepprep' / subject_id / 'tmp' / f'task-{task}'
    layout = bids.BIDSLayout(str(data_path), derivatives=False)
    RestBandpass_node = Node(RestBandpass(), f'RestBandpass_node')
    RestBandpass_node.inputs.fcmri = preprocess_dir / subject_id / 'fcmri'
    RestBandpass_node.inputs.bold = preprocess_dir / subject_id / 'bold'
    RestBandpass_node.inputs.bids_bolds = layout.get(subject=subject_id, suffix='bold', extension='.nii.gz')
    runs = sorted([d.name for d in (preprocess_dir / subject_id / 'bold').iterdir() if d.is_dir()])
    for run in runs:
        RestBandpass_node.inputs.subject_id = subject_id
        RestBandpass_node.inputs.preprocess_dir = preprocess_dir
        RestBandpass_node.inputs.bpss = preprocess_dir / subject_id / 'bold' / run / f'{subject_id}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss.nii.gz'
        RestBandpass_node.run()
if __name__ == '__main__':
    set_envrion()
    # BoldSkipReorient_test()
    # MotionCorrection_test()
    # Stc_test()
    VxmRegistraion_test()

