from pathlib import Path
from nipype import Node, Workflow
from bold import VxmRegistraion, BoldSkipReorient, Stc, MkTemplate, \
    MotionCorrection, Register, MkBrainmask, VxmRegNormMNI152, RestGauss, \
    RestBandpass, RestRegression, Smooth


def init_single_bold_common_wf(subject_id: str, subj: str, task: str, preprocess_method: str,
                             data_path: Path, derivative_deepprep_path: Path,
                             subjects_dir: Path, workdir: Path, preprocess_dir: Path):
    single_bold_common_wf = Workflow(name=f'single_bold_{subject_id.replace("-", "_")}_wf')

    ########################################### BOLD - Common ###########################################
    # bold skip reorient
    BoldSkipReorient_node = Node(BoldSkipReorient(), name='BoldSkipReorient_node')
    BoldSkipReorient_node.inputs.subject_id = subject_id
    BoldSkipReorient_node.inputs.data_path = data_path
    BoldSkipReorient_node.inputs.task = task
    BoldSkipReorient_node.inputs.preprocess_dir = preprocess_dir
    # BoldSkipReorient_node.run()

    # Stc
    Stc_node = Node(Stc(), name='stc_node')
    Stc_node.inputs.subject_id = subject_id
    # Stc_node.inputs.preprocess_dir = preprocess_dir
    # Stc_node.run()

    # make template
    MkTemplate_node = Node(MkTemplate(), name='MkTemplate_node')
    MkTemplate_node.inputs.subject_id = subject_id

    # Motion correction
    MotionCorrection_node = Node(MotionCorrection(), name='MotionCorrection_node')
    # MotionCorrection_node.inputs.preprocess_dir = preprocess_dir
    MotionCorrection_node.inputs.subject_id = subject_id

    # bb register
    Register_node = Node(Register(), name='register_node')
    Register_node.inputs.subject_id = subject_id
    # Register_node.inputs.preprocess_dir = preprocess_dir

    # Make brainmask
    Mkbrainmask_node = Node(MkBrainmask(), name='mkbrainmask_node')
    Mkbrainmask_node.inputs.subject_id = subject_id
    Mkbrainmask_node.inputs.subjects_dir = subjects_dir
    # Mkbrainmask_node.inputs.preprocess_dir = preprocess_dir

    # create workflow

    single_bold_common_wf.connect([
        (BoldSkipReorient_node, Stc_node, [("preprocess_dir", "preprocess_dir"),
                                              ]),
        (Stc_node, MkTemplate_node, [("preprocess_dir", "preprocess_dir"),
                                ]),
        (MkTemplate_node, MotionCorrection_node, [("preprocess_dir", "preprocess_dir"),
                                              ]),
        (MotionCorrection_node, Register_node, [("preprocess_dir", "preprocess_dir"),
                                               ]),
        (MotionCorrection_node, Mkbrainmask_node, [("preprocess_dir", "preprocess_dir"),
                                                    ]),
    ])

    return single_bold_common_wf


def init_single_bold_rest_wf(subject_id: str, subj: str, task: str, preprocess_method: str,
                             data_path: Path, derivative_deepprep_path: Path,
                             subjects_dir: Path, workdir: Path, preprocess_dir: Path):
    single_bold_rest_wf = Workflow(name=f'single_bold_{subject_id.replace("-", "_")}_wf')

    # Voxelmorph registration
    VxmRegistraion_node = Node(VxmRegistraion(), name='VxmRegistraion_node')
    atlas_type = 'MNI152_T1_2mm'
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep'

    VxmRegistraion_node.inputs.subject_id = subject_id
    VxmRegistraion_node.inputs.norm = subjects_dir / subject_id / 'mri' / 'norm.mgz'
    VxmRegistraion_node.inputs.model_file = Path(
        __file__).parent.parent / 'model' / 'voxelmorph' / atlas_type / 'model.h5'
    VxmRegistraion_node.inputs.atlas_type = atlas_type

    VxmRegistraion_node.inputs.vxm_warp = derivative_deepprep_path / 'tmp' / 'warp.nii.gz'
    VxmRegistraion_node.inputs.vxm_warped = derivative_deepprep_path / 'tmp' / 'warped.nii.gz'
    VxmRegistraion_node.inputs.trf = derivative_deepprep_path / 'tmp' / f'{subject_id}_affine.mat'
    VxmRegistraion_node.inputs.warp = derivative_deepprep_path / 'tmp' / f'{subject_id}_warp.nii.gz'
    VxmRegistraion_node.inputs.warped = derivative_deepprep_path / 'tmp' / f'{subject_id}_warped.nii.gz'
    VxmRegistraion_node.inputs.npz = derivative_deepprep_path / 'tmp' / 'vxminput.npz'

    # Rest Gauss
    RestGauss_node = Node(RestGauss(), f'RestGauss_node')
    RestGauss_node.inputs.subject_id = subject_id


    # Rest Bandpass
    RestBandpass_node = Node(RestBandpass(), name='RestBandpass_node')
    RestBandpass_node.inputs.subject_id = subject_id
    RestBandpass_node.inputs.data_path = data_path
    # RestBandpass_node.inputs.preprocess_dir = preprocess_dir
    RestBandpass_node.inputs.subj = subj
    RestBandpass_node.inputs.task = task

    # Rest Regression
    RestRegression_node = Node(RestRegression(), f'RestRegression_node')
    RestRegression_node.inputs.subject_id = subject_id
    # RestRegression_node.inputs.preprocess_dir = preprocess_dir
    RestRegression_node.inputs.bold_dir = preprocess_dir / subject_id / 'bold'
    RestRegression_node.inputs.data_path = data_path
    RestRegression_node.inputs.task = task
    RestRegression_node.inputs.subj = subj
    RestRegression_node.inputs.fcmri_dir = preprocess_dir / subject_id / 'fcmri'

    # voxel morph registration
    VxmRegNormMNI152_node = Node(VxmRegNormMNI152(), name='VxmRegNormMNI152_node')
    VxmRegNormMNI152_node.inputs.subject_id = subject_id
    VxmRegNormMNI152_node.inputs.workdir = workdir
    VxmRegNormMNI152_node.inputs.subj = subj
    VxmRegNormMNI152_node.inputs.task = task
    # VxmRegNormMNI152_node.inputs.preprocess_dir = preprocess_dir
    VxmRegNormMNI152_node.inputs.data_path = data_path
    VxmRegNormMNI152_node.inputs.deepprep_subj_path = derivative_deepprep_path / subject_id
    VxmRegNormMNI152_node.inputs.preprocess_method = preprocess_method
    VxmRegNormMNI152_node.inputs.norm = derivative_deepprep_path / 'Recon' / f'sub-{subj}' / 'mri' / 'norm.mgz'

    # Smooth
    Smooth_node = Node(Smooth(), name='Smooth_node')
    Smooth_node.inputs.subject_id = subject_id
    Smooth_node.inputs.subj = subj
    Smooth_node.inputs.task = task
    Smooth_node.inputs.workdir = workdir
    # Smooth_node.inputs.preprocess_dir = preprocess_dir
    Smooth_node.inputs.data_path = data_path
    Smooth_node.inputs.deepprep_subj_path = derivative_deepprep_path / subject_id
    Smooth_node.inputs.preprocess_method = preprocess_method


    # create workflow

    single_bold_rest_wf.connect([
        (RestGauss_node, RestBandpass_node, [("preprocess_dir", "preprocess_dir"),
                                            ]),
        (RestBandpass_node, RestRegression_node, [("preprocess_dir", "preprocess_dir"),
                                                    ]),
        (RestRegression_node, VxmRegNormMNI152_node, [("preprocess_dir", "preprocess_dir"),
                                                    ]),
        (VxmRegNormMNI152_node, Smooth_node, [("deepprep_subj_path", "deepprep_subj_path"),
                                                ]),
    ])

    return single_bold_rest_wf

def pipeline():

    subject_id = 'sub-MSC01'
    subj = 'MSC01'
    task = 'motor'
    preprocess_method = 'task'

    python_interpret = Path('/home/youjia/anaconda3/envs/3.8/bin/python3')
    freesurfer_home = Path('/usr/local/freesurfer')


    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC') # BIDS path
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'
    deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
    subjects_dir = derivative_deepprep_path / "Recon"  # ！！ structure 预处理结果所在的SUBJECTS_DIR路径

    workdir = deepprep_subj_path / 'tmp' / f'task-{task}'
    preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'


    os.environ['SUBJECTS_DIR'] = str(subjects_dir)


    wf_common = init_single_bold_common_wf(subject_id, subj, task, preprocess_method, data_path, derivative_deepprep_path,
                                  subjects_dir, workdir, preprocess_dir)
    wf_rest = init_single_bold_rest_wf(subject_id, subj, task, preprocess_method, data_path,
                                           derivative_deepprep_path,
                                           subjects_dir, workdir, preprocess_dir)

    wf_common.base_dir = preprocess_dir / subject_id  # ！！ 缓存的存储位置，运行过程的tmp文件，可以删除
    wf_rest.base_dir = preprocess_dir / subject_id  # ！！ 缓存的存储位置，运行过程的tmp文件，可以删除

    wf_rest.RestGauss_node.mc = init_single_bold_common_wf.MotionCorrection_node.outputs.skip_faln_mc
    # wf_common.write_graph(graph2use='flat', simple_form=False)

    # wf_common.run()


if __name__ == '__main__':
    import os


    def set_envrion(threads: int = 1):
        # FreeSurfer recon-all env
        freesurfer_home = '/usr/local/freesurfer720'
        os.environ['FREESURFER_HOME'] = f'{freesurfer_home}'
        os.environ['FREESURFER'] = f'{freesurfer_home}'
        os.environ['SUBJECTS_DIR'] = f'{freesurfer_home}/subjects'
        os.environ['PATH'] = f'{freesurfer_home}/bin:{freesurfer_home}/mni/bin:{freesurfer_home}/tktools:' + \
                             f'{freesurfer_home}/fsfast/bin:' + os.environ['PATH']
        os.environ['MINC_BIN_DIR'] = f'{freesurfer_home}/mni/bin'
        os.environ['MINC_LIB_DIR'] = f'{freesurfer_home}/mni/lib'
        os.environ['PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
        os.environ['MNI_PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
        # FreeSurfer fsfast env
        os.environ['FSF_OUTPUT_FORMAT'] = 'nii.gz'
        os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
        # FSL
        os.environ['PATH'] = '/usr/local/fsl/bin:' + os.environ['PATH']
        # ANFI
        os.environ['PATH'] = '/home/anning/abin:' + os.environ['PATH']
        # ANTs
        os.environ['PATH'] = '/usr/local/ANTs/bin:' + os.environ['PATH']
        # Convert3D
        os.environ['PATH'] = '/usr/local/c3d-1.1.0-Linux-x86_64/bin:' + os.environ['PATH']


    set_envrion()
    pipeline()
