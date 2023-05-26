from pathlib import Path
from nipype import Node, Workflow
from bold import VxmRegistraion, BoldSkipReorient, Stc, MkTemplate, \
    MotionCorrection, Register, MkBrainmask, VxmRegNormMNI152, RestGauss, \
    RestBandpass, RestRegression, Smooth


def init_single_bold_common_wf(subject_id: str, subj: str, task: str,
                               data_path: Path, derivative_deepprep_path: Path,
                               subjects_dir: Path, preprocess_dir: Path):
    single_bold_common_wf = Workflow(name=f'single_bold_common_{subject_id.replace("-", "_")}_wf')

    ########################################### BOLD - Common ###########################################
    # Voxelmorph registration
    VxmRegistraion_node = Node(VxmRegistraion(), name='VxmRegistraion_node')
    atlas_type = 'MNI152_T1_2mm'

    deepprep_subj_path = derivative_deepprep_path / subject_id

    VxmRegistraion_node.inputs.subject_id = subject_id
    VxmRegistraion_node.inputs.data_path = data_path
    VxmRegistraion_node.inputs.deepprep_subj_path = deepprep_subj_path
    VxmRegistraion_node.inputs.preprocess_dir = preprocess_dir
    VxmRegistraion_node.inputs.norm = subjects_dir / subject_id / 'mri' / 'norm.mgz'  # TODO 用workflow_connect
    VxmRegistraion_node.inputs.model_file = Path(
        __file__).parent.parent / 'src' / 'model' / 'voxelmorph' / atlas_type / 'model.h5'
    VxmRegistraion_node.inputs.atlas_type = atlas_type
    VxmRegistraion_node.inputs.model_path = Path(__file__).parent.parent / 'src' / 'model' / 'voxelmorph' / atlas_type


    VxmRegistraion_node.inputs.vxm_warp = deepprep_subj_path / 'tmp' / 'warp.nii.gz'
    VxmRegistraion_node.inputs.vxm_warped = deepprep_subj_path / 'tmp' / 'warped.nii.gz'
    VxmRegistraion_node.inputs.trf = deepprep_subj_path / 'tmp' / f'{subject_id}_affine.mat'
    VxmRegistraion_node.inputs.warp = deepprep_subj_path / 'tmp' / f'{subject_id}_warp.nii.gz'
    VxmRegistraion_node.inputs.warped = deepprep_subj_path / 'tmp' / f'{subject_id}_warped.nii.gz'
    VxmRegistraion_node.inputs.npz = deepprep_subj_path / 'tmp' / 'vxminput.npz'

    # bold skip reorient
    BoldSkipReorient_node = Node(BoldSkipReorient(), name='BoldSkipReorient_node')
    BoldSkipReorient_node.inputs.subject_id = subject_id
    BoldSkipReorient_node.inputs.subj = subj
    BoldSkipReorient_node.inputs.data_path = data_path
    BoldSkipReorient_node.inputs.deepprep_subj_path = deepprep_subj_path
    BoldSkipReorient_node.inputs.task = task
    # BoldSkipReorient_node.inputs.preprocess_dir = preprocess_dir

    # Stc
    Stc_node = Node(Stc(), name='stc_node')
    Stc_node.inputs.subject_id = subject_id

    # make template
    MkTemplate_node = Node(MkTemplate(), name='MkTemplate_node')
    MkTemplate_node.inputs.subject_id = subject_id

    # Motion correction
    MotionCorrection_node = Node(MotionCorrection(), name='MotionCorrection_node')
    MotionCorrection_node.inputs.subject_id = subject_id

    # bb register
    Register_node = Node(Register(), name='register_node')
    Register_node.inputs.subject_id = subject_id

    # Make brainmask
    Mkbrainmask_node = Node(MkBrainmask(), name='mkbrainmask_node')
    Mkbrainmask_node.inputs.subject_id = subject_id
    Mkbrainmask_node.inputs.subjects_dir = subjects_dir

    # create workflow

    single_bold_common_wf.connect([
        (VxmRegistraion_node, BoldSkipReorient_node, [("preprocess_dir", "preprocess_dir"),
                                               ]),
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


def init_single_bold_rest_wf(subject_id: str, subj: str, task: str, mni152_target: str, preprocess_method: str,
                             data_path: Path, derivative_deepprep_path: Path,
                             subjects_dir: Path, preprocess_dir: Path):
    single_bold_rest_wf = Workflow(name=f'single_bold_rest_{subject_id.replace("-", "_")}_wf')

    # # Voxelmorph registration
    # VxmRegistraion_node = Node(VxmRegistraion(), name='VxmRegistraion_node')
    # atlas_type = 'MNI152_T1_2mm'
    #
    # VxmRegistraion_node.inputs.subject_id = subject_id
    # VxmRegistraion_node.inputs.data_path = data_path
    # VxmRegistraion_node.inputs.deepprep_subj_path = derivative_deepprep_path / subject_id
    # VxmRegistraion_node.inputs.norm = subjects_dir / subject_id / 'mri' / 'norm.mgz'
    # VxmRegistraion_node.inputs.model_file = Path(
    #     __file__).parent.parent / 'src' / 'model' / 'voxelmorph' / atlas_type / 'model.h5'
    # VxmRegistraion_node.inputs.atlas_type = atlas_type
    #
    # VxmRegistraion_node.inputs.vxm_warp = derivative_deepprep_path / 'tmp' / 'warp.nii.gz'
    # VxmRegistraion_node.inputs.vxm_warped = derivative_deepprep_path / 'tmp' / 'warped.nii.gz'
    # VxmRegistraion_node.inputs.trf = derivative_deepprep_path / 'tmp' / f'{subject_id}_affine.mat'
    # VxmRegistraion_node.inputs.warp = derivative_deepprep_path / 'tmp' / f'{subject_id}_warp.nii.gz'
    # VxmRegistraion_node.inputs.warped = derivative_deepprep_path / 'tmp' / f'{subject_id}_warped.nii.gz'
    # VxmRegistraion_node.inputs.npz = derivative_deepprep_path / 'tmp' / 'vxminput.npz'

    # Rest Gauss
    RestGauss_node = Node(RestGauss(), f'RestGauss_node')
    RestGauss_node.inputs.subject_id = subject_id

    # Rest Bandpass
    RestBandpass_node = Node(RestBandpass(), name='RestBandpass_node')
    RestBandpass_node.inputs.subject_id = subject_id
    RestBandpass_node.inputs.data_path = data_path
    RestBandpass_node.inputs.subj = subj
    RestBandpass_node.inputs.task = task

    # Rest Regression
    RestRegression_node = Node(RestRegression(), f'RestRegression_node')
    RestRegression_node.inputs.subject_id = subject_id
    RestRegression_node.inputs.subjects_dir = subjects_dir
    RestRegression_node.inputs.bold_dir = preprocess_dir / subject_id / 'bold'
    RestRegression_node.inputs.data_path = data_path
    RestRegression_node.inputs.deepprep_subj_path = derivative_deepprep_path / subject_id
    RestRegression_node.inputs.task = task
    RestRegression_node.inputs.subj = subj
    RestRegression_node.inputs.fcmri_dir = preprocess_dir / subject_id / 'fcmri'

    # create workflow

    single_bold_rest_wf.connect([
        # (VxmRegistraion_node, RestGauss_node, [("preprocess_dir", "preprocess_dir"),
        #                                        ]),
        (RestGauss_node, RestBandpass_node, [("preprocess_dir", "preprocess_dir"),
                                             ]),
        (RestBandpass_node, RestRegression_node, [("preprocess_dir", "preprocess_dir"),
                                                  ]),
        # (RestRegression_node, VxmRegNormMNI152_node, [("preprocess_dir", "preprocess_dir"),
        #                                               ]),
        # (VxmRegNormMNI152_node, Smooth_node, [("deepprep_subj_path", "deepprep_subj_path"),
        #                                       ("preprocess_dir", "preprocess_dir"),
        #                                       ]),
    ])

    return single_bold_rest_wf

def init_single_bold_projection_smooth_wf(subject_id: str, subj: str, task: str, mni152_target: str, preprocess_method: str,
                             data_path: Path, derivative_deepprep_path: Path,
                             subjects_dir: Path, preprocess_dir: Path):
    single_bold_projection_smooth_wf = Workflow(name=f'single_bold_projection_smooth_{subject_id.replace("-", "_")}_wf')

    # voxel morph registration
    VxmRegNormMNI152_node = Node(VxmRegNormMNI152(), name='VxmRegNormMNI152_node')
    VxmRegNormMNI152_node.inputs.subject_id = subject_id
    VxmRegNormMNI152_node.inputs.subj = subj
    VxmRegNormMNI152_node.inputs.task = task
    VxmRegNormMNI152_node.inputs.data_path = data_path
    VxmRegNormMNI152_node.inputs.deepprep_subj_path = derivative_deepprep_path / subject_id
    VxmRegNormMNI152_node.inputs.preprocess_method = preprocess_method
    VxmRegNormMNI152_node.inputs.norm = derivative_deepprep_path / 'Recon' / f'sub-{subj}' / 'mri' / 'norm.mgz'

    # Smooth
    Smooth_node = Node(Smooth(), name='Smooth_node')
    Smooth_node.inputs.subject_id = subject_id
    Smooth_node.inputs.subj = subj
    Smooth_node.inputs.task = task
    Smooth_node.inputs.data_path = data_path
    Smooth_node.inputs.deepprep_subj_path = derivative_deepprep_path / subject_id
    Smooth_node.inputs.preprocess_method = preprocess_method

    Smooth_node.inputs.MNI152_T1_2mm_brain_mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'

    single_bold_projection_smooth_wf.connect([
        (VxmRegNormMNI152_node, Smooth_node, [("deepprep_subj_path", "deepprep_subj_path"),
                                              ("preprocess_dir", "preprocess_dir"),]),
    ])

    return single_bold_projection_smooth_wf
def pipeline():
    subject_id = 'sub-0025427'
    subj = '0025427'
    task = 'rest'   # 'motor' or 'rest'
    preprocess_method = 'rest' # 'task' or 'rest'

    MNI152_target = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'  # Smooth target

    # data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
    # derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'  # bold result output dir path
    #
    # deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
    # subjects_dir = derivative_deepprep_path / "Recon"  # ！！ structure 预处理结果所在的SUBJECTS_DIR路径
    #
    # preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'
    #
    # preprocess_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(f'/mnt/ngshare/DeepPrep/HNU_1/derivatives/HNU_1_bold_test')  # BIDS path
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_bold_test'  # bold result output dir path
    deepprep_subj_path = derivative_deepprep_path / subject_id
    preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'
    subjects_dir = Path('/mnt/ngshare/DeepPrep/HNU_1/derivatives/HNU_1_bold_test/derivatives/deepprep_bold_test/Recon')

    preprocess_dir.mkdir(parents=True, exist_ok=True)

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    wf_common = init_single_bold_common_wf(subject_id, subj, task, data_path, derivative_deepprep_path,
                                           subjects_dir, preprocess_dir)
    wf_rest = init_single_bold_rest_wf(subject_id, subj, task, MNI152_target, preprocess_method, data_path,
                                       derivative_deepprep_path, subjects_dir, preprocess_dir)
    wf_projection_smooth = init_single_bold_projection_smooth_wf(subject_id, subj, task, MNI152_target,
                                                                 preprocess_method, data_path, derivative_deepprep_path,
                                                                 subjects_dir, preprocess_dir)
    # wf_common.base_dir = preprocess_dir / subject_id  # ！！ 缓存的存储位置，运行过程的tmp文件，可以删除
    # wf_rest.base_dir = preprocess_dir / subject_id  # ！！ 缓存的存储位置，运行过程的tmp文件，可以删除

    wf_full = Workflow(name=f'single_bold_full_{subject_id.replace("-", "_")}_wf')
    wf_full.base_dir = preprocess_dir
    if preprocess_method == 'task':
        wf_full.connect([
            (wf_common, wf_projection_smooth,
             [("MotionCorrection_node.preprocess_dir", "VxmRegNormMNI152_node.preprocess_dir")])
        ])
    elif preprocess_method == 'rest':
        wf_full.connect([
            (wf_common, wf_rest,
             [("MotionCorrection_node.preprocess_dir", "RestGauss_node.preprocess_dir")]),
            (wf_rest, wf_projection_smooth,
             [("RestRegression_node.preprocess_dir", "VxmRegNormMNI152_node.preprocess_dir")])
        ])
        # wf_full.connect([
        #     (wf_rest, wf_projection_smooth,
        #      [("RestRegression_node.preprocess_dir", "VxmRegNormMNI152_node.preprocess_dir")])
        # ])
    wf_full.write_graph(graph2use='flat', simple_form=False)

    wf_full.run()


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
