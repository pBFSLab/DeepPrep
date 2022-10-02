from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import Node, Workflow, config, logging
from bold_interface import VxmRegistraion, BoldSkipReorient, Stc, MkTemplate, \
    MotionCorrection, Register, MkBrainmask, VxmRegNormMNI152, RestGauss, \
    RestBandpass, RestRegression, Smooth
from run import set_envrion


def init_bold_part1_wf(subject_ids: list,
                       data_path: Path,
                       vxm_model_path: Path,
                       atlas_type: str,
                       subjects_dir: Path,
                       derivative_deepprep_path: Path):
    bold_part1_wf = Workflow(name=f'bold_part1__wf')

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "subjects_dir",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.subjects_dir = subjects_dir

    # Voxelmorph registration
    VxmRegistraion_node = Node(VxmRegistraion(), name='VxmRegistraion_node')

    VxmRegistraion_node.iterables = [('subject_id', subject_ids)]
    VxmRegistraion_node.synchronize = True
    VxmRegistraion_node.inputs.data_path = data_path
    VxmRegistraion_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    VxmRegistraion_node.inputs.subjects_dir = subjects_dir

    VxmRegistraion_node.inputs.model_file = vxm_model_path / atlas_type / 'model.h5'
    VxmRegistraion_node.inputs.vxm_model_path = vxm_model_path
    VxmRegistraion_node.inputs.atlas_type = atlas_type

    bold_part1_wf.connect([
        (inputnode, VxmRegistraion_node, [("subjects_dir", "subjects_dir"),
                                          ]),
    ])
    return bold_part1_wf


# Part2 CPU
def init_bold_part2_wf(subject_ids: list,
                       task: str,
                       data_path: Path,
                       subjects_dir: Path,
                       derivative_deepprep_path: Path):
    bold_part2_wf = Workflow(name=f'bold_part2__wf')
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "data_path",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.data_path = data_path

    # bold skip reorient
    BoldSkipReorient_node = Node(BoldSkipReorient(), name='BoldSkipReorient_node')
    BoldSkipReorient_node.iterables = [("subject_id", subject_ids)]

    BoldSkipReorient_node.inputs.data_path = data_path
    BoldSkipReorient_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    BoldSkipReorient_node.inputs.task = task

    # Stc
    Stc_node = Node(Stc(), name='stc_node')

    Stc_node.inputs.task = task
    Stc_node.inputs.data_path = data_path
    Stc_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    # make template
    MkTemplate_node = Node(MkTemplate(), name='MkTemplate_node')

    MkTemplate_node.inputs.task = task
    MkTemplate_node.inputs.data_path = data_path
    MkTemplate_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    # Motion correction
    MotionCorrection_node = Node(MotionCorrection(), name='MotionCorrection_node')

    MotionCorrection_node.inputs.task = task
    MotionCorrection_node.inputs.data_path = data_path
    MotionCorrection_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    # bb register
    Register_node = Node(Register(), name='register_node')

    Register_node.inputs.task = task
    Register_node.inputs.data_path = data_path
    Register_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    # Make brainmask
    Mkbrainmask_node = Node(MkBrainmask(), name='mkbrainmask_node')

    Mkbrainmask_node.inputs.subjects_dir = subjects_dir
    Mkbrainmask_node.inputs.task = task
    Mkbrainmask_node.inputs.data_path = data_path
    Mkbrainmask_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    bold_part2_wf.connect([
        (inputnode, BoldSkipReorient_node, [("data_path", "data_path")
                                            ]),
        (BoldSkipReorient_node, Stc_node, [("subject_id", "subject_id")
                                           ]),
        (Stc_node, MkTemplate_node, [("subject_id", "subject_id")
                                     ]),
        (MkTemplate_node, MotionCorrection_node, [("subject_id", "subject_id")
                                                  ]),
        (MotionCorrection_node, Register_node, [("subject_id", "subject_id")
                                                ]),
        (Register_node, Mkbrainmask_node, [("subject_id", "subject_id")
                                           ])
    ])
    return bold_part2_wf


# Part3 CPU
def init_bold_part3_wf(subject_ids: list,
                       task: str,
                       data_path: Path,
                       subjects_dir: Path,
                       derivative_deepprep_path: Path):
    bold_part3_wf = Workflow(name=f'bold_part3__wf')

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "data_path",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.data_path = data_path
    # Rest Gauss
    RestGauss_node = Node(RestGauss(), f'RestGauss_node')
    RestGauss_node.iterables = [("subject_id", subject_ids)]
    RestGauss_node.inputs.subjects_dir = subjects_dir
    RestGauss_node.inputs.data_path = data_path
    RestGauss_node.inputs.task = task
    RestGauss_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    # Rest Bandpass
    RestBandpass_node = Node(RestBandpass(), name='RestBandpass_node')
    RestBandpass_node.inputs.data_path = data_path
    RestBandpass_node.inputs.task = task
    RestBandpass_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    # Rest Regression
    RestRegression_node = Node(RestRegression(), f'RestRegression_node')

    RestRegression_node.inputs.subjects_dir = subjects_dir
    RestRegression_node.inputs.data_path = data_path
    RestRegression_node.inputs.task = task
    RestRegression_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    bold_part3_wf.connect([
        (inputnode, RestGauss_node, [("data_path", "data_path")
                                     ]),
        (RestGauss_node, RestBandpass_node, [("subject_id", "subject_id")
                                             ]),
        (RestBandpass_node, RestRegression_node, [("subject_id", "subject_id")
                                                  ])
    ])
    return bold_part3_wf


# Part4 GPU
def init_bold_part5_wf(subject_ids: list,
                       task: str,
                       data_path: Path,
                       preprocess_method: str,
                       mni152_brain_mask: Path,
                       derivative_deepprep_path: Path):
    bold_part5_wf = Workflow(name=f'bold_part5__wf')

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "data_path",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.data_path = data_path

    # Smooth
    Smooth_node = Node(Smooth(), name='Smooth_node')
    Smooth_node.iterables = [("subject_id", subject_ids)]
    Smooth_node.inputs.task = task
    Smooth_node.inputs.data_path = data_path
    Smooth_node.inputs.preprocess_method = preprocess_method
    Smooth_node.inputs.MNI152_T1_2mm_brain_mask = mni152_brain_mask
    Smooth_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    bold_part5_wf.connect([
        (inputnode, Smooth_node, [("data_path", "data_path")
                                  ])
    ])
    return bold_part5_wf


def init_bold_part4_wf(subject_ids: list,
                       task: str,
                       data_path: Path,
                       subjects_dir: Path,
                       preprocess_method: str,
                       vxm_model_path: Path,
                       atlas_type: str,
                       resource_dir: Path,
                       derivative_deepprep_path: Path):
    bold_part4_wf = Workflow(name=f'bold_part4__wf')

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "data_path",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.data_path = data_path

    # voxel morph registration
    VxmRegNormMNI152_node = Node(VxmRegNormMNI152(), name='VxmRegNormMNI152_node')
    VxmRegNormMNI152_node.inputs.subjects_dir = subjects_dir
    VxmRegNormMNI152_node.iterables = [("subject_id", subject_ids)]
    VxmRegNormMNI152_node.inputs.atlas_type = atlas_type
    VxmRegNormMNI152_node.inputs.task = task
    VxmRegNormMNI152_node.inputs.data_path = data_path
    VxmRegNormMNI152_node.inputs.preprocess_method = preprocess_method
    VxmRegNormMNI152_node.inputs.vxm_model_path = vxm_model_path
    VxmRegNormMNI152_node.inputs.resource_dir = resource_dir
    VxmRegNormMNI152_node.inputs.derivative_deepprep_path = derivative_deepprep_path

    bold_part4_wf.connect([
        (inputnode, VxmRegNormMNI152_node, [("data_path", "data_path")
                                            ])
    ])
    return bold_part4_wf


def pipeline():
    pwd = Path.cwd()

    vxm_model_path = pwd / 'model' / 'voxelmorph'
    resource_dir = pwd / 'resource'
    atlas_type = 'MNI152_T1_2mm'

    # subject_ids = ['sub-MSC01', 'sub-MSC02']
    task = 'rest'  # 'motor' or 'rest'
    preprocess_method = 'rest'  # 'task' or 'rest'
    data_path = Path(f'/mnt/ngshare/DeepPrep/HNU_1')  # BIDS path
    subjects_dir = Path('/mnt/ngshare/DeepPrep/HNU_1/derivatives/DeepPrep/Recon')
    derivative_deepprep_path = Path('/mnt/ngshare/DeepPrep_flowtest/HNU_1_bold_test')
    workflow_cache_dir = Path("/mnt/ngshare/DeepPrep_flowtest/HNU_1_Workflow")  # workflow tmp cache dir
    mni152_brain_mask = Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')

    subject_ids = os.listdir(data_path)
    subject_ids.remove('derivatives')
    subject_ids.remove('dataset_description.json')

    # multi_subj_n_procs = 2
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    # 设置log目录位置
    log_dir = subjects_dir / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)
    config.update_config({'logging': {'log_directory': log_dir,
                                      'log_to_file': True}})
    logging.update_logging(config)

    bold_part1_wf = init_bold_part1_wf(subject_ids=subject_ids,
                                       data_path=data_path,
                                       vxm_model_path=vxm_model_path,
                                       atlas_type=atlas_type,
                                       subjects_dir=subjects_dir,
                                       derivative_deepprep_path=derivative_deepprep_path)
    bold_part1_wf.base_dir = workflow_cache_dir
    bold_part1_wf.run('MultiProc', plugin_args={'n_procs': 8})

    bold_part2_wf = init_bold_part2_wf(subject_ids=subject_ids,
                                       task=task,
                                       data_path=data_path,
                                       subjects_dir=subjects_dir,
                                       derivative_deepprep_path=derivative_deepprep_path)
    bold_part2_wf.base_dir = workflow_cache_dir
    bold_part2_wf.run('MultiProc', plugin_args={'n_procs': 8})
    exit()
    if task == 'rest':
        bold_part3_wf = init_bold_part3_wf(subject_ids=subject_ids,
                                           task=task,
                                           data_path=data_path,
                                           subjects_dir=subjects_dir,
                                           derivative_deepprep_path=derivative_deepprep_path)
        bold_part3_wf.base_dir = workflow_cache_dir
        bold_part3_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})

    bold_part4_wf = init_bold_part4_wf(subject_ids=subject_ids,
                                       task=task,
                                       data_path=data_path,
                                       subjects_dir=subjects_dir,
                                       preprocess_method=preprocess_method,
                                       vxm_model_path=vxm_model_path,
                                       atlas_type=atlas_type,
                                       resource_dir=resource_dir,
                                       derivative_deepprep_path=derivative_deepprep_path)
    bold_part4_wf.base_dir = workflow_cache_dir
    bold_part4_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})

    bold_part5_wf = init_bold_part5_wf(subject_ids=subject_ids,
                                       task=task,
                                       data_path=data_path,
                                       preprocess_method=preprocess_method,
                                       mni152_brain_mask=mni152_brain_mask,
                                       derivative_deepprep_path=derivative_deepprep_path)
    bold_part5_wf.base_dir = workflow_cache_dir
    bold_part5_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})


if __name__ == '__main__':
    import os

    set_envrion()

    pipeline()
