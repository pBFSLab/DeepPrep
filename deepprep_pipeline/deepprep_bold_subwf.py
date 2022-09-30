from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import Node, Workflow, config, logging
from bold_interface import VxmRegistraion, BoldSkipReorient, Stc, MkTemplate, \
    MotionCorrection, Register, MkBrainmask, VxmRegNormMNI152, RestGauss, \
    RestBandpass, RestRegression, Smooth
from run import set_envrion

import threading


# Part1 GPU

def init_structure_part1_wf(subject_ids: list,
                            data_path: Path,
                            subjects_dir: Path):
    structure_part1_wf = Workflow(name=f'structure_part1__wf')

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
    atlas_type = 'MNI152_T1_2mm'

    VxmRegistraion_node.iterables = [('subject_id', subject_ids)]
    VxmRegistraion_node.synchronize = True
    VxmRegistraion_node.inputs.data_path = data_path

    VxmRegistraion_node.inputs.subjects_dir = subjects_dir

    VxmRegistraion_node.inputs.model_file = Path(
        __file__).parent.parent / 'deepprep_pipeline' / 'model' / 'voxelmorph' / atlas_type / 'model.h5'
    VxmRegistraion_node.inputs.atlas_type = atlas_type
    VxmRegistraion_node.inputs.model_path = Path(
        __file__).parent.parent / 'deepprep_pipeline' / 'model' / 'voxelmorph' / atlas_type

    structure_part1_wf.connect([
        (inputnode, VxmRegistraion_node, [("subjects_dir", "subjects_dir"),
                                          ]),
    ])
    return structure_part1_wf


# import os
# set_envrion()
#
# task = 'rest'   # 'motor' or 'rest'
# preprocess_method = 'rest' # 'task' or 'rest'
#
# MNI152_target = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'  # Smooth target
#
# # data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
# # derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'  # bold result output dir path
# #
# # deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
# # subjects_dir = derivative_deepprep_path / "Recon"  # ！！ structure 预处理结果所在的SUBJECTS_DIR路径
# #
# # preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'
# #
# # preprocess_dir.mkdir(parents=True, exist_ok=True)
#
# data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
#
# subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
#
# os.environ['SUBJECTS_DIR'] = str(subjects_dir)
#
# multi_subj_n_procs = 2
# structure_part1_wf = init_structure_part1_wf(subject_ids = ['sub-MSC01', 'sub-MSC02'],
#                                              data_path= data_path,
#                                              subjects_dir = subjects_dir)
# structure_part1_wf.base_dir = '/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test'
# structure_part1_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})
# print()
# exit()

# Part2 CPU
def init_structure_part2_wf(subject_ids: list,
                            task: str,
                            data_path: Path,
                            subjects_dir: Path):
    structure_part2_wf = Workflow(name=f'structure_part2__wf')
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
    BoldSkipReorient_node.inputs.task = task

    # Stc
    Stc_node = Node(Stc(), name='stc_node')

    Stc_node.inputs.task = task
    Stc_node.inputs.data_path = data_path
    # make template
    MkTemplate_node = Node(MkTemplate(), name='MkTemplate_node')

    MkTemplate_node.inputs.task = task
    MkTemplate_node.inputs.data_path = data_path

    # Motion correction
    MotionCorrection_node = Node(MotionCorrection(), name='MotionCorrection_node')

    MotionCorrection_node.inputs.task = task
    MotionCorrection_node.inputs.data_path = data_path

    # bb register
    Register_node = Node(Register(), name='register_node')

    Register_node.inputs.task = task
    Register_node.inputs.data_path = data_path

    # Make brainmask
    Mkbrainmask_node = Node(MkBrainmask(), name='mkbrainmask_node')

    Mkbrainmask_node.inputs.subjects_dir = subjects_dir
    Mkbrainmask_node.inputs.task = task
    Mkbrainmask_node.inputs.data_path = data_path

    structure_part2_wf.connect([
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
    return structure_part2_wf


# import os
# set_envrion()
#
# task = 'rest'   # 'motor' or 'rest'
# # preprocess_method = 'rest' # 'task' or 'rest'
#
# MNI152_target = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'  # Smooth target
#
# # data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
# # derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'  # bold result output dir path
# #
# # deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
# # subjects_dir = derivative_deepprep_path / "Recon"  # ！！ structure 预处理结果所在的SUBJECTS_DIR路径
# #
# # preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'
# #
# # preprocess_dir.mkdir(parents=True, exist_ok=True)
#
# data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
#
# subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
#
# os.environ['SUBJECTS_DIR'] = str(subjects_dir)
#
# multi_subj_n_procs = 2
# structure_part2_wf = init_structure_part2_wf(subject_ids = ['sub-MSC01', 'sub-MSC02'],
#                                              task = task,
#                                              data_path= data_path,
#                                              subjects_dir = subjects_dir)
# structure_part2_wf.base_dir = '/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test'
# structure_part2_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})
# print()
# exit()

# Part3 CPU
def init_structure_part3_wf(subject_ids: list,
                            task: str,
                            data_path: Path,
                            subjects_dir: Path):
    structure_part3_wf = Workflow(name=f'structure_part3__wf')

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
    # Rest Bandpass
    RestBandpass_node = Node(RestBandpass(), name='RestBandpass_node')
    RestBandpass_node.inputs.data_path = data_path
    RestBandpass_node.inputs.task = task

    # Rest Regression
    RestRegression_node = Node(RestRegression(), f'RestRegression_node')

    RestRegression_node.inputs.subjects_dir = subjects_dir
    RestRegression_node.inputs.data_path = data_path
    RestRegression_node.inputs.task = task

    structure_part3_wf.connect([
        (inputnode, RestGauss_node, [("data_path", "data_path")
                                     ]),
        (RestGauss_node, RestBandpass_node, [("subject_id", "subject_id")
                                             ]),
        (RestBandpass_node, RestRegression_node, [("subject_id", "subject_id")
                                                  ])
    ])
    return structure_part3_wf


# import os
# set_envrion()
#
# task = 'rest'   # 'motor' or 'rest'
# # preprocess_method = 'rest' # 'task' or 'rest'
#
# MNI152_target = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'  # Smooth target
#
# # data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
# # derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'  # bold result output dir path
# #
# # deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
# # subjects_dir = derivative_deepprep_path / "Recon"  # ！！ structure 预处理结果所在的SUBJECTS_DIR路径
# #
# # preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'
# #
# # preprocess_dir.mkdir(parents=True, exist_ok=True)
#
# data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
#
# subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
#
# os.environ['SUBJECTS_DIR'] = str(subjects_dir)
#
# multi_subj_n_procs = 2
# structure_part3_wf = init_structure_part3_wf(subject_ids = ['sub-MSC01', 'sub-MSC02'],
#                                              task = task,
#                                              data_path= data_path,
#                                              subjects_dir = subjects_dir)
# structure_part3_wf.base_dir = '/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test'
# structure_part3_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})
# print()
# exit()

# Part4 GPU
def init_structure_part4_wf(subject_ids: list,

                            task: str,
                            data_path: Path,
                            preprocess_method: str):
    structure_part4_wf = Workflow(name=f'structure_part4__wf')

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
    VxmRegNormMNI152_node.iterables = [("subject_id", subject_ids)]
    VxmRegNormMNI152_node.inputs.task = task
    VxmRegNormMNI152_node.inputs.data_path = data_path
    VxmRegNormMNI152_node.inputs.preprocess_method = preprocess_method

    structure_part4_wf.connect([
        (inputnode, VxmRegNormMNI152_node, [("data_path", "data_path")
                                            ])
    ])
    return structure_part4_wf


# import os
# set_envrion()
#
# task = 'rest'   # 'motor' or 'rest'
# preprocess_method = 'rest' # 'task' or 'rest'
#
# MNI152_target = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'  # Smooth target
#
# # data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
# # derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'  # bold result output dir path
# #
# # deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
# # subjects_dir = derivative_deepprep_path / "Recon"  # ！！ structure 预处理结果所在的SUBJECTS_DIR路径
# #
# # preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'
# #
# # preprocess_dir.mkdir(parents=True, exist_ok=True)
#
# data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
#
# subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
#
# os.environ['SUBJECTS_DIR'] = str(subjects_dir)
#
# multi_subj_n_procs = 2
# structure_part4_wf = init_structure_part4_wf(subject_ids = ['sub-MSC01', 'sub-MSC02'],
#                                              task = task,
#                                              data_path= data_path,
#                                              preprocess_method = preprocess_method)
# structure_part4_wf.base_dir = '/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test'
# structure_part4_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})
# print()
# exit()

# Part5 CPU
def init_structure_part5_wf(subject_ids: list,
                            task: str,
                            data_path: Path,
                            preprocess_method: str):
    structure_part5_wf = Workflow(name=f'structure_part5__wf')

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
    Smooth_node.inputs.MNI152_T1_2mm_brain_mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'

    structure_part5_wf.connect([
        (inputnode, Smooth_node, [("data_path", "data_path")
                                  ])
    ])
    return structure_part5_wf


# import os
# set_envrion()
#
# task = 'rest'   # 'motor' or 'rest'
# preprocess_method = 'rest' # 'task' or 'rest'
#
# MNI152_target = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'  # Smooth target
#
# # data_path = Path(f'/mnt/ngshare/DeepPrep/MSC')  # BIDS path
# # derivative_deepprep_path = data_path / 'derivatives' / 'deepprep_wftest'  # bold result output dir path
# #
# # deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
# # subjects_dir = derivative_deepprep_path / "Recon"  # ！！ structure 预处理结果所在的SUBJECTS_DIR路径
# #
# # preprocess_dir = deepprep_subj_path / 'tmp' / f'task-{task}'
# #
# # preprocess_dir.mkdir(parents=True, exist_ok=True)
#
# data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
#
# subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
#
# os.environ['SUBJECTS_DIR'] = str(subjects_dir)
#
# multi_subj_n_procs = 2
# structure_part5_wf = init_structure_part5_wf(subject_ids = ['sub-MSC01', 'sub-MSC02'],
#                                              task = task,
#                                              data_path= data_path,
#                                              preprocess_method = preprocess_method)
# structure_part5_wf.base_dir = '/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test'
# structure_part5_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})
# print()
# exit()


def pipeline():

    subject_ids = ['sub-MSC01', 'sub-MSC02']
    task = 'motor'  # 'motor' or 'rest'
    preprocess_method = 'task'  # 'task' or 'rest'
    data_path = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test')  # BIDS path
    subjects_dir = Path('/mnt/ngshare/DeepPrep/MSC/derivatives/MSC_bold_test/derivatives/deepprep_bold_test/Recon')
    multi_subj_n_procs = 2
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    # 设置log目录位置
    log_dir = subjects_dir / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)
    config.update_config({'logging': {'log_directory': log_dir,
                                      'log_to_file': True}})
    logging.update_logging(config)

    structure_part1_wf = init_structure_part1_wf(subject_ids=subject_ids,
                                                 data_path=data_path,
                                                 subjects_dir=subjects_dir)
    structure_part1_wf.base_dir = subjects_dir
    structure_part1_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})

    structure_part2_wf = init_structure_part2_wf(subject_ids=subject_ids,
                                                 task=task,
                                                 data_path=data_path,
                                                 subjects_dir=subjects_dir)
    structure_part2_wf.base_dir = subjects_dir
    structure_part2_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})
    if task == 'rest':
        structure_part3_wf = init_structure_part3_wf(subject_ids=subject_ids,
                                                     task=task,
                                                     data_path=data_path,
                                                     subjects_dir=subjects_dir)
        structure_part3_wf.base_dir = subjects_dir
        structure_part3_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})

    structure_part4_wf = init_structure_part4_wf(subject_ids=subject_ids,
                                                 task=task,
                                                 data_path=data_path,
                                                 preprocess_method=preprocess_method)
    structure_part4_wf.base_dir = subjects_dir
    structure_part4_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})

    structure_part5_wf = init_structure_part5_wf(subject_ids=subject_ids,
                                                 task=task,
                                                 data_path=data_path,
                                                 preprocess_method=preprocess_method)
    structure_part5_wf.base_dir = subjects_dir
    structure_part5_wf.run('MultiProc', plugin_args={'n_procs': multi_subj_n_procs})


if __name__ == '__main__':
    import os

    set_envrion()

    pipeline()
