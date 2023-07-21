# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : NingAnMe <ninganme0317@gmail.com>
# @Author : Cong Lin       @Email : lincong <lincong8722@gmail.com>
# @Author : Youjia Zhang   @Email : youjia <ireneyou33@gmail.com>
# @Author : Zhenyu Sun     @Email : Kid-sunzhenyu <sun25939789@gmail.com>

from deepprep.interface.bold_node import *
from deepprep.interface.vxm_node import *
from deepprep.interface.node_source import Source

from nipype import Node

"""环境变量f
subjects_dir = Path(settings.SUBJECTS_DIR)
bold_preprocess_dir = Path(settings.BOLD_PREPROCESS_DIR)
workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
fastsurfer_home = Path(settings.FASTSURFER_HOME)
freesurfer_home = Path(settings.FREESURFER_HOME)
fastcsr_home = Path(settings.FASTCSR_HOME)
featreg_home = Path(settings.FEATREG_HOME)
python_interpret = sys.executable
"""


def create_BoldSkipReorient_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    """Skip the first n frames

        Inputs
        ------
        subject_id
            Subject id
        data_path
            BIDS dir
        derivative_deepprep_path
            Bold preprocessing dir
        task
            ``motor`` or ``rest``
        preprocess_method
            ``task`` or ``rest``
        atlas_type
            Atlas tyle (eg: ``MNI152_T1_2mm``)
        nskip_frame
            Skip from the first n frames
        multiprocess
            Using for pool threads set


        Outputs
        -------
        skip_reorient
            Remove the files of the first n frames

        See also
        --------
        * :py:func:`deepprep.interface.bold_node.BoldSkipReorient`

    """
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    derivative_deepprep_path = Path(settings.BOLD_PREPROCESS_DIR)
    data_path = Path(settings.BIDS_DIR)

    BoldSkipReorient_node = Node(BoldSkipReorient(), name=f'{subject_id}_fMRI_BoldSkipReorient_node')
    BoldSkipReorient_node.inputs.subject_id = subject_id
    BoldSkipReorient_node.inputs.data_path = data_path
    BoldSkipReorient_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    BoldSkipReorient_node.inputs.task = task
    BoldSkipReorient_node.inputs.atlas_type = atlas_type
    BoldSkipReorient_node.inputs.preprocess_method = preprocess_method

    BoldSkipReorient_node.base_dir = workflow_cached_dir / subject_id
    CPU_NUM = settings.FMRI.BoldSkipReorient.CPU_NUM
    RAM_MB = settings.FMRI.BoldSkipReorient.RAM_MB
    GPU_MB = settings.FMRI.BoldSkipReorient.GPU_MB
    IO_WRITE_MB = settings.FMRI.BoldSkipReorient.IO_WRITE_MB
    IO_READ_MB = settings.FMRI.BoldSkipReorient.IO_READ_MB
    BoldSkipReorient_node.inputs.nskip_frame = str(settings.FMRI.BoldSkipReorient.NSKIP_FRAME)
    BoldSkipReorient_node.inputs.multiprocess = str(settings.FMRI.BoldSkipReorient.THREADS)
    BoldSkipReorient_node.source = Source(CPU_n=CPU_NUM, GPU_MB=GPU_MB, RAM_MB=RAM_MB,
                                          IO_write_MB=IO_WRITE_MB, IO_read_MB=IO_READ_MB)

    return BoldSkipReorient_node


def create_StcMc_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    """Slice-timing correction and Motion correction

        Inputs
        ------
        subject_id
            Subject id
        data_path
            BIDS dir
        derivative_deepprep_path
            Bold preprocessing dir
        task
            ``motor`` or ``rest``
        preprocess_method
            ``task`` or ``rest``
        atlas_type
            Atlas tyle (eg: ``MNI152_T1_2mm``)

        Outputs
        -------
        faln_fname
            File of functional to anatomical alignment
        mc_fname
            File of Motion Correction

        See also
        --------
        * :py:func:`deepprep.interface.bold_node.StcMc`

    """
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    derivative_deepprep_path = Path(settings.BOLD_PREPROCESS_DIR)
    data_path = Path(settings.BIDS_DIR)

    StcMc_node = Node(StcMc(), name=f'{subject_id}_fMRI_StcMc_node')
    StcMc_node.inputs.subject_id = subject_id
    StcMc_node.inputs.task = task
    StcMc_node.inputs.data_path = data_path
    StcMc_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    StcMc_node.inputs.atlas_type = atlas_type
    StcMc_node.inputs.preprocess_method = preprocess_method

    StcMc_node.base_dir = workflow_cached_dir / subject_id
    CPU_NUM = settings.FMRI.StcMc.CPU_NUM
    RAM_MB = settings.FMRI.StcMc.RAM_MB
    GPU_MB = settings.FMRI.StcMc.GPU_MB
    IO_WRITE_MB = settings.FMRI.StcMc.IO_WRITE_MB
    IO_READ_MB = settings.FMRI.StcMc.IO_READ_MB
    StcMc_node.inputs.multiprocess = str(settings.FMRI.StcMc.THREADS)
    StcMc_node.source = Source(CPU_n=CPU_NUM, GPU_MB=GPU_MB, RAM_MB=RAM_MB,
                               IO_write_MB=IO_WRITE_MB, IO_read_MB=IO_READ_MB)

    return StcMc_node


def create_Register_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    """Registration of functional MRI images to structural MRI images

        Inputs
        ------
        subject_id
            Subject id
        data_path
            BIDS dir
        derivative_deepprep_path
            Bold preprocessing dir
        task
            ``motor`` or ``rest``
        preprocess_method
            ``task`` or ``rest``
        atlas_type
            Atlas tyle (eg: ``MNI152_T1_2mm``)
        mov
            File of Motion Correction

        Outputs
        -------
        reg
            Registration result file

        See also
        --------
        * :py:func:`deepprep.interface.bold_node.Register`

    """
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    derivative_deepprep_path = Path(settings.BOLD_PREPROCESS_DIR)
    data_path = Path(settings.BIDS_DIR)

    Register_node = Node(Register(), name=f'{subject_id}_fMRI_Register_node')
    Register_node.inputs.subject_id = subject_id
    Register_node.inputs.task = task
    Register_node.inputs.data_path = data_path
    Register_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    Register_node.inputs.atlas_type = atlas_type
    Register_node.inputs.preprocess_method = preprocess_method

    Register_node.base_dir = workflow_cached_dir / subject_id
    CPU_NUM = settings.FMRI.Register.CPU_NUM
    RAM_MB = settings.FMRI.Register.RAM_MB
    GPU_MB = settings.FMRI.Register.GPU_MB
    IO_WRITE_MB = settings.FMRI.Register.IO_WRITE_MB
    IO_READ_MB = settings.FMRI.Register.IO_READ_MB
    Register_node.inputs.multiprocess = str(settings.FMRI.Register.THREADS)
    Register_node.source = Source(CPU_n=CPU_NUM, GPU_MB=GPU_MB, RAM_MB=RAM_MB,
                                  IO_write_MB=IO_WRITE_MB, IO_read_MB=IO_READ_MB)

    return Register_node


def create_Mkbrainmask_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    """ Make brainmask

        Inputs
        ------
        subject_id
            Subject id
        subjects_dir
            Recon dir
        data_path
            BIDS dir
        derivative_deepprep_path
            Bold preprocessing dir
        task
            ``motor`` or ``rest``
        preprocess_method
            ``task`` or ``rest``
        atlas_type
            Atlas tyle (eg: ``MNI152_T1_2mm``)
        mov
            File of Motion Correction
        reg
            Registration result file
        seg
            aparc+aseg file

        Outputs
        -------
        func


    """
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    derivative_deepprep_path = Path(settings.BOLD_PREPROCESS_DIR)
    data_path = Path(settings.BIDS_DIR)
    subjects_dir = Path(settings.SUBJECTS_DIR)

    Mkbrainmask_node = Node(MkBrainmask(), name=f'{subject_id}_fMRI_MkBrainmask_node')
    Mkbrainmask_node.inputs.subject_id = subject_id
    Mkbrainmask_node.inputs.subjects_dir = subjects_dir
    Mkbrainmask_node.inputs.task = task
    Mkbrainmask_node.inputs.data_path = data_path
    Mkbrainmask_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    Mkbrainmask_node.inputs.atlas_type = atlas_type
    Mkbrainmask_node.inputs.preprocess_method = preprocess_method

    Mkbrainmask_node.base_dir = workflow_cached_dir / subject_id
    CPU_NUM = settings.FMRI.Mkbrainmask.CPU_NUM
    RAM_MB = settings.FMRI.Mkbrainmask.RAM_MB
    GPU_MB = settings.FMRI.Mkbrainmask.GPU_MB
    IO_WRITE_MB = settings.FMRI.Mkbrainmask.IO_WRITE_MB
    IO_READ_MB = settings.FMRI.Mkbrainmask.IO_READ_MB
    Mkbrainmask_node.inputs.multiprocess = str(settings.FMRI.Mkbrainmask.THREADS)
    Mkbrainmask_node.source = Source(CPU_n=CPU_NUM, GPU_MB=GPU_MB, RAM_MB=RAM_MB,
                                     IO_write_MB=IO_WRITE_MB, IO_read_MB=IO_READ_MB)

    return Mkbrainmask_node


def create_VxmRegistraion_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    """Voxelmorph registration

        Inputs
        ------
        subject_id
           Subject id
        subjects_dir
           Recon dir
        data_path
            BIDS dir
        derivative_deepprep_path
            Bold preprocessing dir
        model_file
            Voxelmorph model
        vxm_model_path
            The address for save Voxelmorph model
        atlas_type
            Atlas tyle (eg: ``MNI152_T1_2mm``)
        task
            ``motor`` or ``rest``
        preprocess_method
            ``task`` or ``rest``
        norm
            Normalization file
        atlas
            MNI152 template
        vxm_atlas
            Vxm_MNI152 template
        vxm_atlas_npz
            Vxm_MNI152 template npz form
        vxm2atlas_trf
            Trf from vxm_MNI152 to MNI152

        Outputs
        -------
        vxm_warped
            Norm_norigid_vxm file
        warped
            Norm_norigid file
        vxm_warp
            Vxm_deformation_field_from_norm_to_vxm file
        vxm_input_npz
            Norm_affined_vxm file
        trf
            Ants_affine_trf_from_norm_to_vxm file

        See also
        --------
        * :py:func:`deepprep.interface.vxm_node.VxmRegistraion`

    """

    subjects_dir = Path(settings.SUBJECTS_DIR)
    derivative_deepprep_path = settings.BOLD_PREPROCESS_DIR
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    data_path = Path(settings.BIDS_DIR)
    vxm_model_path = Path(settings.VXM_MODEL_PATH)
    gpuid = settings.DEVICE

    VxmRegistraion_node = Node(VxmRegistraion(), name=f'{subject_id}_fMRI_VxmRegistraion_node')
    VxmRegistraion_node.inputs.subject_id = subject_id
    VxmRegistraion_node.inputs.data_path = data_path
    VxmRegistraion_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    VxmRegistraion_node.inputs.subjects_dir = subjects_dir
    VxmRegistraion_node.inputs.model_file = vxm_model_path / atlas_type / 'model.h5'
    VxmRegistraion_node.inputs.vxm_model_path = vxm_model_path
    VxmRegistraion_node.inputs.atlas_type = atlas_type
    VxmRegistraion_node.inputs.task = task
    VxmRegistraion_node.inputs.preprocess_method = preprocess_method
    VxmRegistraion_node.inputs.gpuid = gpuid

    VxmRegistraion_node.base_dir = workflow_cached_dir / subject_id

    CPU_NUM = settings.FMRI.VxmRegistraion.CPU_NUM
    RAM_MB = settings.FMRI.VxmRegistraion.RAM_MB
    GPU_MB = settings.FMRI.VxmRegistraion.GPU_MB
    VxmRegistraion_node.source = Source(CPU_n=CPU_NUM, GPU_MB=GPU_MB, RAM_MB=RAM_MB, IO_write_MB=0, IO_read_MB=0)

    return VxmRegistraion_node


def create_VxmRegNormMNI152_node(subject_id: str, task: str, atlas_type: str, preprocess_method: str, settings):
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    derivative_deepprep_path = Path(settings.BOLD_PREPROCESS_DIR)
    data_path = Path(settings.BIDS_DIR)
    subjects_dir = Path(settings.SUBJECTS_DIR)
    vxm_model_path = Path(settings.VXM_MODEL_PATH)
    resource_dir = Path(settings.RESOURCE_DIR)
    gpuid = settings.DEVICE

    VxmRegNormMNI152_node = Node(VxmRegNormMNI152(), name=f'{subject_id}_fMRI_VxmRegNormMNI152_node')
    VxmRegNormMNI152_node.inputs.subjects_dir = subjects_dir
    VxmRegNormMNI152_node.inputs.subject_id = subject_id
    VxmRegNormMNI152_node.inputs.atlas_type = atlas_type
    VxmRegNormMNI152_node.inputs.task = task
    VxmRegNormMNI152_node.inputs.data_path = data_path
    VxmRegNormMNI152_node.inputs.preprocess_method = preprocess_method
    VxmRegNormMNI152_node.inputs.vxm_model_path = vxm_model_path
    VxmRegNormMNI152_node.inputs.resource_dir = resource_dir
    VxmRegNormMNI152_node.inputs.derivative_deepprep_path = derivative_deepprep_path
    VxmRegNormMNI152_node.inputs.gpuid = gpuid
    VxmRegNormMNI152_node.inputs.batch_size = str(settings.fMRI.VxmRegNormMNI152.BATCH_SIZE)

    VxmRegNormMNI152_node.inputs.fsaverage6_space = settings.fMRI.fsaverage6_space
    VxmRegNormMNI152_node.inputs.fs_native_space = settings.fMRI.fs_native_space

    VxmRegNormMNI152_node.base_dir = workflow_cached_dir / subject_id
    CPU_NUM = settings.FMRI.VxmRegNormMNI152.CPU_NUM
    RAM_MB = settings.FMRI.VxmRegNormMNI152.RAM_MB
    GPU_MB = settings.FMRI.VxmRegNormMNI152.GPU_MB
    IO_WRITE_MB = settings.FMRI.VxmRegNormMNI152.IO_WRITE_MB
    IO_READ_MB = settings.FMRI.VxmRegNormMNI152.IO_READ_MB
    VxmRegNormMNI152_node.source = Source(CPU_n=CPU_NUM, GPU_MB=GPU_MB, RAM_MB=RAM_MB,
                                          IO_write_MB=IO_WRITE_MB, IO_read_MB=IO_READ_MB)

    return VxmRegNormMNI152_node


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

    print('#####################################################1#####################################################')

    node = create_BoldSkipReorient_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                                        preprocess_method=preprocess_method_test, settings=settings)
    node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    print('#####################################################2#####################################################')
    node = create_StcMc_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                             preprocess_method=preprocess_method_test, settings=settings)
    node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    print('#####################################################3#####################################################')
    node = create_Register_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                                preprocess_method=preprocess_method_test, settings=settings)
    node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    print('#####################################################4#####################################################')
    node = create_Mkbrainmask_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                                   preprocess_method=preprocess_method_test, settings=settings)
    node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    print('#####################################################5#####################################################')
    node = create_VxmRegistraion_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                             preprocess_method=preprocess_method_test, settings=settings)
    node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    print('####################################################6####################################################')
    node = create_VxmRegNormMNI152_node(subject_id=subject_id_test, task=task_test, atlas_type=atlas_type_test,
                                        preprocess_method=preprocess_method_test, settings=settings)
    node.run()
    # sub_node = node.interface.create_sub_node()
    # sub_node.run()

    return


if __name__ == '__main__':
    from config import settings as settings_main
    create_node_t(settings_main)  # 测试
