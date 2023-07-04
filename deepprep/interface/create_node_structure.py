from deepprep.interface.freesurfer_node import *
from deepprep.interface.fastcsr_node import *
from deepprep.interface.fastsurfer_node import *
from deepprep.interface.sagereg_node import *
from deepprep.interface.node_source import Source
import sys
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


def create_OrigAndRawavg_node(subject_id: str, t1w_files: list, settings):
    """Use ``Recon-all`` in Freesurfer to get Orig And Rawavg files

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        t1w_files
            Path and filename of the structural MRI image file to analyze
        threads
            Number of threads used in recon-all

        Outputs
        -------
        orig_file
            Raw T1-weighted MRI image
        rawavg_file
            Mean image of raw T1-weighted MRI image

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.OrigAndRawavg`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = settings.WORKFLOW_CACHED_DIR

    Origandrawavg_node = Node(OrigAndRawavg(), f'{subject_id}_sMRI_OrigAndRawavg_node')
    Origandrawavg_node.inputs.t1w_files = t1w_files
    Origandrawavg_node.inputs.subjects_dir = subjects_dir
    Origandrawavg_node.inputs.subject_id = subject_id

    Origandrawavg_node.base_dir = Path(workflow_cached_dir) / subject_id

    THREADS = settings.FS_THREADS  # FreeSurfer threads
    CPU_NUM = THREADS
    # CPU_NUM = settings.SMRI.OrigAndRawavg.CPU_NUM
    RAM_MB = settings.SMRI.OrigAndRawavg.RAM_MB

    Origandrawavg_node.inputs.threads = THREADS
    Origandrawavg_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return Origandrawavg_node


def create_Segment_node(subject_id: str, settings):
    """Segment MRI images into multiple distinct tissue and structural regions

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        python_interpret
            The python interpret to use
        eval_py
            FastSurfer segmentation script program
        network_sagittal_path
            Path to pre-trained weights of sagittal network
        network_coronal_path
            Pre-trained weights of coronal network
        network_axial_path
            Pre-trained weights of axial network
        orig_file
            Raw T1-weighted MRI image
        conformed_file
            Conformed file

        Outputs
        -------
        aparc_DKTatlas_aseg_deep
            Generated tag file for deep
        aparc_DKTatlas_aseg_orig
            Generated tag file

        See also
        --------
        * :py:func:`deepprep.interface.fastsurfer_node.Segment`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    fastsurfer_home = Path(settings.FASTSURFER_HOME)
    workflow_cached_dir = settings.WORKFLOW_CACHED_DIR
    python_interpret = sys.executable

    fastsurfer_eval = fastsurfer_home / 'FastSurferCNN' / 'eval.py'  # inference script
    weight_dir = fastsurfer_home / 'checkpoints'  # model checkpoints dir

    network_sagittal_path = weight_dir / "Sagittal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_coronal_path = weight_dir / "Coronal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_axial_path = weight_dir / "Axial_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"

    Segment_node = Node(Segment(), f'{subject_id}_sMRI_Segment_node')
    Segment_node.inputs.subjects_dir = subjects_dir
    Segment_node.inputs.subject_id = subject_id
    Segment_node.inputs.python_interpret = python_interpret
    Segment_node.inputs.eval_py = fastsurfer_eval
    Segment_node.inputs.network_sagittal_path = network_sagittal_path
    Segment_node.inputs.network_coronal_path = network_coronal_path
    Segment_node.inputs.network_axial_path = network_axial_path

    Segment_node.base_dir = Path(workflow_cached_dir) / subject_id

    CPU_NUM = settings.SMRI.Segment.CPU_NUM
    RAM_MB = settings.SMRI.Segment.RAM_MB
    GPU_MB = settings.SMRI.Segment.GPU_MB
    Segment_node.source = Source(CPU_n=CPU_NUM, GPU_MB=GPU_MB, RAM_MB=RAM_MB)

    return Segment_node


def create_Noccseg_node(subject_id: str, settings):
    """Removal of the orbital region in MRI images ======temp

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        python_interpret
            The python interpret to use
        reduce_to_aseg_py
            Reduce to aseg program
        aparc_DKTatlas_aseg_deep
            Generated tag file for deep

        Outputs
        -------
        aseg_noCCseg_file
            Segmentation result without corpus callosum label
        mask_file
            Mask file

        See also
        --------
        * :py:func:`deepprep.interface.fastsurfer_node.Noccseg`


    """
    fastsurfer_home = Path(settings.FASTSURFER_HOME)
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    python_interpret = sys.executable

    reduce_to_aseg_py = fastsurfer_home / 'recon_surf' / 'reduce_to_aseg.py'

    Noccseg_node = Node(Noccseg(), f'{subject_id}_sMRI_Noccseg_node')
    Noccseg_node.inputs.python_interpret = python_interpret
    Noccseg_node.inputs.reduce_to_aseg_py = reduce_to_aseg_py
    Noccseg_node.inputs.subject_id = subject_id
    Noccseg_node.inputs.subjects_dir = subjects_dir

    Noccseg_node.base_dir = Path(workflow_cached_dir) / subject_id

    CPU_NUM = settings.SMRI.Noccseg.CPU_NUM
    RAM_MB = settings.SMRI.Noccseg.RAM_MB
    Noccseg_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return Noccseg_node


def create_N4BiasCorrect_node(subject_id: str, settings):
    """Bias corrected

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        python_interpret
            The python interpret to use
        correct_py
                N4 bias correct program
        orig_file
            Raw T1-weighted MRI image
        mask_file
            Mask file

        Outputs
        -------
        orig_nu_file
            Orig nu file

        See also
        --------
        * :py:func:`deepprep.interface.fastsurfer_node.N4BiasCorrect`


    """
    fastsurfer_home = Path(settings.FASTSURFER_HOME)
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    python_interpret = sys.executable
    sub_mri_dir = subjects_dir / subject_id / "mri"
    correct_py = fastsurfer_home / "recon_surf" / "N4_bias_correct.py"

    orig_file = sub_mri_dir / "orig.mgz"
    mask_file = sub_mri_dir / "mask.mgz"

    N4_bias_correct_node = Node(N4BiasCorrect(), name=f'{subject_id}_sMRI_N4BiasCorrect_node')
    N4_bias_correct_node.inputs.subject_id = subject_id
    N4_bias_correct_node.inputs.subjects_dir = subjects_dir
    N4_bias_correct_node.inputs.python_interpret = python_interpret
    N4_bias_correct_node.inputs.correct_py = correct_py
    N4_bias_correct_node.inputs.mask_file = mask_file
    N4_bias_correct_node.inputs.orig_file = orig_file

    N4_bias_correct_node.base_dir = Path(workflow_cached_dir) / subject_id

    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.N4BiasCorrect.RAM_MB
    N4_bias_correct_node.inputs.threads = THREADS  # FastSurfer.N4BiasCorrect
    N4_bias_correct_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return N4_bias_correct_node


def create_TalairachAndNu_node(subject_id: str, settings):
    """Computing Talairach Transform and NU

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        threads
            Number of threads
        orig_nu_file
            Orig nu file
        orig_file
            Raw T1-weighted MRI image
        mni305
            MNI305 templet

        Outputs
        -------
        talairach_lta
            Transformation matrix from raw MRI image to Talairach space
        nu_file
            NU intensity corrected result

        See also
        --------
        * :py:func:`deepprep.interface.fastsurfer_node.TalairachAndNu`


    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    freesurfer_home = Path(settings.FREESURFER_HOME)

    sub_mri_dir = subjects_dir / subject_id / "mri"
    orig_nu_file = sub_mri_dir / "orig_nu.mgz"
    orig_file = sub_mri_dir / "orig.mgz"
    mni305 = freesurfer_home / "average" / "mni305.cor.mgz"

    TalairachAndNu_node = Node(TalairachAndNu(), name=f'{subject_id}_sMRI_TalairachAndNu_node')
    TalairachAndNu_node.inputs.subjects_dir = subjects_dir
    TalairachAndNu_node.inputs.subject_id = subject_id
    TalairachAndNu_node.inputs.mni305 = mni305
    TalairachAndNu_node.inputs.orig_nu_file = orig_nu_file
    TalairachAndNu_node.inputs.orig_file = orig_file

    TalairachAndNu_node.base_dir = Path(workflow_cached_dir) / subject_id

    CPU_NUM = settings.SMRI.TalairachAndNu.CPU_NUM
    RAM_MB = settings.SMRI.TalairachAndNu.RAM_MB
    TalairachAndNu_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return TalairachAndNu_node


def create_Brainmask_node(subject_id: str, settings):
    """Brainmask applies a mask volume ( typically skull stripped ) ref:freesurfer

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        need_t1
           Whether need T1 file
        nu_file
           Intensity normalized volume
        mask_file
           Mask volume

        Outputs
        -------
        brainmask_file
           Brainmask file
        norm_file
           Brain normalized result
        T1_file
           If ``need_t1==Ture`` ,T1 file will be generated

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.Brainmask`

       """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)

    Brainmask_node = Node(Brainmask(), name=f'{subject_id}_sMRI_Brainmask_node')
    Brainmask_node.inputs.subjects_dir = subjects_dir
    Brainmask_node.inputs.subject_id = subject_id
    Brainmask_node.inputs.need_t1 = True
    Brainmask_node.inputs.nu_file = subjects_dir / subject_id / 'mri' / 'nu.mgz'
    Brainmask_node.inputs.mask_file = subjects_dir / subject_id / 'mri' / 'mask.mgz'

    Brainmask_node.base_dir = Path(workflow_cached_dir) / subject_id
    CPU_NUM = settings.SMRI.Brainmask.CPU_NUM
    RAM_MB = settings.SMRI.Brainmask.RAM_MB
    Brainmask_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return Brainmask_node


def create_UpdateAseg_node(subject_id: str, settings):
    """Update aseg

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        python_interpret
            The python interpret to use
        paint_cc_file
            Paint cc into pred program
        aseg_noCCseg_file
            Remove the segmented files for the orbital region
        seg_file
            Generated tag file for deep

        Outputs
        -------
        aseg_auto_file
            Aseg including CC
        cc_up_file ?? output?

        aparc_aseg_file
            Generated tag file with cc for deep

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.UpdateAseg`


    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    fastsurfer_home = Path(settings.FASTSURFER_HOME)
    python_interpret = sys.executable
    subject_mri_dir = subjects_dir / subject_id / 'mri'

    paint_cc_file = fastsurfer_home / 'recon_surf' / 'paint_cc_into_pred.py'
    Updateaseg_node = Node(UpdateAseg(), name=f'{subject_id}_sMRI_UpdateAseg_node')
    Updateaseg_node.inputs.subjects_dir = subjects_dir
    Updateaseg_node.inputs.subject_id = subject_id
    Updateaseg_node.inputs.paint_cc_file = paint_cc_file
    Updateaseg_node.inputs.python_interpret = python_interpret
    Updateaseg_node.inputs.seg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'
    Updateaseg_node.inputs.aseg_noCCseg_file = subject_mri_dir / 'aseg.auto_noCCseg.mgz'

    Updateaseg_node.base_dir = Path(workflow_cached_dir) / subject_id
    CPU_NUM = settings.SMRI.UpdateAseg.CPU_NUM
    RAM_MB = settings.SMRI.UpdateAseg.RAM_MB
    Updateaseg_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return Updateaseg_node


def create_Filled_node(subject_id: str, settings):
    """Creating filled from brain

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        threads
            Number of threads
        aseg_auto_file
            Aseg including CC
        norm_file
            Brain normalized result
        brainmask_file
            Brainmask file
        talairach_lta
            Transformation matrix from raw MRI image to Talairach space

        Outputs
        -------
        aseg_presurf_file

        brain_file
            File after the second intensity correction
        brain_finalsurfs_file
            A file containing information about the cortical surface model of the entire brain
        wm_file
            White Matter file
        wm_filled
            The filled volume for the cortical reconstruction

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.Filled`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)

    settings.SUBJECTS_DIR = str(subjects_dir)

    Filled_node = Node(Filled(), name=f'{subject_id}_sMRI_Filled_node')
    Filled_node.inputs.subjects_dir = subjects_dir
    Filled_node.inputs.subject_id = subject_id
    Filled_node.inputs.aseg_auto_file = subjects_dir / subject_id / 'mri/aseg.auto.mgz'
    Filled_node.inputs.norm_file = subjects_dir / subject_id / 'mri/norm.mgz'
    Filled_node.inputs.brainmask_file = subjects_dir / subject_id / 'mri/brainmask.mgz'
    Filled_node.inputs.talairach_lta = subjects_dir / subject_id / 'mri/transforms/talairach.lta'

    Filled_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS  # FreeSurfer.recon_all
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.Filled.RAM_MB
    Filled_node.inputs.threads = THREADS
    Filled_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return Filled_node


def create_FastCSR_node(subject_id: str, settings):
    """Fast cortical surface reconstruction using FastCSR

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        python_interpret
            The python interpret to use
        fastcsr_py
            FastCSR script
        parallel_scheduling
            Parallel scheduling, ``on`` or ``off``
        orig_file
            Raw T1-weighted MRI image
        filled_file
            The filled volume for the cortical reconstruction
        aseg_presurf_file

        brainmask_file
            Brainmask file
        wm_file
            White Matter file
        brain_finalsurfs_file
            A file containing information about the cortical surface model of the entire brain

        Outputs
        -------
        hemi_orig_file
            Raw cortical surface model file
        hemi_orig_premesh_file

        See also
        --------
        * :py:func:`deepprep.interface.Fastcsr_node.FastCSR`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    python_interpret = sys.executable
    settings.SUBJECTS_DIR = str(subjects_dir)
    fastcsr_home = Path(settings.FASTCSR_HOME)
    fastcsr_py = fastcsr_home / 'pipeline.py'  # inference script

    Fastcsr_node = Node(FastCSR(), name=f'{subject_id}_sMRI_FastCSR_node')
    Fastcsr_node.inputs.python_interpret = python_interpret
    Fastcsr_node.inputs.fastcsr_py = fastcsr_py
    Fastcsr_node.inputs.subjects_dir = subjects_dir
    Fastcsr_node.inputs.subject_id = subject_id
    Fastcsr_node.inputs.orig_file = Path(subjects_dir) / subject_id / 'mri/orig.mgz'
    Fastcsr_node.inputs.filled_file = Path(subjects_dir) / subject_id / 'mri/filled.mgz'
    Fastcsr_node.inputs.aseg_presurf_file = Path(subjects_dir) / subject_id / 'mri/aseg.presurf.mgz'
    Fastcsr_node.inputs.brainmask_file = Path(subjects_dir) / subject_id / 'mri/brainmask.mgz'
    Fastcsr_node.inputs.wm_file = Path(subjects_dir) / subject_id / 'mri/wm.mgz'
    Fastcsr_node.inputs.brain_finalsurfs_file = Path(subjects_dir) / subject_id / 'mri/brain.finalsurfs.mgz'

    Fastcsr_node.inputs.model_path = os.path.join(settings.DEEPPREP_HOME, 'model/FastCSR')

    Fastcsr_node.base_dir = Path(workflow_cached_dir) / subject_id
    PARALLEL = settings.SMRI.FastCSR.PARALLEL
    CPU_NUM = settings.SMRI.FastCSR.CPU_NUM
    RAM_MB = settings.SMRI.FastCSR.RAM_MB
    GPU_MB = settings.SMRI.FastCSR.GPU_MB
    Fastcsr_node.inputs.parallel_scheduling = PARALLEL
    Fastcsr_node.source = Source(CPU_n=CPU_NUM, GPU_MB=GPU_MB, RAM_MB=RAM_MB)

    return Fastcsr_node


def create_WhitePreaparc1_node(subject_id: str, settings):
    """Generates surface files for cortical and white matter surfaces, curvature file for cortical thickness and surface
        file estimate for layer IV of cortical sheet.

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        aseg_presurf_file

        brain_finalsurfs_file
            A file containing information about the cortical surface model of the entire brain
        filled_file
            The filled volume for the cortical reconstruction
        wm_file
            White Matter file
        hemi_orig_file
            Raw cortical surface model file

        Outputs
        -------
        hemi_white_preaparc
            Hemi white matter surface file
        hemi_curv
            Hemi curvature
        hemi_area
            Hemi surface area
        hemi_cortex_label
            Hemi cortex label
        hemi_cortex_hipamyglabel
            Hemi cortex hipamyglabel
        autodet_gw_stats_hemi_dat

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.WhitePreaparc1`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    settings.SUBJECTS_DIR = str(subjects_dir)

    White_preaparc1_node = Node(WhitePreaparc1(), name=f'{subject_id}_sMRI_WhitePreaparc1_node')
    White_preaparc1_node.inputs.subjects_dir = subjects_dir
    White_preaparc1_node.inputs.subject_id = subject_id

    White_preaparc1_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.WhitePreaparc1.RAM_MB
    White_preaparc1_node.inputs.threads = THREADS
    White_preaparc1_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return White_preaparc1_node


# def create_SampleSegmentationToSurface_node(subject_id: str, settings):
#     """Sample segmentation to surface
#
#         Inputs
#         ------
#         subjects_id
#            Subject id
#         subjects_dir
#            Recon dir
#         python_interpret
#             The python interpret to use
#         freesurfer_home
#             Freesurfer home
#         hemi_DKTatlaslookup_file
#
#         smooth_aparc_file
#             smooth_aparc script
#         aparc_aseg_file
#             Generated tag file with cc for deep
#         hemi_white_preaparc_file
#             Hemi white matter surface file
#         hemi_cortex_label_file
#             Hemi cortex label
#
#         Outputs
#         -------
#         hemi_aparc_DKTatlas_mapped_prefix_file
#
#         hemi_aparc_DKTatlas_mapped_file
#
#         See also
#         --------
#         * :py:func:`deepprep.interface.fastsurfer_node.SampleSegmentationToSurface`
#
#     """
#     subjects_dir = Path(settings.SUBJECTS_DIR)
#     workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
#     python_interpret = sys.executable
#     freesurfer_home = Path(settings.FREESURFER_HOME)
#     fastsurfer_home = Path(settings.FASTSURFER_HOME)
#
#     smooth_aparc_file = fastsurfer_home / 'recon_surf' / 'smooth_aparc.py'
#
#     subject_mri_dir = subjects_dir / subject_id / 'mri'
#     subject_surf_dir = subjects_dir / subject_id / 'surf'
#     subject_label_dir = subjects_dir / subject_id / 'label'
#     lh_DKTatlaslookup_file = fastsurfer_home / 'recon_surf' / f'lh.DKTatlaslookup.txt'
#     rh_DKTatlaslookup_file = fastsurfer_home / 'recon_surf' / f'rh.DKTatlaslookup.txt'
#     settings.SUBJECTS_DIR = str(subjects_dir)
#
#     SampleSegmentationToSurfave_node = Node(SampleSegmentationToSurface(),
#                                             name=f'{subject_id}_sMRI_SampleSegmentationToSurface_node')
#     SampleSegmentationToSurfave_node.inputs.subjects_dir = subjects_dir
#     SampleSegmentationToSurfave_node.inputs.subject_id = subject_id
#     SampleSegmentationToSurfave_node.inputs.python_interpret = python_interpret
#     SampleSegmentationToSurfave_node.inputs.freesurfer_home = freesurfer_home
#     SampleSegmentationToSurfave_node.inputs.lh_DKTatlaslookup_file = lh_DKTatlaslookup_file
#     SampleSegmentationToSurfave_node.inputs.rh_DKTatlaslookup_file = rh_DKTatlaslookup_file
#     SampleSegmentationToSurfave_node.inputs.aparc_aseg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.withCC.mgz'
#     SampleSegmentationToSurfave_node.inputs.smooth_aparc_file = smooth_aparc_file
#     SampleSegmentationToSurfave_node.inputs.lh_white_preaparc_file = subject_surf_dir / f'lh.white.preaparc'
#     SampleSegmentationToSurfave_node.inputs.rh_white_preaparc_file = subject_surf_dir / f'rh.white.preaparc'
#     SampleSegmentationToSurfave_node.inputs.lh_cortex_label_file = subject_label_dir / f'lh.cortex.label'
#     SampleSegmentationToSurfave_node.inputs.rh_cortex_label_file = subject_label_dir / f'rh.cortex.label'
#
#     SampleSegmentationToSurfave_node.base_dir = Path(workflow_cached_dir) / subject_id
#     THREADS = settings.FS_THREADS
#     CPU_NUM = settings.SMRI.Segment.CPU_NUM
#     RAM_MB = settings.SMRI.Segment.RAM_MB
#     GPU_MB = settings.SMRI.Segment.GPU_MB
#     SampleSegmentationToSurfave_node.source = Source(CPU_n=2, GPU_MB=0, RAM_MB=4000)
#
#     return SampleSegmentationToSurfave_node


def create_InflatedSphere_node(subject_id: str, settings):
    """Generate inflated file

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        threads
            Number of threads
        hemi_white_preaparc_file
            Hemi white matter surface file

        Outputs
        -------
        hemi_smoothwm
            Hemi smoothwm
        hemi_inflated
            Hemi inflated
        hemi_sulc
            Hemi sulcal depth
        hemi_sphere
            Hemi sphere

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.InflatedSphere`


    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)

    lh_white_preaparc_file = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
    rh_white_preaparc_file = subjects_dir / subject_id / "surf" / "rh.white.preaparc"

    InflatedSphere_node = Node(InflatedSphere(), f'{subject_id}_sMRI_InflatedSphere_node')
    InflatedSphere_node.inputs.subjects_dir = subjects_dir
    InflatedSphere_node.inputs.subject_id = subject_id
    InflatedSphere_node.inputs.lh_white_preaparc_file = lh_white_preaparc_file
    InflatedSphere_node.inputs.rh_white_preaparc_file = rh_white_preaparc_file

    InflatedSphere_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.InflatedSphere.RAM_MB
    InflatedSphere_node.inputs.threads = THREADS
    InflatedSphere_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return InflatedSphere_node


def create_SageReg_node(subject_id: str, settings):
    """Cortical surface registration

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        python_interpret
            The python interpret to use
        sagereg_py
            Registration script
        device
            GPU set
        freesurfer_home
            Freesurfer home
        hemi_sulc
            Hemi sulcal depth
        hemi_curv
            Hemi curvature
        hemi_sphere
            Hemi sphere

        Outputs
        -------
        hemi_sphere_reg
            Registered sphere

        See also
        --------
        * :py:func:`deepprep.interface.Sagereg_node.SageReg`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    sagereg_home = Path(settings.SAGEREG_HOME)
    freesurfer_home = Path(settings.FREESURFER_HOME)

    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    device = settings.DEVICE

    python_interpret = sys.executable
    sagereg_py = sagereg_home / 'predict.py'  # inference script

    Sagereg_node = Node(SageReg(), f'{subject_id}_sMRI_SageReg_node')
    Sagereg_node.inputs.sagereg_py = sagereg_py
    Sagereg_node.inputs.python_interpret = python_interpret
    Sagereg_node.inputs.device = device

    Sagereg_node.inputs.deepprep_home = Path(settings.DEEPPREP_HOME)
    Sagereg_node.inputs.subjects_dir = subjects_dir
    Sagereg_node.inputs.subject_id = subject_id
    Sagereg_node.inputs.freesurfer_home = freesurfer_home
    Sagereg_node.inputs.lh_sulc = Path(subjects_dir) / subject_id / f'surf/lh.sulc'
    Sagereg_node.inputs.rh_sulc = Path(subjects_dir) / subject_id / f'surf/rh.sulc'
    Sagereg_node.inputs.lh_curv = Path(subjects_dir) / subject_id / f'surf/lh.curv'
    Sagereg_node.inputs.rh_curv = Path(subjects_dir) / subject_id / f'surf/rh.curv'
    Sagereg_node.inputs.lh_sphere = Path(subjects_dir) / subject_id / f'surf/lh.sphere'
    Sagereg_node.inputs.rh_sphere = Path(subjects_dir) / subject_id / f'surf/rh.sphere'

    Sagereg_node.base_dir = Path(workflow_cached_dir) / subject_id
    CPU_NUM = settings.SMRI.Sagereg.CPU_NUM
    RAM_MB = settings.SMRI.Sagereg.RAM_MB
    GPU_MB = settings.SMRI.Sagereg.GPU_MB
    Sagereg_node.source = Source(CPU_n=CPU_NUM, GPU_MB=GPU_MB, RAM_MB=RAM_MB)

    return Sagereg_node


def create_JacobianAvgcurvCortparc_node(subject_id: str, settings):
    """Computes how much the white surface was distorted,resamples the average curvature from the atlas to that of
        the subject and assigns a neuroanatomical label to each location on the cortical surface.

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        threads
            Number of threads
        hemi_white_preaparc_file
            Hemi white matter surface file
        hemi_sphere_reg
            Registered sphere
        aseg_presurf_file

        hemi_cortex_label_file
            Hemi cortex label

        Outputs
        -------
        hemi_jacobian_white
            Hemi jacobian white
        hemi_avg_curv
            Hemi average curvature
        hemi_aparc_annot
            Hemi aparc annot

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.JacobianAvgcurvCortparc`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)

    JacobianAvgcurvCortparc_node = Node(JacobianAvgcurvCortparc(), f'{subject_id}_sMRI_JacobianAvgcurvCortparc_node')
    JacobianAvgcurvCortparc_node.inputs.subjects_dir = subjects_dir
    JacobianAvgcurvCortparc_node.inputs.subject_id = subject_id

    JacobianAvgcurvCortparc_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.JacobianAvgcurvCortparc.RAM_MB
    JacobianAvgcurvCortparc_node.inputs.threads = THREADS  # FreeSurfer.recon_all
    JacobianAvgcurvCortparc_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return JacobianAvgcurvCortparc_node


def create_WhitePialThickness1_node(subject_id: str, settings):
    """Generate white,pial and thickness

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        threads
            Number of threads
        hemi_white_preaparc
            Hemi white matter surface file
        aseg_presurf_file

        brain_finalsurfs
            A file containing information about the cortical surface model of the entire brain
        wm_file
            White Matter file
        hemi_aparc_annot
            Hemi aparc annot
        hemi_cortex_label
            Hemi cortex label

        Outputs
        -------

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)

    White_pial_thickness1_node = Node(WhitePialThickness1(), name=f'{subject_id}_sMRI_WhitePialThickness1_node')
    White_pial_thickness1_node.inputs.subjects_dir = subjects_dir
    White_pial_thickness1_node.inputs.subject_id = subject_id
    White_pial_thickness1_node.inputs.lh_white_preaparc = subjects_dir / subject_id / "surf" / "lh.white.preaparc"
    White_pial_thickness1_node.inputs.rh_white_preaparc = subjects_dir / subject_id / "surf" / "rh.white.preaparc"
    White_pial_thickness1_node.inputs.aseg_presurf = subjects_dir / subject_id / "mri" / "aseg.presurf.mgz"
    White_pial_thickness1_node.inputs.brain_finalsurfs = subjects_dir / subject_id / "mri" / "brain.finalsurfs.mgz"
    White_pial_thickness1_node.inputs.wm_file = subjects_dir / subject_id / "mri" / "wm.mgz"
    White_pial_thickness1_node.inputs.lh_aparc_annot = subjects_dir / subject_id / "label" / "lh.aparc.annot"
    White_pial_thickness1_node.inputs.rh_aparc_annot = subjects_dir / subject_id / "label" / "rh.aparc.annot"
    White_pial_thickness1_node.inputs.lh_cortex_label = subjects_dir / subject_id / "label" / "lh.cortex.label"
    White_pial_thickness1_node.inputs.rh_cortex_label = subjects_dir / subject_id / "label" / "rh.cortex.label"

    White_pial_thickness1_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.WhitePialThickness1.RAM_MB
    White_pial_thickness1_node.inputs.threads = THREADS
    White_pial_thickness1_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return White_pial_thickness1_node


def create_Curvstats_node(subject_id: str, settings):
    """Compute the mean and variances for a curvature file

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        threads
            Number of threads
        hemi_curv
            Hemi curvature
        hemi_sulc
            Hemi sulcal depth
        hemi_smoothwm
            Hemi smoothwm

        Outputs
        -------
        hemi_curv_stats
            The mean and variances for a curvature file

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.Curvstats`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)

    Curvstats_node = Node(Curvstats(), name=f'{subject_id}_sMRI_Curvstats_node')
    Curvstats_node.inputs.subjects_dir = subjects_dir
    Curvstats_node.inputs.subject_id = subject_id
    subject_surf_dir = subjects_dir / subject_id / "surf"

    Curvstats_node.inputs.lh_smoothwm = subject_surf_dir / f'lh.smoothwm'
    Curvstats_node.inputs.rh_smoothwm = subject_surf_dir / f'rh.smoothwm'
    Curvstats_node.inputs.lh_curv = subject_surf_dir / f'lh.curv'
    Curvstats_node.inputs.rh_curv = subject_surf_dir / f'rh.curv'
    Curvstats_node.inputs.lh_sulc = subject_surf_dir / f'lh.sulc'
    Curvstats_node.inputs.rh_sulc = subject_surf_dir / f'rh.sulc'

    Curvstats_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.Curvstats.RAM_MB
    Curvstats_node.inputs.threads = THREADS
    Curvstats_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return Curvstats_node


def create_BalabelsMult_node(subject_id: str, settings):
    """ !!!Too much inputs and outputs


    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    subject_surf_dir = subjects_dir / subject_id / 'surf'

    BalabelsMult_node = Node(BalabelsMult(), name=f'{subject_id}_sMRI_BalabelsMult_node')
    BalabelsMult_node.inputs.subjects_dir = subjects_dir
    BalabelsMult_node.inputs.subject_id = subject_id
    BalabelsMult_node.inputs.freesurfer_dir = settings.FREESURFER_HOME

    BalabelsMult_node.inputs.lh_sphere_reg = subject_surf_dir / f'lh.sphere.reg'
    BalabelsMult_node.inputs.rh_sphere_reg = subject_surf_dir / f'rh.sphere.reg'
    BalabelsMult_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    BalabelsMult_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    BalabelsMult_node.inputs.fsaverage_label_dir = Path(settings.FREESURFER_HOME) / "subjects/fsaverage/label"

    BalabelsMult_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.BalabelsMult.RAM_MB
    BalabelsMult_node.inputs.threads = THREADS
    BalabelsMult_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return BalabelsMult_node


def create_Cortribbon_node(subject_id: str, settings):
    """Creates binary volume masks of the cortical ribbon

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        threads
            Number of threads
        aseg_presurf_file

        hemi_white
            Hemi white
        hemi_pial
            Hemi pial

        Outputs
        -------
        hemi_ribbon
            Hemi cortical ribbon
        ribbon
            Ribbon file

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.Cortribbon`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'

    Cortribbon_node = Node(Cortribbon(), name=f'{subject_id}_sMRI_Cortribbon_node')
    Cortribbon_node.inputs.subjects_dir = subjects_dir
    Cortribbon_node.inputs.subject_id = subject_id

    Cortribbon_node.inputs.aseg_presurf_file = subject_mri_dir / 'aseg.presurf.mgz'
    Cortribbon_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Cortribbon_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    Cortribbon_node.inputs.lh_pial = subject_surf_dir / f'lh.pial'
    Cortribbon_node.inputs.rh_pial = subject_surf_dir / f'rh.pial'

    Cortribbon_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.Cortribbon.RAM_MB
    Cortribbon_node.inputs.threads = THREADS
    Cortribbon_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return Cortribbon_node


def create_Parcstats_node(subject_id: str, settings):
    """Compute statistics of cortical partitions and register cortical partition models with other anatomical structures
        to generate new partition models

        Inputs
        ------
        subjects_id
           Subject id
        subjects_dir
           Recon dir
        threads
            Number of threads
        hemi_aparc_annot
            Hemi aparc annot
        wm_file
            White Matter file
        ribbon_file
            Ribbon file
        hemi_white
            Hemi white
        hemi_pial
            Hemi pial
        hemi_thickness
            Hemi thickness

        Outputs
        -------
        aseg_file
            Volume information of each region in the MRI image
        hemi_aparc_stats
            Statistics for the hemi cerebral cortex
        hemi_aparc_pial_stats
            Statistics of the pial surface of the hemi cerebral cortex
        aparc_annot_ctab
            File containing color and label information
        aseg_presurf_hypos
            File of volume information for various regions in the MRI image
        hemi_wg_pct_stats
            Statistics of hemi brain white matter and gray matter

        See also
        --------
        * :py:func:`deepprep.interface.freesurfer_node.Parcstats`

    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)

    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'

    Parcstats_node = Node(Parcstats(), name=f'{subject_id}_sMRI_Parcstats_node')
    Parcstats_node.inputs.subjects_dir = subjects_dir
    Parcstats_node.inputs.subject_id = subject_id

    Parcstats_node.inputs.lh_aparc_annot = subject_label_dir / f'lh.aparc.annot'
    Parcstats_node.inputs.rh_aparc_annot = subject_label_dir / f'rh.aparc.annot'
    Parcstats_node.inputs.wm_file = subject_mri_dir / 'wm.mgz'
    Parcstats_node.inputs.ribbon_file = subject_mri_dir / 'ribbon.mgz'
    Parcstats_node.inputs.lh_white = subject_surf_dir / f'lh.white'
    Parcstats_node.inputs.rh_white = subject_surf_dir / f'rh.white'
    Parcstats_node.inputs.lh_pial = subject_surf_dir / f'lh.pial'
    Parcstats_node.inputs.rh_pial = subject_surf_dir / f'rh.pial'
    Parcstats_node.inputs.lh_thickness = subject_surf_dir / f'lh.thickness'
    Parcstats_node.inputs.rh_thickness = subject_surf_dir / f'rh.thickness'

    Parcstats_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.Parcstats.RAM_MB
    Parcstats_node.inputs.threads = THREADS
    Parcstats_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return Parcstats_node


def create_Aseg7_node(subject_id: str, settings):
    """


    """
    subjects_dir = Path(settings.SUBJECTS_DIR)
    workflow_cached_dir = Path(settings.WORKFLOW_CACHED_DIR)

    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'

    Aseg7_node = Node(Aseg7(), name=f'{subject_id}_sMRI_Aseg7_node')
    Aseg7_node.inputs.subjects_dir = subjects_dir
    Aseg7_node.inputs.subject_id = subject_id
    Aseg7_node.inputs.aseg_file = subject_mri_dir / 'aseg.mgz'
    Aseg7_node.inputs.lh_cortex_label = subject_label_dir / 'lh.cortex.label'
    Aseg7_node.inputs.lh_white = subject_surf_dir / 'lh.white'
    Aseg7_node.inputs.lh_pial = subject_surf_dir / 'lh.pial'
    Aseg7_node.inputs.lh_aparc_annot = subject_label_dir / 'lh.aparc.annot'
    Aseg7_node.inputs.rh_cortex_label = subject_label_dir / 'rh.cortex.label'
    Aseg7_node.inputs.rh_white = subject_surf_dir / 'rh.white'
    Aseg7_node.inputs.rh_pial = subject_surf_dir / 'rh.pial'
    Aseg7_node.inputs.rh_aparc_annot = subject_label_dir / 'rh.aparc.annot'

    Aseg7_node.base_dir = Path(workflow_cached_dir) / subject_id
    THREADS = settings.FS_THREADS
    CPU_NUM = THREADS
    RAM_MB = settings.SMRI.Aseg7.RAM_MB
    Aseg7_node.inputs.threads = THREADS
    Aseg7_node.source = Source(CPU_n=CPU_NUM, RAM_MB=RAM_MB)

    return Aseg7_node


def create_node_t(settings):
    from interface.run import set_envrion

    bids_data_dir_test = '/mnt/ngshare/test_Time_one_sub/MSC'
    subjects_dir_test = Path('/mnt/ngshare/temp/test_MSC_DeepPrep_Recon')
    bold_preprocess_dir_test = Path('/mnt/ngshare/DeepPrep_workflow_test/test_MSC_BoldPreprocess')
    workflow_cached_dir_test = '/mnt/ngshare/DeepPrep_workflow_test/test_MSC_WorkflowfsT1'

    if not subjects_dir_test.exists():
        subjects_dir_test.mkdir(parents=True, exist_ok=True)

    if not bold_preprocess_dir_test.exists():
        bold_preprocess_dir_test.mkdir(parents=True, exist_ok=True)

    settings.SUBJECTS_DIR = str(subjects_dir_test)
    settings.BOLD_PREPROCESS_DIR = str(bold_preprocess_dir_test)
    settings.WORKFLOW_CACHED_DIR = str(workflow_cached_dir_test)
    settings.BIDS_DIR = bids_data_dir_test
    settings.DEVICE = 'cuda'

    set_envrion(freesurfer_home=settings.FREESURFER_HOME,
                java_home=settings.JAVA_HOME,
                subjects_dir=settings.SUBJECTS_DIR,
                )

    atlas_type_test = 'MNI152_T1_2mm'
    task_test = 'rest'
    preprocess_method_test = 'rest'

    settings.FMRI.ATLAS_SPACE = atlas_type_test
    settings.FMRI.TASK = task_test
    settings.FMRI.PREPROCESS_TYPE = preprocess_method_test

    settings.RECON_ONLY = 'True'
    settings.BOLD_ONLY = 'False'

    subject_id_test = 'sub-MSC01'
    t1w_files = ['/mnt/ngshare/temp/MSC/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz']
    # t1w_files = ['/mnt/ngshare/DeepPrep_workflow_test/sub-R07T1_ses-01_T1w.nii.gz']

    # 测试
    # node = create_WhitePreaparc1_node(subject_id=subject_id_test)
    # node.run()
    # exit()

    # 测试
    node = create_OrigAndRawavg_node(subject_id=subject_id_test, t1w_files=t1w_files, settings=settings)
    node.run()

    node = create_Segment_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_Noccseg_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_N4BiasCorrect_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_TalairachAndNu_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_Brainmask_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_UpdateAseg_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_Filled_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_FastCSR_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_WhitePreaparc1_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_InflatedSphere_node(subject_id=subject_id_test, settings=settings)
    node.run()

    # exit()

    node = create_SageReg_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_JacobianAvgcurvCortparc_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_WhitePialThickness1_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_Curvstats_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_Cortribbon_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_Parcstats_node(subject_id=subject_id_test, settings=settings)
    node.run()

    node = create_Aseg7_node(subject_id=subject_id_test, settings=settings)
    node.run()


if __name__ == '__main__':
    from config import settings as main_settings
    create_node_t(main_settings)  # 测试
