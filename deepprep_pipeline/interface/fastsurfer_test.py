from pathlib import Path
from fastsurfer import Segment, Noccseg, N4BiasCorrect, TalairachAndNu, UpdateAseg, SampleSegmentationToSurfave
from nipype import Node
import os
import argparse


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


def Segment_test():
    pwd = Path.cwd()  # 当前目录,# get FastSurfer dir Absolute path
    fastsurfer_home = pwd.parent / "FastSurfer"
    fastsurfer_eval = fastsurfer_home / 'FastSurferCNN' / 'eval.py'  # inference script
    weight_dir = fastsurfer_home / 'checkpoints'  # model checkpoints dir

    network_sagittal_path = weight_dir / "Sagittal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_coronal_path = weight_dir / "Coronal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_axial_path = weight_dir / "Axial_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"

    segment_node = Node(Segment(), f'segment_node')
    segment_node.inputs.python_interpret = '/home/pbfs18/anaconda3/envs/3.8/bin/python3'
    segment_node.inputs.in_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-01/mri/orig.mgz'
    segment_node.inputs.out_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-01/mri/aparc.DKTatlas+aseg.deep.mgz'
    segment_node.inputs.eval_py = fastsurfer_eval
    segment_node.inputs.network_sagittal_path = network_sagittal_path
    segment_node.inputs.network_coronal_path = network_coronal_path
    segment_node.inputs.network_axial_path = network_axial_path

    # segment_node.run()

    segment_node.inputs.conformed_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-01/mri/conformed.mgz'
    segment_node.run()


def Noccseg_test():
    pwd = Path.cwd()  # 当前目录,# get FastSurfer dir Absolute path
    fastsurfer_home = pwd.parent / "FastSurfer"
    reduce_to_aseg_py = fastsurfer_home / 'recon_surf' / 'reduce_to_aseg.py'

    noccseg_node = Node(Noccseg(), f'noccseg_node')
    noccseg_node.inputs.python_interpret = '/home/pbfs18/anaconda3/envs/3.8/bin/python3'
    noccseg_node.inputs.reduce_to_aseg_py = reduce_to_aseg_py
    noccseg_node.inputs.in_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-1/mri/aparc.DKTatlas+aseg.deep.mgz'

    noccseg_node.inputs.mask_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-1/mri/mask.mgz'
    noccseg_node.inputs.aseg_noccseg_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-1/mri/aseg.auto_noCCseg.mgz'

    noccseg_node.run()


def N4_bias_correct_test():

    subjects_dir = Path("/mnt/ngshare/Data_Mirror/pipeline_test")
    subject_id = "sub-MSC01"
    sub_mri_dir = subjects_dir / subject_id / "mri"

    fastsurfer_home = Path("/home/anning/workspace/DeepPrep/deepprep_pipeline") / "FastSurfer"
    correct_py = fastsurfer_home / "recon_surf" / "N4_bias_correct.py"

    orig_file = sub_mri_dir / "orig.mgz"
    mask_file = sub_mri_dir / "mask.mgz"

    orig_nu_file = sub_mri_dir / "orig_nu.mgz"

    N4_bias_correct_node = Node(N4BiasCorrect(), name="N4_bias_correct_node")
    N4_bias_correct_node.inputs.python_interpret = '/home/anning/miniconda3/envs/3.8/bin/python3'
    N4_bias_correct_node.inputs.correct_py = correct_py
    N4_bias_correct_node.inputs.orig_file = orig_file
    N4_bias_correct_node.inputs.mask_file = mask_file

    N4_bias_correct_node.inputs.orig_nu_file = orig_nu_file
    N4_bias_correct_node.inputs.threads = 8

    res = N4_bias_correct_node.run()
    res = res


def talairach_and_nu_test():
    subjects_dir = Path("/mnt/ngshare/Data_Mirror/pipeline_test")
    subject_id = "sub-MSC01"
    sub_mri_dir = subjects_dir / subject_id / "mri"

    orig_nu_file = sub_mri_dir / "orig_nu.mgz"
    orig_file = sub_mri_dir / "orig.mgz"

    talairach_lta = sub_mri_dir / "transforms" / "talairach.xfm.lta"
    nu_file = sub_mri_dir / "nu.mgz"

    freesurfer_home = Path(os.environ['FREESURFER_HOME'])
    mni305 = freesurfer_home / "average" / "mni305.cor.mgz"

    talairach_and_nu_node = Node(TalairachAndNu(), name="talairach_and_nu_node")
    talairach_and_nu_node.inputs.subjects_dir = subjects_dir
    talairach_and_nu_node.inputs.subject_id = subject_id
    talairach_and_nu_node.inputs.threads = 8
    talairach_and_nu_node.inputs.mni305 = mni305
    talairach_and_nu_node.inputs.orig_nu_file = orig_nu_file
    talairach_and_nu_node.inputs.orig_file = orig_file

    talairach_and_nu_node.inputs.talairach_lta = talairach_lta
    talairach_and_nu_node.inputs.nu_file = nu_file

    talairach_and_nu_node.run()


def UpdateAseg_test():
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon'


    subjects_dir = Path('/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon')
    subject_id = 'sub-001'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)


    paint_cc_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / 'paint_cc_into_pred.py'
    updateaseg_node = Node(UpdateAseg(), name='updateaseg_node')
    updateaseg_node.inputs.subjects_dir = subjects_dir
    updateaseg_node.inputs.subject_id = subject_id
    updateaseg_node.inputs.paint_cc_file = paint_cc_file
    # updateaseg_node.inputs.python_interpret = '/home/lincong/miniconda3/envs/pytorch3.8/bin/python'
    updateaseg_node.inputs.python_interpret = '/home/youjia/anaconda3/envs/3.8/bin/python3'
    updateaseg_node.inputs.seg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'
    updateaseg_node.inputs.aseg_noCCseg_file = subject_mri_dir / 'aseg.auto_noCCseg.mgz'
    updateaseg_node.inputs.aseg_auto_file = subject_mri_dir / 'aseg.auto.mgz'
    updateaseg_node.inputs.cc_up_file = subject_mri_dir / 'transforms' / 'cc_up.lta'
    updateaseg_node.inputs.aparc_aseg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.withCC.mgz'
    updateaseg_node.run()


def SampleSegmentationToSurfave_test():
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-002"
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    SampleSegmentationToSurfave_node = Node(SampleSegmentationToSurfave(), name='SampleSegmentationToSurfave_node')
    SampleSegmentationToSurfave_node.inputs.subjects_dir = subjects_dir
    SampleSegmentationToSurfave_node.inputs.subject_id = subject_id
    SampleSegmentationToSurfave_node.inputs.python_interpret = Path('/home/youjia/anaconda3/envs/3.8/bin/python3')
    SampleSegmentationToSurfave_node.inputs.freesurfer_home = Path('/usr/local/freesurfer')
    # SampleSegmentationToSurfave_node.inputs.aparc_aseg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.withCC.mgz'
    SampleSegmentationToSurfave_node.inputs.aparc_aseg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.orig.mgz'
    # SampleSegmentationToSurfave_node.inputs.aparc_aseg_file = subject_mri_dir / 'aseg.auto.mgz'
    smooth_aparc_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / 'smooth_aparc.py'
    SampleSegmentationToSurfave_node.inputs.smooth_aparc_file = smooth_aparc_file

    lh_DKTatlaslookup_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / f'lh.DKTatlaslookup.txt'
    rh_DKTatlaslookup_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / f'rh.DKTatlaslookup.txt'
    SampleSegmentationToSurfave_node.inputs.lh_DKTatlaslookup_file = lh_DKTatlaslookup_file
    SampleSegmentationToSurfave_node.inputs.rh_DKTatlaslookup_file = rh_DKTatlaslookup_file
    SampleSegmentationToSurfave_node.inputs.lh_white_preaparc_file = subject_surf_dir / f'lh.white.preaparc'
    SampleSegmentationToSurfave_node.inputs.rh_white_preaparc_file = subject_surf_dir / f'rh.white.preaparc'
    lh_aparc_DKTatlas_mapped_prefix_file = subject_label_dir / f'lh.aparc.DKTatlas.mapped.prefix.annot'
    rh_aparc_DKTatlas_mapped_prefix_file = subject_label_dir / f'rh.aparc.DKTatlas.mapped.prefix.annot'
    SampleSegmentationToSurfave_node.inputs.lh_aparc_DKTatlas_mapped_prefix_file = lh_aparc_DKTatlas_mapped_prefix_file
    SampleSegmentationToSurfave_node.inputs.rh_aparc_DKTatlas_mapped_prefix_file = rh_aparc_DKTatlas_mapped_prefix_file
    SampleSegmentationToSurfave_node.inputs.lh_cortex_label_file = subject_label_dir / f'lh.cortex.label'
    SampleSegmentationToSurfave_node.inputs.rh_cortex_label_file = subject_label_dir / f'rh.cortex.label'
    lh_aparc_DKTatlas_mapped_file = subject_label_dir / f'lh.aparc.DKTatlas.mapped.annot'
    rh_aparc_DKTatlas_mapped_file = subject_label_dir / f'rh.aparc.DKTatlas.mapped.annot'
    SampleSegmentationToSurfave_node.inputs.lh_aparc_DKTatlas_mapped_file = lh_aparc_DKTatlas_mapped_file
    SampleSegmentationToSurfave_node.inputs.rh_aparc_DKTatlas_mapped_file = rh_aparc_DKTatlas_mapped_file
    SampleSegmentationToSurfave_node.run()


if __name__ == '__main__':
    set_envrion()

    # Segment_test()

    # N4_bias_correct_test()

    # talairach_and_nu_test()

    UpdateAseg_test()

    # SampleSegmentationToSurfave_test()
