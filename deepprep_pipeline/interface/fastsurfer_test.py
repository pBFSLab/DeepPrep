from pathlib import Path
from fastsurfer import Segment, Noccseg, N4BiasCorrect, TalairachAndNu, UpdateAseg
from nipype import Node
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bd', required=True, help='directory of bids type')
    parser.add_argument('--fsd', default=os.environ.get('FREESURFER_HOME'),
                        help='Output directory $FREESURFER_HOME (pass via environment or here)')
    parser.add_argument('--respective', default='off',
                        help='if on, while processing T1w file respectively')
    parser.add_argument('--rewrite', default='on',
                        help='set off, while not preprocess if subject recon path exist')
    parser.add_argument('--python', default='python3',
                        help='which python version to use')

    args = parser.parse_args()
    args_dict = vars(args)

    if args.fsd is None:
        args_dict['fsd'] = '/usr/local/freesurfer'
    args_dict['respective'] = True if args.respective == 'on' else False
    args_dict['rewrite'] = True if args.rewrite == 'on' else False

    return argparse.Namespace(**args_dict)


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
    fastsurfer_reduce_to_aseg = fastsurfer_home / 'recon_surf' / 'reduce_to_aseg.py'


    noccseg_node = Node(Noccseg(), f'noccseg_node')
    noccseg_node.inputs.python_interpret = '/home/pbfs18/anaconda3/envs/3.8/bin/python3'
    noccseg_node.inputs.in_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-1/mri/aparc.DKTatlas+aseg.deep.mgz'
    noccseg_node.inputs.out_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-1/mri/aseg.auto_noCCseg.mgz'
    noccseg_node.inputs.reduce_to_aseg_py = fastsurfer_reduce_to_aseg
    noccseg_node.inputs.mask_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-1/mri/mask.mgz'

    noccseg_node.run()

def N4_bias_correct_test():
    args = parse_args()

    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-001"
    sub_mri_dir = subjects_dir / subject_id / "mri"

    fastsurfer_home = Path("/home/youjia/workspace/DeepPrep/deepprep_pipeline") / "FastSurfer"
    orig_file = sub_mri_dir / "orig.mgz"
    orig_nu_file = sub_mri_dir / "orig_nu.mgz"
    mask_file = sub_mri_dir / "mask.mgz"

    N4_bias_correct_node = Node(N4BiasCorrect(fastsurfer_home), name="N4_bias_correct_node")
    N4_bias_correct_node.inputs.python = args.python
    N4_bias_correct_node.inputs.orig_file = orig_file
    N4_bias_correct_node.inputs.orig_nu_file = orig_nu_file
    N4_bias_correct_node.inputs.mask_file = mask_file
    N4_bias_correct_node.inputs.threads = 30

    N4_bias_correct_node.run()


def talairach_and_nu_test():
    subjects_dir = Path("/mnt/ngshare/DeepPrep_flowtest/V001/derivatives/deepprep/Recon")
    subject_id = "sub-001"
    sub_mri_dir = subjects_dir / subject_id / "mri"
    orig_nu_file = sub_mri_dir / "orig_nu.mgz"
    nu_file = sub_mri_dir / "nu.mgz"
    # talairach_auto_xfm
    talairach_xfm = sub_mri_dir / "transforms" / "talairach.xfm"
    orig_file = sub_mri_dir / "orig.mgz"
    freesurfer_home = Path(os.environ['FREESURFER_HOME'])
    # freesurfer_home = Path("/usr/local/freesurfer")
    talairach_xfm_lta = sub_mri_dir / "transforms" / "talairach.xfm.lta"
    talairach_lta = sub_mri_dir / "transforms" / "talairach.lta"

    talairach_and_nu_node = Node(TalairachAndNu(freesurfer_home), name="talairach_and_nu_node")
    talairach_and_nu_node.inputs.sub_mri_dir = sub_mri_dir
    talairach_and_nu_node.inputs.threads = 30
    talairach_and_nu_node.inputs.orig_nu_file = orig_nu_file
    talairach_and_nu_node.inputs.orig_file = orig_file
    talairach_and_nu_node.inputs.talairach_xfm_lta = talairach_xfm_lta
    talairach_and_nu_node.inputs.talairach_xfm = str(talairach_xfm)
    talairach_and_nu_node.inputs.talairach_lta = talairach_lta
    talairach_and_nu_node.inputs.nu_file = nu_file

    talairach_and_nu_node.run()


def UpdateAseg_test():
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon'
    paint_cc_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / 'paint_cc_into_pred.py'
    updateaseg_node = Node(UpdateAseg(), name='updateaseg_node')
    updateaseg_node.inputs.subjects_dir = subjects_dir
    updateaseg_node.inputs.subject_id = subject_id
    updateaseg_node.inputs.paint_cc_file = paint_cc_file
    updateaseg_node.inputs.python_interpret = '/home/lincong/miniconda3/envs/pytorch3.8/bin/python'
    updateaseg_node.inputs.seg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'
    updateaseg_node.inputs.aseg_noCCseg_file = subject_mri_dir / 'aseg.auto_noCCseg.mgz'
    updateaseg_node.inputs.aseg_auto_file = subject_mri_dir / 'aseg.auto.mgz'
    updateaseg_node.inputs.cc_up_file = subject_mri_dir / 'transforms' / 'cc_up.lta'
    updateaseg_node.inputs.aparc_aseg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.withCC.mgz'
    updateaseg_node.run()


def SampleSegmentationToSurfave():
    subjects_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_mri_dir = subjects_dir / subject_id / 'mri'
    subject_surf_dir = subjects_dir / subject_id / 'surf'
    subject_label_dir = subjects_dir / subject_id / 'label'
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon'
    for hemi in ['lh', 'rh']:
        SampleSegmentationToSurfave_node = Node(SampleSegmentationToSurfave(), name='SampleSegmentationToSurfave_node')
        SampleSegmentationToSurfave_node.inputs.subjects_dir = subjects_dir
        SampleSegmentationToSurfave_node.inputs.subject_id = subject_id
        SampleSegmentationToSurfave_node.inputs.python_interpret = '/home/lincong/miniconda3/envs/pytorch3.8/bin/python'
        SampleSegmentationToSurfave_node.inputs.freesufer_home = os.environ['FREESURFER_HOME']
        SampleSegmentationToSurfave_node.inputs.aparc_aseg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.withCC.mgz'
        smooth_aparc_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / 'smooth_aparc.py'
        SampleSegmentationToSurfave_node.inputs.smooth_aparc_file = smooth_aparc_file
        SampleSegmentationToSurfave_node.inputs.hemi = hemi
        hemi_DKTatlaslookup_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / f'{hemi}.DKTatlaslookup.txt'
        SampleSegmentationToSurfave_node.inputs.hemi_DKTatlaslookup_file = hemi_DKTatlaslookup_file
        SampleSegmentationToSurfave_node.inputs.hemi_white_preaparc_file = subject_surf_dir / f'{hemi}.white.preaparc'
        hemi_DKTatlas_mapped_prefix = subject_label_dir / f'{hemi}.aparc.DKTatlas.mapped.prefix.annot'
        SampleSegmentationToSurfave_node.inputs.hemi_aparc_DKTatlas_mapped_prefix_file = hemi_DKTatlas_mapped_prefix
        SampleSegmentationToSurfave_node.inputs.hemi_cortex_label_file = subject_label_dir / f'{hemi}.cortex.label'
        hemi_DKTatlas_mapped = subject_label_dir / f'{hemi}.aparc.DKTatlas.mapped.annot'
        SampleSegmentationToSurfave_node.inputs.hemi_aparc_DKTatlas_mapped_file = hemi_DKTatlas_mapped
        SampleSegmentationToSurfave_node.run()


if __name__ == '__main__':
    set_envrion()

    # Segment_test()

    # N4_bias_correct_test()

    talairach_and_nu_test()
