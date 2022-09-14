from pathlib import Path
from nipype import Node, Workflow
from interface.freesurfer import OrigAndRawavg
from interface.fastsurfer import Segment, Noccseg, N4BiasCorrect, TalairachAndNu


def init_single_structure_wf(t1w_files: list, subjects_dir: Path, subject_id: str,
                             python_interpret: Path,
                             fastsurfer_home: Path,
                             freesurfer_home: Path):
    single_structure_wf = Workflow(name=f'single_structure_{subject_id}_wf')

    # orig_and_rawavg_node
    orig_and_rawavg_node = Node(OrigAndRawavg(), name='orig_and_rawavg_node')

    orig_and_rawavg_node.inputs.t1w_files = t1w_files
    orig_and_rawavg_node.inputs.subjects_dir = subjects_dir
    orig_and_rawavg_node.inputs.subject_id = subject_id
    orig_and_rawavg_node.inputs.threads = 8

    # segment_node
    fastsurfer_eval = fastsurfer_home / 'FastSurferCNN' / 'eval.py'  # inference script
    weight_dir = fastsurfer_home / 'checkpoints'  # model checkpoints dir
    network_sagittal_path = weight_dir / "Sagittal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_coronal_path = weight_dir / "Coronal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_axial_path = weight_dir / "Axial_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"

    segment_node = Node(Segment(), f'segment_node')
    segment_node.inputs.python_interpret = python_interpret
    segment_node.inputs.eval_py = fastsurfer_eval
    segment_node.inputs.network_sagittal_path = network_sagittal_path
    segment_node.inputs.network_coronal_path = network_coronal_path
    segment_node.inputs.network_axial_path = network_axial_path

    segment_node.inputs.out_file = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'
    segment_node.inputs.conformed_file = subjects_dir / subject_id / 'mri' / 'conformed.mgz'

    # auto_noccseg_node
    fastsurfer_reduce_to_aseg_py = fastsurfer_home / 'recon_surf' / 'reduce_to_aseg.py'  # inference script

    auto_noccseg_node = Node(Noccseg(), name='auto_noccseg_node')
    auto_noccseg_node.inputs.python_interpret = python_interpret
    auto_noccseg_node.inputs.reduce_to_aseg_py = fastsurfer_reduce_to_aseg_py
    auto_noccseg_node.inputs.in_file = subjects_dir / subject_id / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'

    auto_noccseg_node.inputs.mask_file = subjects_dir / subject_id / 'mri' / 'mask.mgz'
    auto_noccseg_node.inputs.aseg_noccseg_file = subjects_dir / subject_id / 'mri' / 'aseg.auto_noCCseg.mgz'

    # N4_bias_correct_node
    correct_py = fastsurfer_home / "recon_surf" / "N4_bias_correct.py"

    N4_bias_correct_node = Node(N4BiasCorrect(), name="N4_bias_correct_node")
    N4_bias_correct_node.inputs.threads = 8
    N4_bias_correct_node.inputs.python_interpret = python_interpret
    N4_bias_correct_node.inputs.correct_py = correct_py
    N4_bias_correct_node.inputs.orig_nu_file = subjects_dir / subject_id / "mri" / "orig_nu.mgz"

    # TalairachAndNu
    talairach_and_nu_node = Node(TalairachAndNu(), name="talairach_and_nu_node")
    talairach_and_nu_node.inputs.subjects_dir = subjects_dir
    talairach_and_nu_node.inputs.subject_id = subject_id
    talairach_and_nu_node.inputs.threads = 8

    talairach_and_nu_node.inputs.mni305 = freesurfer_home / "average" / "mni305.cor.mgz"  # atlas

    talairach_and_nu_node.inputs.talairach_lta = subjects_dir / subject_id / 'mri' / 'transforms' / 'talairach.xfm.lta'
    talairach_and_nu_node.inputs.nu_file = subjects_dir / subject_id / 'mri' / 'nu.mgz'


    # create workflow
    single_structure_wf.connect([(orig_and_rawavg_node, segment_node, [("orig_file", "in_file"),
                                                                       ]),
                                 (segment_node, auto_noccseg_node, [("aseg_deep_file", "in_file"),
                                                                    ]),
                                 (orig_and_rawavg_node, N4_bias_correct_node, [("orig_file", "orig_file"),
                                                                               ]),
                                 (auto_noccseg_node, N4_bias_correct_node, [("mask_file", "mask_file"),
                                                                            ]),
                                 (orig_and_rawavg_node, talairach_and_nu_node, [("orig_file", "orig_file"),
                                                                                ]),
                                 (N4_bias_correct_node, talairach_and_nu_node, [("orig_nu_file", "orig_nu_file"),
                                                                                ]),
                                 ])

    return single_structure_wf


def pipeline():
    t1w_files = [
        f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
    ]
    pwd = Path.cwd()
    python_interpret = Path('/home/anning/miniconda3/envs/3.8/bin/python3')
    fastsurfer_home = pwd / "FastSurfer"
    freesurfer_home = Path('/usr/local/freesurfer')

    subjects_dir = Path('/mnt/ngshare/Data_Mirror/pipeline_test')
    subject_id = 'sub-MSC01'

    os.environ['SUBJECTS_DIR'] = str(subjects_dir)

    wf = init_single_structure_wf(t1w_files, subjects_dir, subject_id, python_interpret, fastsurfer_home,
                                  freesurfer_home)
    wf.base_dir = f'/mnt/ngshare/Data_Mirror/pipeline_test'
    wf.write_graph(graph2use='flat', simple_form=False)
    wf.run()


if __name__ == '__main__':
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
        # ANFI
        os.environ['PATH'] = '/home/anning/abin:' + os.environ['PATH']
        # ANTs
        os.environ['PATH'] = '/usr/local/ANTs/bin:' + os.environ['PATH']
        # Convert3D
        os.environ['PATH'] = '/usr/local/c3d-1.1.0-Linux-x86_64/bin:' + os.environ['PATH']


    set_envrion()
    pipeline()
