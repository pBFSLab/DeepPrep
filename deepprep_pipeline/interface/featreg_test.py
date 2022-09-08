from pathlib import Path
from featreg_interface import FeatReg
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


def FeatReg_test():
    pwd = Path.cwd()  # 当前目录,# get featreg dir Absolute path
    featreg_home = pwd.parent / "FeatReg"
    featreg_py = featreg_home / "featreg" / 'predict.py'  # inference script

    for hemi in ['lh', 'rh']:

        featreg_node = Node(FeatReg(), f'featreg_node')
        featreg_node.inputs.python_interpret = '/home/anning/miniconda3/envs/3.8/bin/python3'
        featreg_node.inputs.featreg_py = featreg_py

        subjects_dir = '/mnt/ngshare/Data_Mirror/pipeline_test'
        subject_id = 'sub-MSC01'

        os.environ['SUBJECTS_DIR'] = subjects_dir

        featreg_node.inputs.subjects_dir = subjects_dir
        featreg_node.inputs.subject_id = subject_id
        featreg_node.inputs.freesurfer_home = '/usr/local/freesurfer'
        featreg_node.inputs.hemisphere = hemi
        featreg_node.inputs.sulc_file = Path(subjects_dir) / subject_id / f'surf/{hemi}.sulc'
        featreg_node.inputs.curv_file = Path(subjects_dir) / subject_id / f'surf/{hemi}.curv'
        featreg_node.inputs.sphere_file = Path(subjects_dir) / subject_id / f'surf/{hemi}.sphere'

        featreg_node.run()


if __name__ == '__main__':
    set_envrion()
    FeatReg_test()
