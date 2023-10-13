# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : NingAnMe <ninganme0317@gmail.com>
# @Author : Cong Lin       @Email : lincong <lincong8722@gmail.com>
# @Author : Youjia Zhang   @Email : youjia <ireneyou33@gmail.com>
# @Author : Zhenyu Sun     @Email : Kid-sunzhenyu <sun25939789@gmail.com>

from pathlib import Path
from featreg_node import FeatReg
from nipype import Node
import os
import argparse
from run import set_envrion


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


def FeatReg_test():
    pwd = Path.cwd()  # 当前目录,# get featreg dir Absolute path
    featreg_home = pwd.parent / "FeatReg"
    featreg_py = featreg_home / "featreg" / 'predict.py'  # inference script

    featreg_node = Node(FeatReg(), f'featreg_node')
    featreg_node.inputs.python_interpret = Path('/home/anning/miniconda3/envs/3.8/bin/python3')
    featreg_node.inputs.featreg_py = featreg_py

    subjects_dir = '/mnt/ngshare/Data_Mirror/pipeline_test'
    subject_id = 'sub-MSC01'

    os.environ['SUBJECTS_DIR'] = subjects_dir

    featreg_node.inputs.subjects_dir = subjects_dir
    featreg_node.inputs.subject_id = subject_id
    featreg_node.inputs.freesurfer_home = '/usr/local/freesurfer'
    featreg_node.inputs.lh_sulc = Path(subjects_dir) / subject_id / f'surf/lh.sulc'
    featreg_node.inputs.rh_sulc = Path(subjects_dir) / subject_id / f'surf/rh.sulc'
    featreg_node.inputs.lh_curv = Path(subjects_dir) / subject_id / f'surf/lh.curv'
    featreg_node.inputs.rh_curv = Path(subjects_dir) / subject_id / f'surf/rh.curv'
    featreg_node.inputs.lh_sphere = Path(subjects_dir) / subject_id / f'surf/lh.sphere'
    featreg_node.inputs.rh_sphere = Path(subjects_dir) / subject_id / f'surf/rh.sphere'

    featreg_node.run()


if __name__ == '__main__':
    set_envrion()
    FeatReg_test()
