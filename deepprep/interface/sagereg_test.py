from pathlib import Path
from sagereg_interface import SageReg
from nipype import Node
import os
import argparse
from run import set_envrion
from config import settings

def SageReg_test():
    pwd = Path.cwd()  # 当前目录,# get featreg dir Absolute path
    sagereg_home = pwd.parent / "SageReg"
    sagereg_py = sagereg_home / 'predict.py'  # inference script

    sagereg_node = Node(SageReg(), f'sagereg_node')
    sagereg_node.inputs.python_interpret = Path('/home/lincong/anaconda3/envs/3.8/bin/python3')
    sagereg_node.inputs.sagereg_py = sagereg_py

    subjects_dir = '/mnt/ngshare/SurfReg/NAMIC_tmp'
    subject_id = 'sub01'

    os.environ['SUBJECTS_DIR'] = subjects_dir

    sagereg_node.inputs.model_path = Path(settings.SAGEREG_MODEL_PATH)
    sagereg_node.inputs.subjects_dir = subjects_dir
    sagereg_node.inputs.subject_id = subject_id
    sagereg_node.inputs.freesurfer_home = '/usr/local/freesurfer'
    sagereg_node.inputs.lh_sulc = Path(subjects_dir) / subject_id / f'surf/lh.sulc'
    sagereg_node.inputs.rh_sulc = Path(subjects_dir) / subject_id / f'surf/rh.sulc'
    sagereg_node.inputs.lh_curv = Path(subjects_dir) / subject_id / f'surf/lh.curv'
    sagereg_node.inputs.rh_curv = Path(subjects_dir) / subject_id / f'surf/rh.curv'
    sagereg_node.inputs.lh_sphere = Path(subjects_dir) / subject_id / f'surf/lh.sphere'
    sagereg_node.inputs.rh_sphere = Path(subjects_dir) / subject_id / f'surf/rh.sphere'

    sagereg_node.run()


if __name__ == '__main__':
    set_envrion()

    SageReg_test()
