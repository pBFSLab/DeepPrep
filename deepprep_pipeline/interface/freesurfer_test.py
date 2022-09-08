import os
from freesurfer import OrigAndRawavg
from pathlib import Path
from freesurfer import Brainmask, UpdateAseg
from nipype import Node
from run import set_envrion


def OrigAndRawavg_test():
    set_envrion(1)
    subject_dir = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon'
    os.environ['SUBJECTS_DIR'] = subject_dir  # 设置FreeSurfer的subjects_dir

    subject_id = 'OrigAndRawavg_test1'
    t1w_files = [
        f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
    ]
    origandrawavg_node = Node(OrigAndRawavg(), f'origandrawavg_node')
    origandrawavg_node.inputs.t1w_files = t1w_files
    origandrawavg_node.inputs.subject_dir = subject_dir
    origandrawavg_node.inputs.subject_id = subject_id
    origandrawavg_node.inputs.threads = 1
    origandrawavg_node.run()

    subject_id = 'OrigAndRawavg_test2'
    t1w_files = [
        f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
        f'/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz',
    ]
    origandrawavg_node = Node(OrigAndRawavg(), f'origandrawavg_node')
    origandrawavg_node.inputs.t1w_files = t1w_files
    origandrawavg_node.inputs.subject_dir = subject_dir
    origandrawavg_node.inputs.subject_id = subject_id
    origandrawavg_node.inputs.threads = 1
    origandrawavg_node.run()


def Brainmask_test():
    set_envrion()
    subject_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    brainmask_node = Node(Brainmask(), name='brainmask_node')
    brainmask_node.inputs.subject_dir = subject_dir
    brainmask_node.inputs.subject_id = subject_id
    brainmask_node.inputs.need_t1 = True
    brainmask_node.inputs.nu_file = subject_dir / subject_id / 'mri' / 'nu.mgz'
    brainmask_node.inputs.mask_file = subject_dir / subject_id / 'mri' / 'mask.mgz'
    brainmask_node.inputs.T1_file = subject_dir / subject_id / 'mri' / 'T1.mgz'
    brainmask_node.inputs.brainmask_file = subject_dir / subject_id / 'mri' / 'brainmask.mgz'
    brainmask_node.inputs.norm_file = subject_dir / subject_id / 'mri' / 'norm.mgz'
    brainmask_node.run()


def UpdateAseg_test():
    set_envrion()
    subject_dir = Path(f'/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon')
    subject_id = 'sub-MSC01'
    subject_mri_dir = subject_dir / subject_id / 'mri'
    os.environ['SUBJECTS_DIR'] = '/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/Recon'
    paint_cc_file = Path.cwd().parent / 'FastSurfer' / 'recon_surf' / 'paint_cc_into_pred.py'
    updateaseg_node = Node(UpdateAseg(), name='updateaseg_node')
    updateaseg_node.inputs.subject_dir = subject_dir
    updateaseg_node.inputs.subject_id = subject_id
    updateaseg_node.inputs.paint_cc_file = paint_cc_file
    updateaseg_node.inputs.python_interpret = '/home/lincong/miniconda3/envs/pytorch3.8/bin/python'
    updateaseg_node.inputs.seg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'
    updateaseg_node.inputs.aseg_noCCseg_file = subject_mri_dir / 'aseg.auto_noCCseg.mgz'
    updateaseg_node.inputs.aseg_auto_file = subject_mri_dir / 'aseg.auto.mgz'
    updateaseg_node.inputs.cc_up_file = subject_mri_dir / 'transforms' / 'cc_up.lta'
    updateaseg_node.inputs.aparc_aseg_file = subject_mri_dir / 'aparc.DKTatlas+aseg.deep.withCC.mgz'
    updateaseg_node.run()


if __name__ == '__main__':
    OrigAndRawavg_test()
    Brainmask_test()
