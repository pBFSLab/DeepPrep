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



if __name__ == '__main__':
    OrigAndRawavg_test()
    Brainmask_test()
