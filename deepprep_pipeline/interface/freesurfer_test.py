from pathlib import Path
from freesurfer import Brainmask
from nipype import Node
from .cmd import set_envrion


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
    Brainmask_test()
