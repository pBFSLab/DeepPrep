from pathlib import Path
from fastsurfer import Segment
from nipype import Node


def Segment_test():
    pwd = Path.cwd()  # 当前目录,# get FastSurfer dir Absolute path
    fastsurfer_home = pwd.parent / "FastSurfer"
    fastsurfer_eval = fastsurfer_home / 'FastSurferCNN' / 'eval.py'  # inference script
    weight_dir = fastsurfer_home / 'checkpoints'  # model checkpoints dir

    network_sagittal_path = weight_dir / "Sagittal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_coronal_path = weight_dir / "Coronal_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"
    network_axial_path = weight_dir / "Axial_Weights_FastSurferCNN" / "ckpts" / "Epoch_30_training_state.pkl"

    segment_node = Node(Segment(), f'segment_node')
    segment_node.inputs.python_interpret = '/home/anning/miniconda3/envs/3.8/bin/python3'
    segment_node.inputs.in_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-01/mri/orig.mgz'
    segment_node.inputs.out_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-01/mri/aparc.DKTatlas+aseg.deep.mgz'
    segment_node.inputs.eval_py = fastsurfer_eval
    segment_node.inputs.network_sagittal_path = network_sagittal_path
    segment_node.inputs.network_coronal_path = network_coronal_path
    segment_node.inputs.network_axial_path = network_axial_path

    segment_node.run()

    segment_node.inputs.conformed_file = '/mnt/ngshare/Data_Mirror/SDCFlows_test/MSC1/derivatives/deepprep/Recon/sub-MSC01_ses-struct01_run-01/mri/conformed.mgz'
    segment_node.run()


if __name__ == '__main__':

    Segment_test()
