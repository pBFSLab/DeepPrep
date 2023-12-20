from nipype.algorithms.confounds import ComputeDVARS, FramewiseDisplacement
from pathlib import Path
from confounds import FMRISummary
import os
import nibabel as nib
import numpy as np
from nipype import Node
import pandas as pd
import argparse
import shutil


def reshape_bold(bold_file):
    bold = nib.load(bold_file).get_fdata()
    n_frame = bold.shape[3]
    n_vertex = bold.shape[0] * bold.shape[1] * bold.shape[2]
    bold_surf = bold.reshape((n_vertex, n_frame), order='F')
    return bold_surf


def reshape_annot(annot_file):
    annot = nib.load(annot_file).get_fdata()

    n_vertex = annot.shape[0] * annot.shape[1] * annot.shape[2]
    annot_surf = annot.reshape(n_vertex, order='F')
    return annot_surf


class ComputeDVARSNode:
    def __call__(self, bold_file, bold_mask, base_dir):
        ComputeDVARS_node = Node(ComputeDVARS(), name='ComputeDVARS_node')
        ComputeDVARS_node.inputs.in_file = bold_file
        ComputeDVARS_node.inputs.in_mask = bold_mask
        ComputeDVARS_node.base_dir = base_dir
        result = ComputeDVARS_node.run()
        return os.path.abspath(result.outputs.out_std)


class FramewiseDisplacementNode:
    def __call__(self, mcdat, base_dir):
        FramewiseDisplacement_node = Node(FramewiseDisplacement(), f'FramewiseDisplacement_node')
        FramewiseDisplacement_node.inputs.in_file = mcdat
        FramewiseDisplacement_node.inputs.parameter_source = 'FSFAST'
        FramewiseDisplacement_node.base_dir = base_dir
        result = FramewiseDisplacement_node.run()
        return os.path.abspath(result.outputs.out_file)


class FMRISummaryNode:
    def __call__(self, in_nifti, in_segm, confounds_file, base_dir):
        confounds_list = [
            ("global_signal", None, "GS"),
            ("csf", None, "GSCSF"),
            ("white_matter", None, "GSWM"),
            ("std_dvars", None, "DVARS"),
            ("framewise_displacement", "mm", "FD"),
            ("rel_transform", None, "RHM")]
        FMRISummary_node = Node(FMRISummary(), f'FMRISummary_node')
        FMRISummary_node.inputs.in_nifti = in_nifti
        FMRISummary_node.inputs.in_segm = in_segm
        FMRISummary_node.inputs.confounds_list = confounds_list
        FMRISummary_node.inputs.confounds_file = confounds_file
        FMRISummary_node.base_dir = base_dir

        result = FMRISummary_node.run()
        return os.path.abspath(result.outputs.out_file)

def calculate_mc(mcdat):
    data = np.loadtxt(mcdat)
    columns = ['n', 'roll', 'pitch', 'yaw', 'dS', 'dL', 'dP', '.', '..', 'abs_transform']
    datafile = pd.DataFrame(data, columns=columns)
    rel_transform = abs(np.concatenate(([0], np.diff(datafile['abs_transform']))))

    return rel_transform.reshape(-1, 1)


def AverageSingnal(bold_preprocess_dir, save_svg_dir, subject_id, bold_id,
                   mc, mcdat, aseg, brainmask, brainmask_bin, wm, csf):
    # ses = re.search(r'ses-(\w+\d+)', bold_id).group(0)
    base_dir = Path(bold_preprocess_dir) / 'tmp' / bold_id
    base_dir.mkdir(exist_ok=True, parents=True)
    bold = reshape_bold(mc)
    roi_inf = {'global_signal': brainmask_bin, 'white_matter': wm, 'csf': csf}
    results = []
    for key in roi_inf.keys():
        annot_file = roi_inf[key]
        parc_annot = reshape_annot(annot_file)
        roi_mean = np.mean(bold[np.where(parc_annot == 1.0)[0]], axis=0)
        results.append(np.expand_dims(roi_mean, 1))
    columns = list(roi_inf.keys())
    bold_mask = brainmask
    compute_dvars = ComputeDVARSNode()
    std_dvars_path = compute_dvars(mc, bold_mask, base_dir)
    std_dvars = pd.read_csv(std_dvars_path, header=None).values
    std_dvars = np.insert(std_dvars, 0, np.array([np.nan]), axis=0)
    results.append(std_dvars)
    columns.append('std_dvars')
    mcdat = mcdat
    framewisedisplacement = FramewiseDisplacementNode()
    fd_path = framewisedisplacement(mcdat, base_dir)
    fd = pd.read_csv(fd_path, sep='\t', encoding='utf-8').values
    fd = np.insert(fd, 0, np.array([np.nan]), axis=0)
    results.append(fd)
    columns.append('framewise_displacement')
    rel_transform = calculate_mc(mcdat)
    results.append(rel_transform)
    columns.append('rel_transform')
    data = np.concatenate(results, axis=1).astype(np.float32)
    data_df = pd.DataFrame(data=data, columns=columns)
    csv_file = Path(bold_preprocess_dir) / 'func' / mc.name.replace('bold.nii.gz', 'desc-averagesingnal_timeseries.tsv')
    data_df.to_csv(csv_file, sep="\t")

    summary = FMRISummaryNode()
    summary_path = summary(mc_file, aseg, csv_file, base_dir)
    carpet_path = save_svg_dir / f'{bold_id}_desc-carpet_bold.svg'
    cmd = f'cp {summary_path} {carpet_path}'
    os.system(cmd)
    shutil.rmtree(base_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- AvreageSingnal"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--qc_result_path", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--mc", required=True)
    parser.add_argument("--mcdat", required=True)
    parser.add_argument("--anat_aseg", required=True)
    parser.add_argument("--anat_brainmask", required=True)
    parser.add_argument("--anat_brainmaskbin", required=True)
    parser.add_argument("--anat_wm", required=True)
    parser.add_argument("--anat_csf", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)
    # bold_preprocess_path = Path(cur_path) / Path(args.bold_preprocess_dir).name
    # mc_file = subj_func_dir / f'{args.bold_id}_space-mc_bold.nii.gz'
    mc_file = subj_func_dir / os.path.basename(args.mc)
    mcdat_file = subj_func_dir / os.path.basename(args.mcdat)
    aseg_file = subj_func_dir / os.path.basename(args.anat_aseg)
    brainmask_file = subj_func_dir / os.path.basename(args.anat_brainmask)
    brainmaskbin_file = subj_func_dir / os.path.basename(args.anat_brainmaskbin)
    wm_file = subj_func_dir / os.path.basename(args.anat_wm)
    csf_file = subj_func_dir / os.path.basename(args.anat_csf)
    svg_path = Path(cur_path) / str(args.qc_result_path) / args.subject_id / 'figures'
    svg_path.mkdir(parents=True, exist_ok=True)
    AverageSingnal(preprocess_dir, svg_path, args.subject_id, args.bold_id,
                   mc_file, mcdat_file, aseg_file, brainmask_file, brainmaskbin_file, wm_file, csf_file)

