#! /usr/bin/env python3
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
from bold_mkbrainmask import anat2bold_t1w


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


def AverageSingnal(bold_averagesingnal_dir, save_svg_dir, bold_id,
                   bold_space_t1w, abs_dat_file, rel_dat_file, aseg, brainmask, brainmask_bin, wm, csf):

    bold = reshape_bold(bold_space_t1w)
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
    std_dvars_path = compute_dvars(bold_space_t1w, bold_mask, bold_averagesingnal_dir)
    std_dvars = pd.read_csv(std_dvars_path, header=None).values
    std_dvars = np.insert(std_dvars, 0, np.array([np.nan]), axis=0)
    results.append(std_dvars)
    columns.append('std_dvars')
    fd = np.loadtxt(abs_dat_file).reshape(-1, 1)
    results.append(fd)
    columns.append('framewise_displacement')
    rel_transform = np.loadtxt(rel_dat_file)
    rel_transform = np.insert(rel_transform, 0, 0).reshape(-1, 1)
    results.append(rel_transform)
    columns.append('rel_transform')
    data = np.concatenate(results, axis=1).astype(np.float32)
    data_df = pd.DataFrame(data=data, columns=columns)
    csv_file = bold_averagesingnal_dir / Path(bold_space_t1w).name.replace('bold.nii.gz', 'desc-averagesingnal_timeseries.tsv')
    data_df.to_csv(csv_file, sep="\t")

    summary = FMRISummaryNode()
    summary_path = summary(bold_space_t1w, aseg, csv_file, bold_averagesingnal_dir)
    carpet_path = save_svg_dir / f'{bold_id}_desc-carpet_bold.svg'
    cmd = f'cp {summary_path} {carpet_path}'
    os.system(cmd)


def get_space_t1w_bold(bids_orig, bids_preproc, bold_orig_file):
    from bids import BIDSLayout
    layout_orig = BIDSLayout(bids_orig, validate=False)
    layout_preproc = BIDSLayout(bids_preproc, validate=False)
    info = layout_orig.parse_file_entities(bold_orig_file)

    boldref_t1w_info = info.copy()
    boldref_t1w_info['space'] = 'T1w'
    boldref_t1w_info['suffix'] = 'boldref'
    boldref_t1w_file = layout_preproc.get(**boldref_t1w_info)[0]

    bold_t1w_info = info.copy()
    bold_t1w_info['space'] = 'T1w'
    bold_t1w_info['desc'] = 'preproc'
    bold_t1w_info['suffix'] = 'bold'
    bold_t1w_file = layout_preproc.get(**bold_t1w_info)[0]

    fd_t1w_info = info.copy()
    fd_t1w_info['suffix'] = 'mcf'
    fd_t1w_info['extension'] = 'nii.gz_abs.rms'
    fd_t1w_file = layout_preproc.get(**fd_t1w_info)[0]

    rel_t1w_info = info.copy()
    rel_t1w_info['suffix'] = 'mcf'
    rel_t1w_info['extension'] = 'nii.gz_rel.rms'
    rel_t1w_file = layout_preproc.get(**rel_t1w_info)[0]

    return bold_t1w_file.path, boldref_t1w_file.path, fd_t1w_file.path, rel_t1w_file.path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- AvreageSingnal"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--qc_result_path", required=True)
    parser.add_argument("--tmp_workdir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--bold_file", required=True)
    parser.add_argument("--aseg_mgz", required=True)
    parser.add_argument("--brainmask_mgz", required=True)
    args = parser.parse_args()

    with open(args.bold_file, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_orig_file = data[1]

    bold_space_t1w_file, boldref_space_t1w_file, abs_dat_file, rel_dat_file = get_space_t1w_bold(args.bids_dir,
                                                                                     args.bold_preprocess_dir,
                                                                                     bold_orig_file)

    bold_averagesingnal_dir = Path(args.tmp_workdir) / args.bold_id / 'bold_averagesingnal'
    bold_averagesingnal_dir.mkdir(parents=True, exist_ok=True)

    aseg = bold_averagesingnal_dir / 'dseg.nii.gz'
    wm = bold_averagesingnal_dir / 'label-WM_probseg.nii.gz'
    vent = bold_averagesingnal_dir / 'label-ventricles_probseg.nii.gz'
    csf = bold_averagesingnal_dir / 'label-CSF_probseg.nii.gz'
    # project brainmask.mgz to mc
    mask = bold_averagesingnal_dir / 'desc-brain_mask.nii.gz'
    binmask = bold_averagesingnal_dir / 'desc-brain_maskbin.nii.gz'

    anat2bold_t1w(args.aseg_mgz, args.brainmask_mgz, boldref_space_t1w_file,
                  str(aseg), str(wm), str(vent), str(csf), str(mask), str(binmask))

    svg_path = Path(args.qc_result_path) / args.subject_id / 'figures'
    svg_path.mkdir(parents=True, exist_ok=True)
    AverageSingnal(bold_averagesingnal_dir, svg_path, args.bold_id,
                   bold_space_t1w_file, abs_dat_file, rel_dat_file, aseg, mask, binmask, wm, csf)
