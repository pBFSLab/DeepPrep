#! /usr/bin/env python3

# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.decomposition import PCA
from bids import BIDSLayout

from fmriprep.workflows.bold.confounds import init_bold_confs_wf
from nipype.pipeline import engine as pe
from fmriprep.interfaces import DerivativesDataSink
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
)


class eCompCorInputSpec(BaseInterfaceInputSpec):
    meta_dict = File(
        argstr="--meta_dict %s",
        mandatory=True,
        desc=(
            "The metadata from previous step (~/BOLD/*/sub-*_desc-confounds_timeseries.tsv)"
        ),
        exists=True,
    )
    boldref = File(
        argstr="--boldref %s",
        mandatory=True,
        desc=(
            "The T1w image in bold space (~/BOLD/*/sub-*_space-T1w_boldref.nii.gz)"
        ),
        exists=True,
    )
    aseg = File(
        argstr="--aseg %s",
        mandatory=True,
        desc=(
            "The aseg.mgz in Recon (~/Recon/sub-*/mri/aseg.mgz)"
        ),
        exists=True,
    )
    brainmask = File(
        argstr="--brainmask %s",
        mandatory=True,
        desc=(
            "The brainmask.mgz in Recon (~/Recon/sub-*/mri/brainmask.mgz)"
        ),
        exists=True,
    )
    bpss = File(
        argstr="--bpss %s",
        mandatory=True,
        desc=(
            "The preprocessed BOLD image (~/BOLD/sub-*/func/sub-*_space-T1w_desc-preproc_bold.nii.gz)"
        ),
        exists=True,
    )

    output_dseg = File(mandatory=True, desc="mri_convert -rl boldref aseg output_dseg")
    output_mask = File(mandatory=True, desc="mri_convert -rl boldref brainmask output_mask")


class eCompCorOutputSpec(TraitedSpec):
    output_dseg = File(exists=True, desc="mri_convert -rl boldref aseg output_dseg")
    output_mask = File(exists=True, desc="mri_convert -rl boldref brainmask output_mask")
    meta_dict = File(
        exists=True,
        desc=(
            "The UPDATED metadata from previous step, including the first 10 PCA components (~/BOLD/*/sub-*_desc-confounds_timeseries.tsv)"
        ),
    )


class eCompCor(SimpleInterface):
    input_spec = eCompCorInputSpec
    output_spec = eCompCorOutputSpec

    def _run_interface(self, runtime):

        cmd = f'mri_convert -rl {self.inputs.boldref} {self.inputs.aseg} {self.inputs.output_dseg}'
        os.system(cmd)
        assert os.path.exists(self.inputs.output_dseg)

        cmd = f'mri_convert -rl {self.inputs.boldref} {self.inputs.brainmask} {self.inputs.output_mask}'
        os.system(cmd)
        assert os.path.exists(self.inputs.output_mask)

        regressors_PCA(self.inputs.bpss, self.inputs.output_dseg, Path(self.inputs.meta_dict))

        return runtime


def get_preproc_file(bids_orig, bids_preproc, bold_orig_file, update_entities):
    layout_orig = BIDSLayout(bids_orig, validate=False)
    layout_preproc = BIDSLayout(bids_preproc, validate=False)
    info = layout_orig.parse_file_entities(bold_orig_file)

    bold_t1w_info = info.copy()
    if update_entities:
        for k,v in update_entities.items():
            bold_t1w_info[k] = v
        bold_t1w_file = layout_preproc.get(**bold_t1w_info)[0]
    else:
        bold_t1w_file = layout_preproc.get(**bold_t1w_info)[0]

    return Path(bold_t1w_file)


def bold_confounds_v2(bold_preprocess_dir, bold, bold_mask, movpar_file, rmsd_file, boldref2anat_xfm, skip_vols,
                      t1w_tpms, t1w_mask, source_file, boldref, aseg, brainmask, bpss, output_dseg, output_mask):
    bold_confounds_wf = init_bold_confs_wf(
        mem_gb=1,
        metadata={},
        regressors_all_comps=False,
        regressors_fd_th=0.5,
        regressors_dvars_th=1.5,
        name="bold_confounds_wf",
    )

    ds_confounds = pe.Node(
        DerivativesDataSink(
            base_directory=bold_preprocess_dir,
            desc='confounds',
            suffix='timeseries',
        ),
        name="ds_confounds",
        run_without_submitting=True,
    )

    ecompcor = pe.Node(
        eCompCor(),
        desc='the first 10 PCA components',
        name="ecompcor"
    )

    bold_confounds_wf.inputs.inputnode.bold = bold
    bold_confounds_wf.inputs.inputnode.bold_mask = bold_mask
    bold_confounds_wf.inputs.inputnode.movpar_file = movpar_file
    bold_confounds_wf.inputs.inputnode.rmsd_file = rmsd_file
    bold_confounds_wf.inputs.inputnode.boldref2anat_xfm = boldref2anat_xfm
    bold_confounds_wf.inputs.inputnode.skip_vols = skip_vols
    bold_confounds_wf.inputs.inputnode.t1w_tpms = t1w_tpms
    bold_confounds_wf.inputs.inputnode.t1w_mask = t1w_mask

    ds_confounds.inputs.source_file = source_file

    ecompcor.inputs.boldref = boldref
    ecompcor.inputs.aseg = aseg
    ecompcor.inputs.brainmask = brainmask
    ecompcor.inputs.bpss = bpss
    ecompcor.inputs.output_dseg = output_dseg
    ecompcor.inputs.output_mask = output_mask

    # Fill-in datasinks of reportlets seen so far
    for node in bold_confounds_wf.list_node_names():
        if node.split(".")[-1].startswith("ds_report"):
            bold_confounds_wf.get_node(
                node).inputs.source_file = source_file

    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    workflow = Workflow(name="bold_confounds_v2_wf")
    workflow.connect([
        (bold_confounds_wf, ds_confounds, [
            ('outputnode.confounds_file', 'in_file'),
        ]),
        (ds_confounds, ecompcor, [
            ('out_file', 'meta_dict'),
        ]),
    ])
    workflow.run()


def regressor_PCA_singlebold(pca_data, n):
    pca = PCA(n_components=n, random_state=False)
    pca_regressor = pca.fit_transform(pca_data.T)
    return pca_regressor


def regressors_PCA(bpss_path, maskpath, outpath):
    '''
    Generate PCA regressor from outer points of brain.
        bpss_path - path. Path of bold after bpass process.
        maskpath - Path to file containing mask.
        outpath  - Path to file to place the output.
    '''
    # PCA parameter.
    n = 10

    # Open mask.
    mask_img = nib.load(maskpath)
    mask = mask_img.get_fdata().swapaxes(0, 1)
    mask = mask.flatten(order='F') == 0
    nvox = float(mask.sum())
    assert nvox > 0, 'Null mask found in %s' % maskpath

    img = nib.load(bpss_path)
    data = img.get_fdata().swapaxes(0, 1)
    vol_data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]), order='F')
    pca_data = vol_data[mask]
    pca_regressor = np.asarray(regressor_PCA_singlebold(pca_data, n))
    df = pd.read_csv(outpath, sep='\t')
    col = [f'comp{str(i+1)}' for i in range(n)]
    df_pca = pd.DataFrame(pca_regressor, columns=col)
    df = pd.concat((df, df_pca), axis=1)
    df.to_csv(outpath, sep='\t', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="DeepPrep: Confounds v2 workflow"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--bold_file", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--t1w_tpms_CSF", required=True)
    parser.add_argument("--t1w_tpms_GM", required=True)
    parser.add_argument("--t1w_tpms_WM", required=True)
    parser.add_argument("--mask_nii", required=True)
    parser.add_argument("--aseg", required=True)
    parser.add_argument("--brainmask", required=True)
    args = parser.parse_args()
    """
    inputs:
    --bids_dir: ~/input/dataset
    --bold_preprocess_dir: ~/output/BOLD
    --work_dir: ~/output/WorkDir
    --bold_id: sub-001_ses-01_task-rest_run-01
    --bold_file: ~/WorkDir/sub-001_ses-01_task-rest_run-01
    --subject_id: sub-001
    --t1w_tpms_CSF: ~/output/BOLD/sub-001/anat/sub-001-CSF_probseg.nii.gz
    --t1w_tpms_GM: ~/output/BOLD/sub-001/anat/sub-001-GM_probseg.nii.gz
    --t1w_tpms_WM: ~/output/BOLD/sub-001/anat/sub-001-WM_probseg.nii.gz
    --mask_nii: ~/output/BOLD/sub-001/anat/sub-001_desc-brain_mask.nii.gz
    --aseg: ~/output/Recon/sub-001/mri/aseg.mgz
    --brainmask: ~/output/Recon/sub-001/mri/brainmask.mgz
    
    outputs:
    meta_dict: ~/output/BOLD/sub-001/ses-01/func/sub-001-01_task-rest_run-01_desc-confounds_timeseries.tsv
    """

    # The required input confound files were generated in <process:bold_pre_process>, and copied to <confounds_dir_path>.
    # If there's any missing files under <confounds_dir_path>, please go to <process:bold_pre_process> and double check if its original path exists.
    confounds_dir_path = Path(args.work_dir) / args.subject_id / 'confounds'
    bold_file = [args.bold_file]
    boldresampled = confounds_dir_path / f'{args.bold_id}_boldresampled.nii.gz'
    bold_mask = confounds_dir_path / f'{args.bold_id}_bold_average_corrected_brain_mask_maths.nii.gz'
    movpar_file = confounds_dir_path / 'motion_params.txt'
    rmsd_file = confounds_dir_path / f'{args.bold_id}_bold_mcf.nii.gz_rel.rms'
    skip_vols = 2
    t1w_tpms = [args.t1w_tpms_CSF, args.t1w_tpms_GM, args.t1w_tpms_WM]
    t1w_mask = args.mask_nii

    with open(bold_file[0], 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    source_file = data[1]

    update_entities = {'desc': 'coreg', 'suffix': 'xfm', 'extension': '.txt'}
    boldref2anat_xfm = get_preproc_file(args.bids_dir, args.bold_preprocess_dir, source_file, update_entities)

    # eCompCor inputs
    aseg = args.aseg
    update_entities = {'space': 'T1w', 'suffix': 'boldref', 'extension': '.nii.gz'}
    boldref = get_preproc_file(args.bids_dir, args.bold_preprocess_dir, source_file, update_entities)
    brainmask = args.brainmask
    update_entities = {'space': 'T1w', 'desc': 'preproc', 'suffix': 'bold', 'extension': '.nii.gz'}
    bpss = get_preproc_file(args.bids_dir, args.bold_preprocess_dir, source_file, update_entities)
    output_dseg = confounds_dir_path / 'dseg.nii.gz'
    output_mask = confounds_dir_path / 'desc-brain_mask.nii.gz'

    confounds_tsv = boldref2anat_xfm.parent / f'{args.bold_id}_desc-confounds_timeseries.tsv'
    bold_confounds_v2(args.bold_preprocess_dir, boldresampled, bold_mask, movpar_file, rmsd_file, str(boldref2anat_xfm), skip_vols,
                      t1w_tpms, t1w_mask, source_file, boldref, aseg, brainmask, bpss, output_dseg, output_mask)
    assert confounds_tsv.exists()




