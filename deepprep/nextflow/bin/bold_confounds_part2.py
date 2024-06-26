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
from bids import BIDSLayout

from fmriprep.workflows.bold.confounds import init_bold_confs_wf
from nipype.pipeline import engine as pe
from fmriprep.interfaces import DerivativesDataSink



def get_preproc_file(subject_id, bids_preproc, bold_orig_file, update_entities):
    assert subject_id.startswith('sub-')
    layout_preproc = BIDSLayout(str(os.path.join(bids_preproc, subject_id)),
                                config=['bids', 'derivatives'], validate=False)
    info = layout_preproc.parse_file_entities(bold_orig_file)

    bold_t1w_info = info.copy()
    if update_entities:
        for k,v in update_entities.items():
            bold_t1w_info[k] = v
        bold_t1w_file = layout_preproc.get(**bold_t1w_info)[0]
    else:
        bold_t1w_file = layout_preproc.get(**bold_t1w_info)[0]

    return Path(bold_t1w_file)


def bold_confounds_v2(bold_preprocess_dir, bold, bold_mask, movpar_file, rmsd_file, boldref2anat_xfm, skip_vols,
                      t1w_tpms, t1w_mask, source_file):
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

    bold_confounds_wf.inputs.inputnode.bold = bold
    bold_confounds_wf.inputs.inputnode.bold_mask = bold_mask
    bold_confounds_wf.inputs.inputnode.movpar_file = movpar_file
    bold_confounds_wf.inputs.inputnode.rmsd_file = rmsd_file
    bold_confounds_wf.inputs.inputnode.boldref2anat_xfm = boldref2anat_xfm
    bold_confounds_wf.inputs.inputnode.skip_vols = skip_vols
    bold_confounds_wf.inputs.inputnode.t1w_tpms = t1w_tpms
    bold_confounds_wf.inputs.inputnode.t1w_mask = t1w_mask

    ds_confounds.inputs.source_file = source_file

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
            ('outputnode.confounds_metadata', 'meta_dict'),
        ]),
    ])
    workflow.run()


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
    
    outputs:
    meta_dict: ~/output/BOLD/sub-001/ses-01/func/sub-001-01_task-rest_run-01_desc-confounds_timeseries.json
    meta_dict: ~/output/BOLD/sub-001/ses-01/func/sub-001-01_task-rest_run-01_desc-confounds_timeseries.tsv
    """

    # The required input confound files were generated in <process:bold_pre_process>, and linked to <confounds_dir>.
    # If there's any missing file under <confounds_dir>, please go to <process:bold_pre_process> and double check if its original path exists.
    confounds_dir = Path(args.work_dir) / 'confounds' / args.subject_id / args.bold_id
    bold_file = [args.bold_file]
    boldresampled = confounds_dir / f'{args.bold_id}_boldresampled.nii.gz'
    bold_mask = confounds_dir / f'{args.bold_id}_bold_average_corrected_brain_mask_maths.nii.gz'
    movpar_file = confounds_dir / f'{args.bold_id}_motion_params.txt'
    rmsd_file = confounds_dir / f'{args.bold_id}_bold_mcf.nii.gz_rel.rms'
    skip_vols = 2
    t1w_tpms = [args.t1w_tpms_GM, args.t1w_tpms_WM, args.t1w_tpms_CSF]
    t1w_mask = args.mask_nii

    with open(bold_file[0], 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    source_file = data[1]

    update_entities = {'desc': 'coreg', 'suffix': 'xfm', 'extension': '.txt'}
    boldref2anat_xfm = get_preproc_file(args.subject_id, args.bold_preprocess_dir, source_file, update_entities)

    confounds_tsv = boldref2anat_xfm.parent / f'{args.bold_id}_desc-confounds_timeseries.tsv'
    bold_confounds_v2(args.bold_preprocess_dir, boldresampled, bold_mask, movpar_file, rmsd_file, str(boldref2anat_xfm), skip_vols,
                      t1w_tpms, t1w_mask, source_file)
    assert confounds_tsv.exists()

