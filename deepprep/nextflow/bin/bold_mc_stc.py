#! /usr/bin/env python3
from pathlib import Path
import bids
import argparse
import os
import shutil

from fmriprep import config
from fmriprep.workflows.bold import init_bold_stc_wf, init_bold_hmc_wf
from fmriprep.workflows.bold.base import _create_mem_gb
from niworkflows.utils.connections import listify, pop_file

def hmc(bold_file, raw_ref_image):
    mem_gb = {"filesize": 1, "resampled": 1, "largemem": 1}
    omp_nthreads = config.nipype.omp_nthreads
    if os.path.isfile(bold_file):
        bold_tlen, mem_gb = _create_mem_gb(bold_file)

    # HMC on the BOLD
    bold_hmc_wf = init_bold_hmc_wf(name="bold_hmc_wf", mem_gb=mem_gb["filesize"], omp_nthreads=omp_nthreads)
    bold_hmc_wf.inputs.inputnode.bold_file = bold_file
    bold_hmc_wf.inputs.inputnode.raw_ref_image = raw_ref_image
    bold_hmc_wf.run()

def stc(metadata, bold_file):
    bold_stc_wf = init_bold_stc_wf(name="bold_stc_wf", metadata=metadata)
    bold_stc_wf.inputs.inputnode.bold_file = bold_file
    bold_stc_wf.run()

def cmd(bids_dir,bold_file, raw_ref_image):
    # run mc
    hmc(bold_file, raw_ref_image)

    # run stc if metadata is provided
    layout = bids.BIDSLayout(str(bids_dir), derivatives=True)
    all_metadata = [layout.get_metadata(fname) for fname in listify(bold_file)]
    metadata = all_metadata[0]
    run_stc = bool(metadata.get("SliceTiming")) and "slicetiming" not in config.workflow.ignore
    if run_stc:
        stc(metadata, bold_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- STC"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--reorient", required=True)  # _space-reorient_bold.nii.gz
    args = parser.parse_args()

    cur_path = os.getcwd()

    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)
    subj_tmp_dir = Path(preprocess_dir) / 'tmp'
    subj_tmp_dir.mkdir(parents=True, exist_ok=True)

    skip_reorient_file = subj_func_dir / os.path.basename(args.reorient)

    run = '001'
    cmd(subj_func_dir, subj_tmp_dir, skip_reorient_file, run, args.subject_id, args.bold_id)
