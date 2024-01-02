#! /usr/bin/env python3
from pathlib import Path
import bids
import argparse
import os
import shutil

from fmriprep import config
from fmriprep.workflows.bold import init_bold_stc_wf, init_bold_hmc_wf
from fmriprep.workflows.bold.base import _create_mem_gb
from niworkflows.utils.connections import listify


def set_envrion():
    # FreeSurfer recon-all env
    os.environ['FREESURFER_HOME'] = "/usr/local/freesurfer720"
    os.environ['FREESURFER'] = "/usr/local/freesurfer720"
    os.environ['SUBJECTS_DIR'] = "/home/youjia/Downloads"
    os.environ['PATH'] = ('/usr/local/freesurfer720/bin:'
                          + '/home/youjia/abin:'
                          + '/usr/local/c3d/bin:'
                          + os.environ['PATH'])



def hmc(subj_func_dir, subj_tmp_dir, bold_file, raw_ref_image, bold_id):
    mem_gb = {"filesize": 1, "resampled": 1, "largemem": 1}
    omp_nthreads = config.nipype.omp_nthreads
    if os.path.isfile(bold_file):
        bold_tlen, mem_gb = _create_mem_gb(bold_file)

    # HMC on the BOLD
    bold_hmc_wf = init_bold_hmc_wf(name="bold_hmc_wf", mem_gb=mem_gb["filesize"], omp_nthreads=omp_nthreads)
    bold_hmc_wf.inputs.inputnode.bold_file = bold_file
    bold_hmc_wf.inputs.inputnode.raw_ref_image = raw_ref_image
    bold_hmc_wf.base_dir = subj_tmp_dir
    bold_hmc_wf.run()

    xform = subj_tmp_dir / 'bold_hmc_wf' / 'fsl2itk' / 'mat2itk.txt'
    out_xform = subj_func_dir / f'{bold_id}_from-scanner_to-boldref_mode-image_xfm.txt'
    shutil.copy(xform, out_xform)

def stc(subj_func_dir, subj_tmp_dir, metadata, bold_file, bold_id):
    bold_stc_wf = init_bold_stc_wf(name="bold_stc_wf", metadata=metadata)
    bold_stc_wf.inputs.inputnode.bold_file = bold_file
    bold_stc_wf.base_dir = subj_tmp_dir
    bold_stc_wf.run()

    stc_file = subj_tmp_dir / 'bold_stc_wf' / 'copy_xform' / f'{bold_id}_space-reorient_bold_tshift_xform.nii.gz'
    out_stc_file = subj_func_dir / f'{bold_id}_space-reorient_bold_tshift_xform.nii.gz'
    shutil.copy(stc_file, out_stc_file)

def cmd(subj_func_dir, subj_tmp_dir, bids_dir, bold_file, raw_ref_image, orig_bold_file, bold_id):
    # run mc
    hmc(subj_func_dir, subj_tmp_dir, bold_file, raw_ref_image, bold_id)
    print('hmc DONE!!!!!!!!')

    # # run stc if metadata is provided
    layout = bids.BIDSLayout(str(bids_dir), derivatives=True)
    # bold_file = layout.get_files(f'{bold_id}_bold.nii.gz')
    all_metadata = [layout.get_metadata(fname) for fname in listify(orig_bold_file)]
    metadata = all_metadata[0]
    run_stc = bool(metadata.get("SliceTiming"))
    if run_stc:
        stc(subj_func_dir, subj_tmp_dir, metadata, bold_file, bold_id)
    else:
        print('No stc!!!!!!!!')
        # TODO copy? output: bold_file _reorient_bold.nii.gz
    print('stc DONE!!!!!!')



if __name__ == '__main__':
    set_envrion()

    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- MC & STC"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--reorient", required=True)  # _space-reorient_bold.nii.gz
    parser.add_argument("--orig_bold_file", required=True)  # _bold.nii.gz
    args = parser.parse_args()

    preprocess_dir = Path(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)
    subj_tmp_dir = Path(preprocess_dir) / 'tmp'
    subj_tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_ref_image = Path(subj_func_dir) / f'{args.subject_id}_boldref.nii.gz'

    skip_reorient_file = subj_func_dir / os.path.basename(args.reorient)

    cmd(subj_func_dir, subj_tmp_dir, args.bids_dir, args.reorient, raw_ref_image, args.orig_bold_file, args.bold_id)

