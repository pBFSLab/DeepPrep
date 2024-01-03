# Copyright 2023 The DeepPrep Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import bids
import argparse
import os
import shutil

from fmriprep import config
from fmriprep.workflows.bold import init_bold_stc_wf, init_bold_hmc_wf
from fmriprep.workflows.bold.base import _create_mem_gb
from niworkflows.utils.connections import listify


def hmc(subj_tmp_dir, bold_file, raw_ref_image, mc_xform):
    omp_nthreads = config.nipype.omp_nthreads
    bold_tlen, mem_gb = _create_mem_gb(bold_file)

    # HMC on the BOLD
    bold_hmc_wf = init_bold_hmc_wf(name="bold_hmc_wf", mem_gb=mem_gb["filesize"], omp_nthreads=omp_nthreads)
    bold_hmc_wf.inputs.inputnode.bold_file = bold_file
    bold_hmc_wf.inputs.inputnode.raw_ref_image = raw_ref_image
    bold_hmc_wf.base_dir = subj_tmp_dir
    bold_hmc_wf.run()

    xform = subj_tmp_dir / 'bold_hmc_wf' / 'fsl2itk' / 'mat2itk.txt'
    shutil.copy(xform, mc_xform)

def stc(subj_tmp_dir, metadata, bold_file, stc_bold):
    bold_stc_wf = init_bold_stc_wf(name="bold_stc_wf", metadata=metadata)
    bold_stc_wf.inputs.inputnode.bold_file = bold_file
    bold_stc_wf.base_dir = subj_tmp_dir
    bold_stc_wf.run()

    stc_file = sorted(Path(subj_tmp_dir, 'bold_stc_wf', 'copy_xform').glob('*_tshift_xform.nii.gz'))[0]
    shutil.copy(stc_file, stc_bold)

def cmd(subj_tmp_dir, bids_dir, bold_file, raw_ref_image, orig_bold_file, mc_xform, stc_bold):
    # run mc
    hmc(subj_tmp_dir, bold_file, raw_ref_image, mc_xform)
    print('hmc DONE!!!!!!!!')

    # # run stc if metadata is provided
    layout = bids.BIDSLayout(str(bids_dir), derivatives=True)
    all_metadata = [layout.get_metadata(fname) for fname in listify(orig_bold_file)]
    metadata = all_metadata[0]
    run_stc = bool(metadata.get("SliceTiming"))
    if run_stc:
        stc(subj_tmp_dir, metadata, bold_file, stc_bold)
    else:
        print('No stc!!!!!!!!')
        # TODO copy? output: bold_file _reorient_bold.nii.gz
    print('stc DONE!!!!!!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- MC & STC"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--reorient", required=True)  # _space-reorient_bold.nii.gz
    parser.add_argument("--orig_bold_file", required=True)  # _bold.nii.gz
    parser.add_argument("--raw_ref_image", required=True)
    parser.add_argument("--mc_xform", required=True)  # _from-scanner_to-boldref_mode-image_xfm.txt
    parser.add_argument("--stc_bold", required=True)  # _tshift_xform.nii.gz
    args = parser.parse_args()

    preprocess_dir = Path(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)
    subj_tmp_dir = Path(preprocess_dir) / 'tmp'
    subj_tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd(subj_tmp_dir, args.bids_dir, args.reorient, args.raw_ref_image, args.orig_bold_file, args.mc_xform, args.stc_bold)

