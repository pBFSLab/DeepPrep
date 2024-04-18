#! /usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
import templateflow.api as tflow
from bold_concat import concat_bold
from bold_upsampled import split_bold_convert


def run_norigid_registration_apply_bold(script, upsampled_dir, fframe_bold_output, T1_file, template, transvoxel, bold_file, batch_size):

    cmd = f'python3 {script}  -fbo {fframe_bold_output} {T1_file} {template} -tv {transvoxel} -ob {bold_file} -up {upsampled_dir} -bs {batch_size}'
    os.system(cmd)

def run_norigid_registration_apply_anat(script, anat_output, anat_t1_file, T1_file, template, transvoxel):

    cmd = f'python3 {script} {T1_file} {template} -tv {transvoxel} -ai {anat_t1_file} -ao {anat_output}'
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Apply transformation matrix from T1w to MNI on Bold or anat"
    )

    parser.add_argument("--bold_file", default=None, help="Bold file in T1w space")
    parser.add_argument("--bold_output", default=None, help="Bold output file in standard space")
    parser.add_argument("--fframe_bold_output", default=None, help="Boldref output file if bold_file exist")
    parser.add_argument("--orig_bold", default=None, help="Orig Bold file if bold_file exist")
    parser.add_argument("--anat_file", default=None, help="Anat file in T1w space")
    parser.add_argument("--anat_output", default=None, help="Anat output file in standard space")
    parser.add_argument("--t1_native2mm", required=True, help="T1w native2mm file")
    parser.add_argument("--trans_vox", required=True, help="transformation matrix")
    parser.add_argument("--template_space", required=True)
    parser.add_argument("--template_resolution", required=True)
    parser.add_argument("--work_dir", default=None, help="Folder to store bold temporary files")
    parser.add_argument("--synth_script", required=True, help="Execute the apply program")
    args = parser.parse_args()

    transvoxel = args.trans_vox
    template_resolution = args.template_resolution
    template = tflow.get(args.template_space, desc=None, resolution=template_resolution, suffix='T1w',
                         extension='nii.gz')
    if args.bold_file is not None:
        ori_bold_file = args.bold_file
        bold_id = Path(ori_bold_file).name.split('_bold')[0]
        upsampled_dir, rm_dir = split_bold_convert(ori_bold_file, args.work_dir, args.t1_native2mm, bold_id, process_num=5)
        run_norigid_registration_apply_bold(args.synth_script, upsampled_dir, args.fframe_bold_output, args.t1_native2mm, template,
                                       transvoxel, args.orig_bold, batch_size=10)
        transform_dir = Path(upsampled_dir).parent / 'transform'
        concat_bold(transform_dir, args.bold_output)
        shutil.rmtree(rm_dir)
    else:
        run_norigid_registration_apply_anat(args.synth_script, args.anat_output, args.anat_file, args.t1_native2mm, template,
                                       transvoxel)



