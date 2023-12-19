import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import shutil
from nipype import Node
from reports.reports_node import FunctionalSummary
import os


def get_tr(bold_file):
    bold_img = nib.load(bold_file)
    tr = bold_img.header.get_zooms()[3]
    return tr


def FunctionalSummary_run(subject_id: str, bold_id: str, skip_frame: int, orientation: str,
                          tr: int, slice_timing: bool, sdc: str, qc_report_dir: str):
    node_name = 'functional_Reports_run_node'
    FunctionalSummary_node = Node(FunctionalSummary(), node_name)

    FunctionalSummary_node.inputs.orientation = orientation  # 'LAS'
    FunctionalSummary_node.inputs.tr = tr  # 2 unit(s)
    FunctionalSummary_node.inputs.skip_frame = skip_frame  # 0
    FunctionalSummary_node.inputs.slice_timing = slice_timing  # True
    FunctionalSummary_node.inputs.distortion_correction = sdc
    FunctionalSummary_node.inputs.registration = 'FreeSurfer and SynthMorph'

    # TODO these are not valid
    FunctionalSummary_node.inputs.pe_direction = 'i'
    FunctionalSummary_node.inputs.registration_dof = 9
    FunctionalSummary_node.inputs.registration_init = 'register'
    FunctionalSummary_node.inputs.fallback = True

    FunctionalSummary_node.base_dir = Path().cwd()
    FunctionalSummary_node.run()

    figures_dir = Path(qc_report_dir) / subject_id / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    assert (Path(node_name) / 'report.html').exists()
    shutil.copyfile(Path(node_name) / 'report.html',
                    figures_dir / f'{bold_id}_desc-functionalsummary_report.html')


def reorient_to_ras(input_path, output_path):
    img = nib.load(input_path)
    orig_ornt = nib.orientations.io_orientation(img.header.get_sform())
    orig_axcodes = nib.orientations.ornt2axcodes(orig_ornt)
    RAS_ornt = nib.orientations.axcodes2ornt('RAS')
    if np.array_equal(orig_ornt, RAS_ornt) is True:
        print(f"{input_path} is already in RAS orientation. Copying to {output_path}.")
        shutil.copy(input_path, output_path)
    else:
        newimg = img.as_reoriented(orig_ornt)
        nib.save(newimg, output_path)
        print(f"Successfully reorient {input_path} to RAS orientation and saved to {output_path}.")
    return orig_axcodes


def cmd(subj_func_dir: Path, bold: str, reorient: str, nskip_frame: int):
    skip_bold = Path(subj_func_dir) / Path(bold).name.replace('_bold.nii.gz', '_desc-skip_bold.nii.gz')
    reorient_skip_bold = Path(subj_func_dir) / Path(bold).name.replace('_bold.nii.gz', '_space-reorient_bold.nii.gz')

    # skip 0 frame
    if nskip_frame > 0:
        sh.mri_convert('-i', bold, '--nskip', nskip_frame, '-o', skip_bold, _out=sys.stdout)
    else:
        skip_bold = bold

    # reorient
    if reorient.upper() == 'TRUE':
        orig_reorient = reorient_to_ras(skip_bold, reorient_skip_bold)
    else:
        img = nib.load(skip_bold)
        orig_ornt = nib.orientations.io_orientation(img.header.get_sform())
        orig_reorient = nib.orientations.ornt2axcodes(orig_ornt)
    orig_reorient = ''.join(orig_reorient)
    return orig_reorient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- BoldSkipReorient"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--qc_report_dir", required=True)
    parser.add_argument("--boldfile_path", required=True)
    parser.add_argument("--reorient", required=True)
    parser.add_argument("--skip_frame", required=True)
    parser.add_argument("--sdc", required=True, default='False')
    parser.add_argument("--stc", default='True')
    args = parser.parse_args()

    # preprocess
    with open(args.boldfile_path, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    subject_id = data[0]
    bold_file = data[1]
    preprocess_dir = Path(args.bold_preprocess_dir) / subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    orig_reorient = cmd(subj_func_dir, bold_file, args.reorient, int(args.skip_frame))

    # for qc report
    bold_id = os.path.basename(bold_file).replace('_bold.nii.gz', '')
    tr = get_tr(bold_file)
    skip_frame = int(args.skip_frame)
    if args.stc.upper() == 'TRUE':
        slice_time = True
    else:
        slice_time = False

    FunctionalSummary_run(subject_id, bold_id,
                          skip_frame, orig_reorient, tr, slice_time, args.sdc, args.qc_report_dir)
