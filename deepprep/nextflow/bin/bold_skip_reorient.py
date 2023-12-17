import sys
import sh
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import shutil


def reorient_to_ras(input_path, output_path):
    img = nib.load(input_path)
    orig_ornt = nib.orientations.io_orientation(img.header.get_sform())
    RAS_ornt = nib.orientations.axcodes2ornt('RAS')
    if np.array_equal(orig_ornt, RAS_ornt) is True:
        print(f"{input_path} is already in RAS orientation. Copying to {output_path}.")
        shutil.copy(input_path, output_path)
    else:
        newimg = img.as_reoriented(orig_ornt)
        nib.save(newimg, output_path)
        print(f"Successfully reorient {input_path} to RAS orientation and saved to {output_path}.")


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
        reorient_to_ras(skip_bold, reorient_skip_bold)  # TODO 这里需要一个回传，告知是否进行了 reorient 操作, 并且返回Original orientation 和 Final orientation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- BoldSkipReorient"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--boldfile_path", required=True)
    parser.add_argument("--reorient", required=True)
    parser.add_argument("--skip_frame", required=True)
    args = parser.parse_args()


    with open(args.boldfile_path, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    subject_id = data[0]
    bold_file = data[1]
    preprocess_dir = Path(args.bold_preprocess_dir) / subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)

    cmd(subj_func_dir, bold_file, args.reorient, int(args.skip_frame))
