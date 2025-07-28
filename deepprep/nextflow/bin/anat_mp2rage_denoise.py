#! /usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def mp2rage_denoise(inv1_file, inv2_file, uni_file, out_file, reg):
    # loading data
    inv1_data = nib.load(inv1_file).get_fdata(dtype=np.float32)
    inv2_data = nib.load(inv2_file).get_fdata(dtype=np.float32)
    uni_img = nib.load(uni_file)
    uni_data = uni_img.get_fdata(dtype=np.float32)

    # convert uni_data to -0.5 --- 0.5 scale
    uni_convert = np.zeros_like(uni_data)
    integer_format = False
    if np.min(uni_data) >= 0 and np.max(uni_data) >= 0.51:
        uni_convert = (uni_data - np.max(uni_data) / 2) / np.max(uni_data)
        integer_format = True

    # Give the correct polarity to INV1
    inv1_data = np.sign(uni_convert) * inv1_data
    uni_convert_denominator = uni_convert.copy()
    uni_convert_denominator[uni_convert_denominator == 0] = np.inf
    inv1_pos = (-1 * inv2_data + np.sqrt(np.square(inv2_data) - 4 * uni_convert *
                                         (np.square(inv2_data) * uni_convert))) / (-2 * uni_convert_denominator)
    inv1_pos[np.isnan(inv1_pos)] = 0

    inv1_neg = (-1 * inv2_data - np.sqrt(np.square(inv2_data) - 4 * uni_convert *
                                         (np.square(inv2_data) * uni_convert))) / (-2 * uni_convert_denominator)
    inv1_neg[np.isnan(inv1_neg)] = 0

    inv1_pos_index = np.abs(inv1_data - inv1_pos) > np.abs(inv1_data - inv1_neg)
    inv1_data[inv1_pos_index] = inv1_neg[inv1_pos_index]

    inv1_neg_index = np.abs(inv1_data - inv1_pos) <= np.abs(inv1_data - inv1_neg)
    inv1_data[inv1_neg_index] = inv1_pos[inv1_neg_index]

    # lambda calculation
    noise_level = reg * np.mean(inv2_data[-10:, :, -10:])
    t1w = (np.conj(inv1_data) * inv2_data - np.square(noise_level)) / \
          (np.square(inv1_data) + np.square(inv2_data) + 2 * np.square(noise_level))
    # convert the final image to uint (if necessary)
    if integer_format:
        t1w = np.round(4095 * (t1w + 0.5))

    t1w_img = nib.Nifti1Image(t1w, uni_img.affine, uni_img.header)
    nib.save(t1w_img, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: denoise mp2rage data"
    )
    parser.add_argument("--subjects-dir", required=True)
    parser.add_argument("--mp2ragefile-path", required=True)
    parser.add_argument('--mp2rage-config', type=int, default=5, help='factor to denoise mp2rage')
    args = parser.parse_args()

    with open(args.mp2ragefile_path, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    subject_id = data[0]
    output_dir = Path(args.subjects_dir) / subject_id / "mri" / "denoise"
    output_dir.mkdir(parents=True, exist_ok=True)

    unit1_files = []
    inv1_files = []
    inv2_files = []
    
    for file in data[1:]:
        file_name = os.path.basename(file)
        if "UNIT1" in file_name:
            unit1_files.append(file)
        elif "inv-1" in file_name:
            inv1_files.append(file)
        elif "inv-2" in file_name:
            inv2_files.append(file)
    denoise_files = []
    for i in range(len(unit1_files)):
        denoise_file = output_dir / os.path.basename(unit1_files[i]).replace("UNIT1.", "dec-mp2rage_T1w.")
        mp2rage_denoise(inv1_files[i], inv2_files[i], unit1_files[i], denoise_file, args.mp2rage_config)
        denoise_files.append(str(denoise_file))
    with open(os.path.join(os.getcwd(), f'{subject_id}'), 'w') as f:
        f.write(subject_id + '\n')
        f.write('\n'.join(denoise_files))
    