#!/usr/bin/env python3

import os
import shutil

import nibabel as nib
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
import argparse

from nitransforms.io import get_linear_factory
import templateflow.api as tflow
from multiprocessing import Pool
from bids import BIDSLayout


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


def affine_to_3x3(itk):
    matrix = itk[:3, :3]
    translation = itk[:3, -1:]
    return matrix, translation

def apply_hmc_pool(frame, warped_mesh_frame, matrix_frame, ras2vox_A, ras2vox_b, bold_orig, fixed, bold_orig_header, transform_save_path):
    # bold_orig = nib.load(bold_file)
    bold_orig_values = bold_orig.slicer[..., frame:frame+1].get_fdata()[..., 0]

    hmc_A, hmc_b = affine_to_3x3(matrix_frame)
    warped_mesh_frame = hmc_A @ warped_mesh_frame + hmc_b

    warped_mesh_frame = ras2vox_A @ warped_mesh_frame + ras2vox_b
    warped_mesh_frame = warped_mesh_frame.reshape(3, *fixed.shape)

    # interp values
    output = np.zeros(
        list(fixed.shape),
        order='F',
        dtype=np.float64
    )
    result = ndi.map_coordinates(
        bold_orig_values,
        warped_mesh_frame,
        output=output,
        order=3,
        mode='constant',
        cval=0.0,
        prefilter=True,
    )
    result = result[..., np.newaxis]
    nib.save(nib.Nifti1Image(result, affine=fixed.affine, header=bold_orig_header),
         f'{transform_save_path}/t{str(frame).zfill(5)}.nii.gz')

def concat_frames(transform_save_path, output_path, boldref_path, t1_json):
    # concat frames
    files = sorted(transform_save_path.glob('*.nii.gz'))
    in_files = [str(f) for f in files]
    cmd = f'mri_concat --i {transform_save_path}/* --o {output_path}'
    os.system(cmd)

    # copy the first frame as boldref
    shutil.copy(in_files[0], boldref_path)

    # generate .json, it is consistent with T1w.json
    boldref_json_path = str(output_path).replace('.nii.gz', '.json')
    try:
        shutil.copy(t1_json, boldref_json_path)
    except shutil.SameFileError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold apply transform chain "
                    "-- register the original BOLD to Template."
    )
    """
    --bids_dir: ~/input/dataset
    --bold_preprocess_dir: ~/output/BOLD
    --work_dir: ~/output/WorkDir
    --subject_id sub-01
    --bold_id: sub-01_ses-01_task-rest_run-01
    --subject_boldfile_txt_bold /mnt/ngshare2/DeepPrep_workdir/issue_synthmorph_result/BOLD/sub-01_task-test_run-01
    --template_space MNI152NLin6Asym
    --template_resolution 02
    --nonlinear_file ~/output/BOLD/sub-01/anat/sub-01_from-T1w_to-MNI152NLin6Asym_desc-fixed_xfm.nii.gz
    """
    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--subject_boldfile_txt_bold", required=True)
    parser.add_argument("--template_space", required=True)
    parser.add_argument("--template_resolution", required=True)
    parser.add_argument("--nonlinear_file", required=True)
    parser.add_argument("--reference", required=False, default=None)
    parser.add_argument("--moving", required=False, default=None)
    args = parser.parse_args()

    # load required files
    fixed_file = tflow.get(args.template_space, desc=None, resolution=args.template_resolution, suffix='T1w', extension='nii.gz')
    nonlinear_file = args.nonlinear_file

    # load the original BOLD
    with open(args.subject_boldfile_txt_bold, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_file = data[1]

    # get the coreg.xfm
    update_entities = {'desc': 'coreg', 'suffix': 'xfm', 'extension': '.txt'}
    coreg_xfm = get_preproc_file(args.subject_id, args.bold_preprocess_dir, bold_file, update_entities)

    # get the hmc.xfm
    update_entities = {'desc': 'hmc', 'suffix': 'xfm', 'extension': '.txt'}
    hmc_xfm = get_preproc_file(args.subject_id, args.bold_preprocess_dir, bold_file, update_entities)

    # get the T1w.json
    update_entities = {'desc': 'preproc', 'suffix': 'bold', 'extension': '.json'}
    t1_json = get_preproc_file(args.subject_id, args.bold_preprocess_dir, bold_file, update_entities)

    transform_save_path = Path(args.work_dir) / 'bold_synthmorph_transform_chain' / args.bold_id
    transform_save_path.mkdir(exist_ok=True, parents=True)
    output_path = Path(coreg_xfm.parent) / f'{args.bold_id}_space-{args.template_space}_res-{args.template_resolution}_desc-preproc_bold.nii.gz'
    boldref_path = Path(coreg_xfm.parent) / f'{args.bold_id}_space-{args.template_space}_res-{args.template_resolution}_boldref.nii.gz'

    # Load the fixed file
    fixed = nib.load(fixed_file)
    vox2ras = fixed.affine
    fixed_vox2ras_rot, fixed_vox2ras_trans = affine_to_3x3(vox2ras)
    # generate meshgrid
    i = np.linspace(0, fixed.shape[0] - 1, fixed.shape[0])
    j = np.linspace(0, fixed.shape[1] - 1, fixed.shape[1])
    k = np.linspace(0, fixed.shape[2] - 1, fixed.shape[2])
    mesh = np.meshgrid(i, j, k, indexing='ij')

    # load non-linear transform, vox
    warp_matrix = nib.load(nonlinear_file).get_fdata()

    # reshape to 3xN
    mesh = np.asarray(mesh).reshape(3, -1)
    # convert to RAS space
    mesh = fixed_vox2ras_rot @ mesh + fixed_vox2ras_trans
    mesh = mesh.reshape(3, *fixed.shape)
    # add the nonlinear transformation
    warped_mesh = [mesh[d].astype(np.float32) + warp_matrix[..., d] for d in range(3)]
    warped_mesh = np.asarray(warped_mesh, dtype=np.float32)
    warped_mesh = warped_mesh.reshape(3, -1)

    # load the coreg
    struct = get_linear_factory(
        'itk',
        is_array=True
    ).from_filename(coreg_xfm)
    matrix = struct.to_ras(reference=args.reference, moving=args.moving).squeeze()
    A, b = affine_to_3x3(matrix)
    warped_mesh = A @ warped_mesh + b

    # load hmc
    struct = get_linear_factory(
        'itk',
        is_array=True
    ).from_filename(hmc_xfm)
    matrix = struct.to_ras(reference=args.reference, moving=args.moving)

    bold_orig = nib.load(bold_file)
    bold_orig_header = bold_orig.header.copy()
    ras2vox_bold = np.linalg.inv(bold_orig.affine)
    ras2vox_A, ras2vox_b = affine_to_3x3(ras2vox_bold)

    args_apply_hmc = []
    for i in range(matrix.shape[0]):
        args_apply_hmc.append([int(i), warped_mesh, matrix[i], ras2vox_A, ras2vox_b, bold_orig, fixed, bold_orig_header, transform_save_path])
    pool = Pool(10)
    pool.starmap(apply_hmc_pool, args_apply_hmc)
    pool.close()
    pool.join()

    concat_frames(transform_save_path, output_path, boldref_path, t1_json)

    # check the output file has correct number of frames
    concat_bold = nib.load(output_path)
    assert concat_bold.shape[-1] == bold_orig.shape[-1]
