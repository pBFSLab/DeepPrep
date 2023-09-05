import sh
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path
import os
import tensorflow as tf
import ants
import voxelmorph as vxm
import argparse


def VxmRegistration(subject_id: str, subjects_dir: Path, bold_preprocess_dir: Path, atlas_type: str, vxm_model_path: Path, gpuid: str):
    deepprep_subj_path = Path(bold_preprocess_dir) / subject_id

    func_dir = Path(deepprep_subj_path) / 'func'
    anat_dir = Path(deepprep_subj_path) / 'anat'
    func_dir.mkdir(parents=True, exist_ok=True)
    anat_dir.mkdir(parents=True, exist_ok=True)

    norm = Path(subjects_dir) / subject_id / 'mri' / 'norm.mgz'

    trf_fsnative2vxmatlask_affine_path = anat_dir / f'{subject_id}_from_fsnative_to_vxm{atlas_type}_ants_affine.mat'  # fromaffine trf from native T1 to vxm_MNI152
    vxm_warp = anat_dir / f'{subject_id}_from_fsnative_to_vxm{atlas_type}_vxm_nonrigid.nii.gz'  # from_fsnative_to_vxm{atlas_type}_norigid.nii.gz norigid warp from native T1 to vxm_MNI152
    trf_vxmatlas2atlask_rigid_path = anat_dir / f'{subject_id}_from_vxm{atlas_type}_to_{atlas_type}_ants_rigid.mat'

    vxm_input_npz = anat_dir / f'{subject_id}_norm_affine_space-vxm{atlas_type}.npz'  # from_fsnative_to_vxm{atlas_type}_affined.npz
    vxm_warped_path = anat_dir / f'{subject_id}_norm_space-vxm{atlas_type}.nii.gz'
    warped_path = anat_dir / f'{subject_id}_norm_space-{atlas_type}.nii.gz'

    # atlas and model
    model_file = vxm_model_path / atlas_type / 'model.h5'
    atlas_path = vxm_model_path / atlas_type / f'{atlas_type}_brain.nii.gz'  # MNI152空间模板
    vxm_atlas_path = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.nii.gz'  # vxm_MNI152空间模板
    vxm_atlas_npz_path = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.npz'
    vxm2atlas_trf = vxm_model_path / atlas_type / f'{atlas_type}_vxm2atlas.mat'  # from vxm_MNI152_nraoi to MNI152

    # ####################### ants affine transform norm to vxm_atlas
    norm = ants.image_read(str(norm))
    vxm_atlas = ants.image_read(str(vxm_atlas_path))
    tx = ants.registration(fixed=vxm_atlas, moving=norm, type_of_transform='Affine')

    # save moved
    affined = tx['warpedmovout']  # vxm的输入，应用deformation_field，输出moved
    vol = affined.numpy() / 255.0  # vxm模型输入，输入模型用来计算deformation_field
    np.savez_compressed(vxm_input_npz, vol=vol)  # vxm输入，
    # save transforms matrix
    fwdtransforms_file = Path(tx['fwdtransforms'][0])
    shutil.move(fwdtransforms_file, trf_fsnative2vxmatlask_affine_path)

    # ####################### voxelmorph norigid
    # tensorflow device handling
    if 'cuda' in gpuid:
        if len(gpuid) == 4:
            deepprep_device = '0'
        else:
            deepprep_device = gpuid.split(":")[1]
    else:
        deepprep_device = -1
    device, nb_devices = vxm.tf.utils.setup_device(deepprep_device)

    # load moving and fixed images
    moving_divide_255 = vxm.py.utils.load_volfile(str(vxm_input_npz), add_batch_axis=True,
                                                  add_feat_axis=True)
    fixed, fixed_affine = vxm.py.utils.load_volfile(str(vxm_atlas_npz_path), add_batch_axis=True,
                                                    add_feat_axis=True,
                                                    ret_affine=True)
    vxm_atlas_nib = nib.load(str(vxm_atlas_path))
    fixed_affine = vxm_atlas_nib.affine.copy()
    inshape = moving_divide_255.shape[1:-1]
    nb_feats = moving_divide_255.shape[-1]

    with tf.device(device):
        # load model and predict
        warp = vxm.networks.VxmDense.load(model_file).register(moving_divide_255, fixed)
        moving = affined.numpy()[np.newaxis, ..., np.newaxis]
        moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict(
            [moving, warp])  # if combine transform，need to know how to trans vxm_trf to ants_trf

    # save warp from norm_affine to vxmatlas
    vxm.py.utils.save_volfile(warp.squeeze(), str(vxm_warp), fixed_affine)

    # save moved image
    vxm.py.utils.save_volfile(moved.squeeze(), str(vxm_warped_path), fixed_affine)

    # from vxmatlas to atlas
    atlas = ants.image_read(str(atlas_path))
    vxm_warped = ants.image_read(str(vxm_warped_path))
    warped = ants.apply_transforms(fixed=atlas, moving=vxm_warped, transformlist=[str(vxm2atlas_trf)])
    ants.image_write(warped, str(warped_path))
    # copy trf from vxmatlas to atlas
    shutil.copy(vxm2atlas_trf, trf_vxmatlas2atlask_rigid_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- VxmRegidtration"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--atlas_type", required=True)
    parser.add_argument("--vxm_model_path", required=True)
    # parser.add_argument("--bold_id", required=True)
    parser.add_argument("--gpuid", required=True)
    args = parser.parse_args()

    # cur_path = os.getcwd()
    VxmRegistration(args.subject_id, args.subjects_dir, args.bold_preprocess_dir, args.atlas_type, Path(args.vxm_model_path), args.gpuid)
