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


def native_project_to_fs6(subj_recon_dir, input_path, out_path, reg_path, hemi):
    """
    Project a volume (e.g., residuals) in native space onto the
    fsaverage6 surface.

    subject     - str. Subject ID.
    input_path  - Path. Volume to project.
    reg_path    - Path. Registaration .dat file.
    hemi        - str from {'lh', 'rh'}.

    Output file created at $DATA_DIR/surf/
     input_path.name.replace(input_path.ext, '_fsaverage6' + input_path.ext))
    Path object pointing to this file is returned.
    """
    fsaverage6_dir = subj_recon_dir / 'fsaverage6'
    if not fsaverage6_dir.exists():
        src_fsaverage6_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage6'
        os.symlink(src_fsaverage6_dir, fsaverage6_dir)
    os.environ['SUBJECTS_DIR'] = str(subj_recon_dir)
    sh.mri_vol2surf(
        '--mov', input_path,
        '--reg', reg_path,
        '--hemi', hemi,
        '--projfrac', 0.5,
        '--trgsubject', 'fsaverage6',
        '--o', out_path,
        '--reshape',
        '--interp', 'trilinear',
    )
    return out_path


def register_dat_to_fslmat(bold_mc_file, norm_fsnative_2mm_file, register_dat_file, fslmat_file):
    sh.tkregister2('--mov', bold_mc_file,
                   '--targ', norm_fsnative_2mm_file,
                   '--reg', register_dat_file,
                   '--fslregout', fslmat_file,
                   '--noedit')
    (register_dat_file.parent / register_dat_file.name.replace('.dat', '.dat~')).unlink(missing_ok=True)


def register_dat_to_trf(bold_mc_file: Path, norm_fsnative_2mm_file, register_dat_file, ants_trf_file, resource_dir):
    import SimpleITK as sitk

    tmp_dir = bold_mc_file.parent / bold_mc_file.name.replace('.nii.gz', '')
    tmp_dir.mkdir(exist_ok=True)

    fsltrf_file = tmp_dir / bold_mc_file.name.replace('.nii.gz', f'_from_mc_to_norm_2mm_fsl_rigid.fsl')
    register_dat_to_fslmat(bold_mc_file, norm_fsnative_2mm_file, register_dat_file, fsltrf_file)

    c3d_affine_tool = Path(resource_dir) / 'c3d_affine_tool'
    template_file = bold_mc_file.parent / bold_mc_file.name.replace('_mc.nii.gz', '_boldref.nii.gz')
    tfm_file = tmp_dir / bold_mc_file.name.replace('.nii.gz', f'_from_mc_to_norm2mm_itk_rigid.tfm')
    cmd = f'{c3d_affine_tool} -ref {norm_fsnative_2mm_file} -src {template_file} {fsltrf_file} -fsl2ras -oitk {tfm_file}'
    os.system(cmd)

    trf_sitk = sitk.ReadTransform(str(tfm_file))
    trf = ants.new_ants_transform()
    trf.set_parameters(trf_sitk.GetParameters())
    trf.set_fixed_parameters(trf_sitk.GetFixedParameters())
    ants.write_transform(trf, ants_trf_file)

    shutil.rmtree(tmp_dir)


def bold_mc_to_fsnative2mm_ants(bold_mc_file: Path, norm_fsnative2mm_file, register_dat_file,
                                bold_fsnative2mm_file: Path, func_dir: Path, resource_dir: Path, verbose=False):
    """
    bold_mc_file : moving
    norm_fsnative_file : norm.mgz
    register_dat_file : bbregister.register.dat
    """

    # 将bbregister dat文件转换为ants trf mat文件
    ants_rigid_trf_file = func_dir / bold_mc_file.name.replace('.nii.gz', '_from_mc_to_fsnative_ants_rigid.mat')
    register_dat_to_trf(bold_mc_file, norm_fsnative2mm_file, register_dat_file, ants_rigid_trf_file, resource_dir)

    bold_img = ants.image_read(str(bold_mc_file))
    fixed = ants.image_read(str(norm_fsnative2mm_file))
    affined_bold_img = ants.apply_transforms(fixed=fixed, moving=bold_img, transformlist=[str(ants_rigid_trf_file)],
                                             imagetype=3)

    if verbose:
        affine_info = nib.load(norm_fsnative2mm_file).affine
        header_info = nib.load(bold_mc_file).header
        affined_bold_img_np = affined_bold_img.numpy().astype(int)
        affined_nib_img = nib.Nifti1Image(affined_bold_img_np, affine=affine_info, header=header_info)
        nib.save(affined_nib_img, bold_fsnative2mm_file)

        # save one frame for plotting
        nib_fframe_img = ants.apply_transforms(fixed=fixed,
                                               moving=ants.from_numpy(bold_img[..., 0:1], bold_img.origin, bold_img.spacing, bold_img.direction),
                                               interpolator='nearestNeighbor',
                                               transformlist=[str(ants_rigid_trf_file)], imagetype=3)
        nib_fframe_img = nib.Nifti1Image(nib_fframe_img.numpy().astype(int)[..., 0], affine=affine_info, header=header_info)
        nib.save(nib_fframe_img, bold_fsnative2mm_file.parent / bold_fsnative2mm_file.name.replace('.nii.gz', '_fframe.nii.gz'))

    return affined_bold_img


def vxm_warp_bold_2mm(vxm_model_path, bold_fsnative2mm, bold_fsnative2mm_file, atlas_type, gpuid,
                      trt_ants_affine_file, trt_vxm_norigid_file, warped_file, batch_size, verbose=True):
    import voxelmorph as vxm

    vxm_model_path = Path(vxm_model_path)
    atlas_type = atlas_type

    vxm_atlas_file = vxm_model_path / atlas_type / f'{atlas_type}_brain_vxm.nii.gz'
    MNI152_2mm_file = vxm_model_path / atlas_type / f'{atlas_type}_brain.nii.gz'
    MNI152_2mm = ants.image_read(str(MNI152_2mm_file))
    vxm_atlas = ants.image_read(str(vxm_atlas_file))
    if isinstance(bold_fsnative2mm, str):
        bold_img = ants.image_read(bold_fsnative2mm)
    else:
        bold_img = bold_fsnative2mm
    n_frame = bold_img.shape[3]
    bold_origin = bold_img.origin
    bold_spacing = bold_img.spacing
    bold_direction = bold_img.direction.copy()

    # tensorflow device handling
    if 'cuda' in gpuid:
        if len(gpuid) == 4:
            deepprep_device = '0'
        else:
            deepprep_device = gpuid.split(":")[1]
    else:
        deepprep_device = -1
    device, nb_devices = vxm.tf.utils.setup_device(deepprep_device)

    fwdtrf_MNI152_2mm = [str(trt_ants_affine_file)]
    trf_file = vxm_model_path / atlas_type / f'{atlas_type}_vxm2atlas.mat'
    fwdtrf_atlas2MNI152_2mm = [str(trf_file)]
    deform, deform_affine = vxm.py.utils.load_volfile(str(trt_vxm_norigid_file), add_batch_axis=True, ret_affine=True)

    # affine to MNI152 croped
    affined_np = ants.apply_transforms(vxm_atlas, bold_img, fwdtrf_MNI152_2mm, imagetype=3).numpy()

    # voxelmorph warp
    warped_np = np.zeros(shape=(*vxm_atlas.shape, n_frame), dtype=np.float32)
    with tf.device(device):
        transform = vxm.networks.Transform(vxm_atlas.shape, interp_method='linear', nb_feats=1)
        tf_dataset = tf.data.Dataset.from_tensor_slices(np.transpose(affined_np, (3, 0, 1, 2)))
        del affined_np
        batch_size = int(batch_size)
        deform = tf.convert_to_tensor(deform)
        deform = tf.keras.backend.tile(deform, [batch_size, 1, 1, 1, 1])
        for idx, batch_data in enumerate(tf_dataset.batch(batch_size=batch_size)):
            if batch_data.shape[0] != deform.shape[0]:
                deform = deform[:batch_data.shape[0], :, :, :, :]
            moved = transform.predict([batch_data, deform]).squeeze()
            if len(moved.shape) == 4:
                moved_data = np.transpose(moved, (1, 2, 3, 0))
            else:
                moved_data = moved[:, :, :, np.newaxis]
            warped_np[:, :, :, idx * batch_size:(idx + 1) * batch_size] = moved_data
        del transform
        del tf_dataset
        del moved
        del moved_data

    # affine to MNI152
    origin = (*vxm_atlas.origin, bold_origin[3])
    spacing = (*vxm_atlas.spacing, bold_spacing[3])
    direction = bold_direction.copy()
    direction[:3, :3] = vxm_atlas.direction

    warped_img = ants.from_numpy(warped_np, origin=origin, spacing=spacing, direction=direction)
    del warped_np
    moved_img = ants.apply_transforms(MNI152_2mm, warped_img, fwdtrf_atlas2MNI152_2mm, imagetype=3)
    del warped_img

    if verbose:
        affine_info = nib.load(MNI152_2mm_file).affine
        header_info = nib.load(bold_fsnative2mm_file).header
        moved_img_np = moved_img.numpy().astype(int)
        nib_img = nib.Nifti1Image(moved_img_np, affine=affine_info, header=header_info)
        nib.save(nib_img, warped_file)
        nib_fframe_img = nib.Nifti1Image(moved_img_np[..., 0], affine=affine_info, header=header_info)
        nib.save(nib_fframe_img, warped_file.parent / warped_file.name.replace('.nii.gz', '_fframe.nii.gz'))
    return moved_img


def VxmRegNormMNI152(subj_recon_dir, deepprep_subj_path, subject_id, atlas_type, subjects_dir, vxm_model_path, bold_mc_file,
                     register_dat_file, resource_dir, batch_size, gpuid, standard_space, fs_native_space):
    subj_func_dir = Path(deepprep_subj_path) / 'func'
    subj_anat_dir = Path(deepprep_subj_path) / 'anat'
    subj_surf_dir = Path(deepprep_subj_path) / 'surf'
    subj_func_dir.mkdir(parents=True, exist_ok=True)
    subj_surf_dir.mkdir(parents=True, exist_ok=True)

    norm_fsnative_file = Path(subjects_dir) / subject_id / 'mri' / 'norm.mgz'
    norm_fsnative2mm_file = subj_anat_dir / f'{subject_id}_norm_2mm.nii.gz'
    if not norm_fsnative2mm_file.exists():
        sh.mri_convert('-ds', 2, 2, 2,
                       '-i', norm_fsnative_file,
                       '-o', norm_fsnative2mm_file)

    bold_fsnative2mm_file = subj_func_dir / bold_mc_file.name.replace('.nii.gz',
                                                                      '_space-fsnative2mm.nii.gz')  # save reg to T1 result file

    bold_fsnative2mm_img = bold_mc_to_fsnative2mm_ants(bold_mc_file, norm_fsnative2mm_file, register_dat_file,
                                                       bold_fsnative2mm_file, subj_func_dir, resource_dir,
                                                       verbose=fs_native_space)

    ants_affine_trt_file = subj_anat_dir / f'{subject_id}_from_fsnative_to_vxm{atlas_type}_ants_affine.mat'
    vxm_nonrigid_trt_file = subj_anat_dir / f'{subject_id}_from_fsnative_to_vxm{atlas_type}_vxm_nonrigid.nii.gz'
    bold_atlas_file = subj_func_dir / bold_mc_file.name.replace('.nii.gz',
                                                                f'_space-{atlas_type}.nii.gz')  # save reg to MNI152 result file
    vxm_warp_bold_2mm(vxm_model_path, bold_fsnative2mm_img, bold_mc_file, atlas_type, gpuid,
                      ants_affine_trt_file, vxm_nonrigid_trt_file, bold_atlas_file,
                      batch_size, verbose=standard_space)

    bold_fsaverage_lh_file = subj_surf_dir / bold_mc_file.name.replace('.nii.gz', '_hemi-L_space-fsaverage6.nii.gz')
    native_project_to_fs6(subj_recon_dir, bold_mc_file, bold_fsaverage_lh_file, register_dat_file, 'lh')
    bold_fsaverage_rh_file = subj_surf_dir / bold_mc_file.name.replace('.nii.gz', '_hemi-R_space-fsaverage6.nii.gz')
    native_project_to_fs6(subj_recon_dir, bold_mc_file, bold_fsaverage_rh_file, register_dat_file, 'rh')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- VxmRegNormMNI152"
    )

    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--atlas_type", required=True)
    parser.add_argument("--vxm_model_path", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--mc", required=True)
    parser.add_argument("--bbregister_dat", required=True)
    parser.add_argument("--resource_dir", required=True)
    parser.add_argument("--batch_size", required=True)
    parser.add_argument("--gpuid", required=True)
    parser.add_argument("--ants_affine_trt", required=True)
    parser.add_argument("--vxm_nonrigid_trt", required=True)
    parser.add_argument("--standard_space", required=True)
    parser.add_argument("--fs_native_space", required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()
    preprocess_dir = Path(cur_path) / str(args.bold_preprocess_dir) / args.subject_id
    subj_func_dir = Path(preprocess_dir) / 'func'
    subj_func_dir.mkdir(parents=True, exist_ok=True)
    subj_recon_dir = Path(cur_path) / str(args.subjects_dir)

    mc_file = subj_func_dir / f'{args.bold_id}_skip_reorient_stc_mc.nii.gz'
    bbregister_dat = subj_func_dir / f'{args.bold_id}_skip_reorient_stc_mc_from_mc_to_fsnative_bbregister_rigid.dat'
    VxmRegNormMNI152(subj_recon_dir, preprocess_dir, args.subject_id, args.atlas_type, args.subjects_dir, args.vxm_model_path,
                     Path(mc_file), Path(bbregister_dat), args.resource_dir, args.batch_size, args.gpuid,
                     bool(args.standard_space), bool(args.fs_native_space))
