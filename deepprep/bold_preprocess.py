import os
import argparse
import sys
from pathlib import Path
import json
import shutil
import time
import bids
import sh
import ants
import nibabel as nib
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import SimpleITK as sitk
from bidict import bidict
from app.surface_projection import surface_projection as sp
from app.volume_projection import volume_projection as vp
from app.utils.utils import timing_func
from app.filters.filters import gauss_nifti, bandpass_nifti
from app.regressors.regressors import compile_regressors, regression


# from memory_profiler import profile
# import gc


def set_envrion():
    # FreeSurfer recon-all env
    os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
    os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
    os.environ['PATH'] = '/usr/local/freesurfer/bin:/usr/local/freesurfer/mni/bin:/usr/local/freesurfer/tktools:' + \
                         '/usr/local/freesurfer/fsfast/bin:' + os.environ['PATH']
    os.environ['MINC_BIN_DIR'] = '/usr/local/freesurfer/mni/bin'
    os.environ['MINC_LIB_DIR'] = '/usr/local/freesurfer/mni/lib'
    os.environ['PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'
    os.environ['MNI_PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'
    # FreeSurfer fsfast env
    os.environ['FSF_OUTPUT_FORMAT'] = 'nii.gz'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

    # FSL
    os.environ['PATH'] = '/usr/local/fsl/bin:' + os.environ['PATH']


@timing_func
def vxm_registraion(atlas_type, model_file, workdir, norm_path, trf_path, warp_path, warped_path):
    vxm_warp_path = workdir / 'warp.nii.gz'
    vxm_warped_path = workdir / 'warped.nii.gz'

    model_path = Path(__file__).parent / 'model' / 'voxelmorph' / atlas_type
    # atlas
    if atlas_type == 'MNI152_T1_1mm':
        atlas_path = '../../data/atlas/MNI152_T1_1mm_brain.nii.gz'
        vxm_atlas_path = '../../data/atlas/MNI152_T1_1mm_brain_vxm.nii.gz'
        vxm_atlas_npz_path = '../../data/atlas/MNI152_T1_1mm_brain_vxm.npz'
        vxm2atlas_trf = '../../data/atlas/MNI152_T1_1mm_vxm2atlas.mat'
    elif atlas_type == 'MNI152_T1_2mm':
        atlas_path = model_path / 'MNI152_T1_2mm_brain.nii.gz'
        vxm_atlas_path = model_path / 'MNI152_T1_2mm_brain_vxm.nii.gz'
        vxm_atlas_npz_path = model_path / 'MNI152_T1_2mm_brain_vxm.npz'
        vxm2atlas_trf = model_path / 'MNI152_T1_2mm_vxm2atlas.mat'
    elif atlas_type == 'FS_T1_2mm':
        atlas_path = '../../data/atlas/FS_T1_2mm_brain.nii.gz'
        vxm_atlas_path = '../../data/atlas/FS_T1_2mm_brain_vxm.nii.gz'
        vxm_atlas_npz_path = '../../data/atlas/FS_T1_2mm_brain_vxm.npz'
        vxm2atlas_trf = '../../data/atlas/FS_T1_2mm_vxm2atlas.mat'
    else:
        raise Exception('atlas type error')

    norm = ants.image_read(str(norm_path))
    vxm_atlas = ants.image_read(str(vxm_atlas_path))
    tx = ants.registration(fixed=vxm_atlas, moving=norm, type_of_transform='Affine')
    trf = ants.read_transform(tx['fwdtransforms'][0])
    ants.write_transform(trf, str(trf_path))
    affined = tx['warpedmovout']
    vol = affined.numpy() / 255.0
    npz_path = workdir / 'vxminput.npz'
    np.savez_compressed(npz_path, vol=vol)

    # voxelmorph
    # tensorflow device handling
    gpuid = '0'
    device, nb_devices = vxm.tf.utils.setup_device(gpuid)

    # load moving and fixed images
    add_feat_axis = True
    moving = vxm.py.utils.load_volfile(str(npz_path), add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(str(vxm_atlas_npz_path), add_batch_axis=True,
                                                    add_feat_axis=add_feat_axis,
                                                    ret_affine=True)
    vxm_atlas_nib = nib.load(str(vxm_atlas_path))
    fixed_affine = vxm_atlas_nib.affine.copy()
    inshape = moving.shape[1:-1]
    nb_feats = moving.shape[-1]

    with tf.device(device):
        # load model and predict
        warp = vxm.networks.VxmDense.load(model_file).register(moving, fixed)
        # warp = vxm.networks.VxmDenseSemiSupervisedSeg.load(args.model).register(moving, fixed)
        moving = affined.numpy()[np.newaxis, ..., np.newaxis]
        moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

    # save warp
    vxm.py.utils.save_volfile(warp.squeeze(), str(vxm_warp_path), fixed_affine)
    shutil.copy(vxm_warp_path, warp_path)

    # save moved image
    vxm.py.utils.save_volfile(moved.squeeze(), str(vxm_warped_path), fixed_affine)

    # affine to atlas
    atlas = ants.image_read(str(atlas_path))
    vxm_warped = ants.image_read(str(vxm_warped_path))
    warped = ants.apply_transforms(fixed=atlas, moving=vxm_warped, transformlist=[str(vxm2atlas_trf)])
    warped_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(warped, str(warped_path))


def native_bold_to_T1_2mm(residual_file, subj, subj_t1_file, reg_file, save_file):
    subj_t1_2mm_file = os.path.join(os.path.split(save_file)[0], 'norm_2mm.mgz')
    sh.mri_convert('-ds', 2, 2, 2,
                   '-i', subj_t1_file,
                   '-o', subj_t1_2mm_file)
    sh.mri_vol2vol('--mov', residual_file,
                   '--s', subj,
                   '--targ', subj_t1_2mm_file,
                   '--reg', reg_file,
                   '--o', save_file)


def register_dat_to_fslmat(mov_file, ref_file, reg_file, fslmat_file):
    sh.tkregister2('--mov', mov_file,
                   '--targ', ref_file,
                   '--reg', reg_file,
                   '--fslregout', fslmat_file,
                   '--noedit')


def register_dat_to_trf(mov_file, ref_file, reg_file, workdir, trf_file):
    fsltrf_file = os.path.join(workdir, 'fsl_trf.fsl')
    register_dat_to_fslmat(mov_file, ref_file, reg_file, fsltrf_file)
    first_frame_file = os.path.join(workdir, 'frame0.nii.gz')
    bold = ants.image_read(str(mov_file))
    frame0_np = bold[:, :, :, 0]
    origin = bold.origin[:3]
    spacing = bold.spacing[:3]
    direction = bold.direction[:3, :3].copy()
    frame0 = ants.from_numpy(frame0_np, origin=origin, spacing=spacing, direction=direction)
    ants.image_write(frame0, str(first_frame_file))
    tfm_file = os.path.join(workdir, 'itk_trf.tfm')
    base_path, _ = os.path.split(os.path.abspath(__file__))
    c3d_affine_tool = os.path.join(base_path, 'resource', 'c3d_affine_tool')
    cmd = f'{c3d_affine_tool} -ref {ref_file} -src {first_frame_file} {fsltrf_file} -fsl2ras -oitk {tfm_file}'
    os.system(cmd)
    trf_sitk = sitk.ReadTransform(tfm_file)
    trf = ants.new_ants_transform()
    trf.set_parameters(trf_sitk.GetParameters())
    trf.set_fixed_parameters(trf_sitk.GetFixedParameters())
    ants.write_transform(trf, trf_file)


# @profile
def native_bold_to_T1_2mm_ants(residual_file, subj, subj_t1_file, reg_file, save_file, workdir, verbose=False):
    subj_t1_2mm_file = os.path.join(os.path.split(save_file)[0], 'norm_2mm.nii.gz')
    sh.mri_convert('-ds', 2, 2, 2,
                   '-i', subj_t1_file,
                   '-o', subj_t1_2mm_file)
    trf_file = os.path.join(workdir, 'reg.mat')
    register_dat_to_trf(residual_file, subj_t1_2mm_file, reg_file, workdir, trf_file)
    bold_img = ants.image_read(str(residual_file))
    fixed = ants.image_read(subj_t1_2mm_file)
    affined_bold_img = ants.apply_transforms(fixed=fixed, moving=bold_img, transformlist=[trf_file], imagetype=3)
    if verbose:
        ants.image_write(affined_bold_img, save_file)
    return affined_bold_img


def bold_smooth_6(t12mm_file, t12mm_sm6_file):
    MNI152_T1_2mm_brain_mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    sh.mri_fwhm('--nthreads', 8,
                '--fwhm', 6,
                '--mask', MNI152_T1_2mm_brain_mask,
                '--i', t12mm_file,
                '--o', t12mm_sm6_file)


# @profile
def bold_smooth_6_ants(t12mm, t12mm_sm6_file, temp_file, bold_file, verbose=False):
    # mask file
    MNI152_T1_2mm_brain_mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    brain_mask = ants.image_read(MNI152_T1_2mm_brain_mask)

    if isinstance(t12mm, str):
        bold_img = ants.image_read(t12mm)
    else:
        bold_img = t12mm

    bold_origin = bold_img.origin
    bold_spacing = bold_img.spacing
    bold_direction = bold_img.direction.copy()

    # smooth
    smoothed_img = ants.from_numpy(bold_img.numpy(), bold_origin[:3], bold_spacing[:3],
                                   bold_direction[:3, :3].copy(), has_components=True)
    # mask
    smoothed_np = ants.smooth_image(smoothed_img, sigma=6, FWHM=True).numpy()
    del smoothed_img
    mask_np = brain_mask.numpy()
    masked_np = np.zeros(smoothed_np.shape, dtype=np.float32)
    idx = mask_np == 1
    masked_np[idx, :] = smoothed_np[idx, :]
    del smoothed_np
    masked_img = ants.from_numpy(masked_np, bold_origin, bold_spacing, bold_direction)
    del masked_np
    if verbose:
        # save
        save_bold(masked_img, temp_file, bold_file, t12mm_sm6_file)
        # ants.image_write(masked_img, str(t12mm_sm6_file))
    return masked_img


def save_bold(warped_img, temp_file, bold_file, save_file):
    ants.image_write(warped_img, str(temp_file))
    bold_info = nib.load(bold_file)
    affine_info = nib.load(temp_file)
    bold2 = nib.Nifti1Image(warped_img.numpy(), affine=affine_info.affine, header=bold_info.header)
    del bold_info
    del affine_info
    os.remove(temp_file)
    nib.save(bold2, save_file)


# @profile
def vxm_warp_bold_2mm(resid_t1, affine_file, warp_file, warped_file, verbose=False):
    atlas_file = Path(__file__).parent / 'model' / 'voxelmorph' / 'MNI152_T1_2mm' / 'MNI152_T1_2mm_brain_vxm.nii.gz'
    MNI152_2mm_file = Path(__file__).parent / 'model' / 'voxelmorph' / 'MNI152_T1_2mm' / 'MNI152_T1_2mm_brain.nii.gz'
    MNI152_2mm = ants.image_read(str(MNI152_2mm_file))
    atlas = ants.image_read(str(atlas_file))
    if isinstance(resid_t1, str):
        bold_img = ants.image_read(resid_t1)
    else:
        bold_img = resid_t1
    n_frame = bold_img.shape[3]
    bold_origin = bold_img.origin
    bold_spacing = bold_img.spacing
    bold_direction = bold_img.direction.copy()

    # tensorflow device handling
    gpuid = '0'
    device, nb_devices = vxm.tf.utils.setup_device(gpuid)

    fwdtrf_MNI152_2mm = [str(affine_file)]
    trf_file = Path(__file__).parent / 'model' / 'voxelmorph' / 'MNI152_T1_2mm' / 'MNI152_T1_2mm_vxm2atlas.mat'
    fwdtrf_atlas2MNI152_2mm = [str(trf_file)]
    deform, deform_affine = vxm.py.utils.load_volfile(str(warp_file), add_batch_axis=True, ret_affine=True)

    # affine to MNI152 croped
    tic = time.time()
    # affined_img = ants.apply_transforms(atlas, bold_img, fwdtrf_MNI152_2mm, imagetype=3)
    affined_np = ants.apply_transforms(atlas, bold_img, fwdtrf_MNI152_2mm, imagetype=3).numpy()
    # print(sys.getrefcount(affined_img))
    # del affined_img
    toc = time.time()
    print(toc - tic)
    # gc.collect()
    # voxelmorph warp
    tic = time.time()
    warped_np = np.zeros(shape=(*atlas.shape, n_frame), dtype=np.float32)
    with tf.device(device):
        transform = vxm.networks.Transform(atlas.shape, interp_method='linear', nb_feats=1)
        # for idx in range(affined_np.shape[3]):
        #     frame_np = affined_np[:, :, :, idx]
        #     frame_np = frame_np[..., np.newaxis]
        #     frame_np = frame_np[np.newaxis, ...]
        #
        #     moved = transform.predict([frame_np, deform])
        #     warped_np[:, :, :, idx] = moved.squeeze()
        tf_dataset = tf.data.Dataset.from_tensor_slices(np.transpose(affined_np, (3, 0, 1, 2)))
        del affined_np
        batch_size = 16
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
            print(f'batch: {idx}')
        del transform
        del tf_dataset
        del moved
        del moved_data
    toc = time.time()
    print(toc - tic)

    # affine to MNI152
    tic = time.time()
    origin = (*atlas.origin, bold_origin[3])
    spacing = (*atlas.spacing, bold_spacing[3])
    direction = bold_direction.copy()
    direction[:3, :3] = atlas.direction

    warped_img = ants.from_numpy(warped_np, origin=origin, spacing=spacing, direction=direction)
    del warped_np
    moved_img = ants.apply_transforms(MNI152_2mm, warped_img, fwdtrf_atlas2MNI152_2mm, imagetype=3)
    del warped_img
    moved_np = moved_img.numpy()
    del moved_img
    toc = time.time()
    print(toc - tic)

    # save
    origin = (*MNI152_2mm.origin, bold_origin[3])
    spacing = (*MNI152_2mm.spacing, bold_spacing[3])
    direction = bold_direction.copy()
    direction[:3, :3] = MNI152_2mm.direction
    warped_bold_img = ants.from_numpy(moved_np, origin=origin, spacing=spacing, direction=direction)
    del moved_np
    if verbose:
        ants.image_write(warped_bold_img, warped_file)
    return warped_bold_img


def warp_bold_2mm(subj_func_path, subj, workdir, norm_file, bold_file, reg_file, save_file, verbose=False):
    bold_t1_file = subj_func_path / f'{subj}_native_t1_2mm.nii.gz'
    # native_bold_to_T1_2mm(residual_file, subj, subj_t1_file, register_dat_file, resid_t1_file)
    bold_t1 = native_bold_to_T1_2mm_ants(bold_file, subj, norm_file, reg_file, bold_t1_file, workdir,
                                         verbose=verbose)
    warp_file = subj_func_path / f'sub-{subj}_warp.nii.gz'
    affine_file = subj_func_path / f'sub-{subj}_affine.mat'
    warped_file = subj_func_path / f'sub-{subj}_MNI2mm.nii.gz'

    warped_img = vxm_warp_bold_2mm(bold_t1, affine_file, warp_file, warped_file, verbose=verbose)
    # bold_smooth_6(warped_file, smoothed_file)
    bold_smooth_6_ants(warped_img, save_file, verbose=True)
    # exit()
    # gc.collect()


def dimstr2dimno(dimstr):
    if 'x' in dimstr:
        return 0

    if 'y' in dimstr:
        return 1

    if 'z' in dimstr:
        return 2


def swapdim(infile, a, b, c, outfile):
    '''
    infile  - str. Path to file to read and swap dimensions of.
    a       - str. New x dimension.
    b       - str. New y dimension.
    c       - str. New z dimension.
    outfile - str. Path to file to create.

    Returns None.
    '''

    # Read original file.
    img = nib.load(infile)

    # Build orientation matrix.
    ornt = np.zeros((3, 2))
    order_strs = [a, b, c]
    dim_order = list(map(dimstr2dimno, order_strs))
    i_dim = np.argsort(dim_order)
    for i, dim in enumerate(i_dim):
        ornt[i, 1] = -1 if '-' in order_strs[dim] else 1

    ornt[:, 0] = i_dim

    # Transform and save.
    newimg = img.as_reoriented(ornt)
    nib.save(newimg, outfile)


@timing_func
def bold_skip_reorient(preprocess_dir, subj):
    runs = sorted([d.name for d in (preprocess_dir / subj / 'bold').iterdir() if d.is_dir()])
    for run in runs:
        bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest.nii.gz'
        skip_bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest_skip.nii.gz'
        # skip 0 frame
        sh.mri_convert('-i', bold_file, '-o', skip_bold_file, _out=sys.stdout)
        # reorient
        reorient_bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest_reorient_skip.nii.gz'
        swapdim(str(skip_bold_file), 'x', '-y', 'z', str(reorient_bold_file))


@timing_func
def preprocess_common(preprocess_dir, subj):
    bold_skip_reorient(preprocess_dir, subj)
    runs = sorted([d.name for d in (preprocess_dir / subj / 'bold').iterdir() if d.is_dir()])
    for run in runs:
        src_bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest_reorient_skip.nii.gz'
        dst_bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip.nii.gz'
        os.rename(src_bold_file, dst_bold_file)

    # stc
    input_fname = f'{subj}_bld_rest_reorient_skip'
    output_fname = f'{subj}_bld_rest_reorient_skip_faln'
    shargs = [
        '-s', subj,
        '-d', preprocess_dir,
        '-fsd', 'bold',
        '-so', 'odd',
        '-ngroups', 1,
        '-i', input_fname,
        '-o', output_fname,
        '-nolog']
    sh.stc_sess(*shargs, _out=sys.stdout)

    # mk_template
    shargs = [
        '-s', subj,
        '-d', preprocess_dir,
        '-fsd', 'bold',
        '-funcstem', f'{subj}_bld_rest_reorient_skip_faln',
        '-nolog']
    sh.mktemplate_sess(*shargs, _out=sys.stdout)

    # mc
    shargs = [
        '-s', subj,
        '-d', preprocess_dir,
        '-per-session',
        '-fsd', 'bold',
        '-fstem', f'{subj}_bld_rest_reorient_skip_faln',
        '-fmcstem', f'{subj}_bld_rest_reorient_skip_faln_mc',
        '-nolog']
    sh.mc_sess(*shargs, _out=sys.stdout)

    # register
    for run in runs:
        mov_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'
        reg_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.register.dat'
        shargs = [
            '--bold',
            '--s', subj,
            '--mov', mov_file,
            '--reg', reg_file]
        sh.bbregister(*shargs, _out=sys.stdout)

    # mk_brainmask
    recon_dir = Path(os.environ['SUBJECTS_DIR'])
    for run in runs:
        func_path = preprocess_dir / subj / 'bold' / run / f'{subj}.func.aseg.nii'
        mov_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'
        reg_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.register.dat'
        shargs = [
            '--seg', recon_dir / subj / 'mri/aparc+aseg.mgz',
            '--temp', mov_file,
            '--reg', reg_file,
            '--o', func_path]
        sh.mri_label2vol(*shargs, _out=sys.stdout)

        wm_path = preprocess_dir / subj / 'bold' / run / f'{subj}.func.wm.nii.gz'
        shargs = [
            '--i', func_path,
            '--wm',
            '--erode', 1,
            '--o', wm_path]
        sh.mri_binarize(*shargs, _out=sys.stdout)

        vent_path = preprocess_dir / subj / 'bold' / run / f'{subj}.func.ventricles.nii.gz'
        shargs = [
            '--i', func_path,
            '--ventricles',
            '--o', vent_path]
        sh.mri_binarize(*shargs, _out=sys.stdout)

        mask_path = preprocess_dir / subj / 'bold' / run / f'{subj}.brainmask.nii.gz'
        shargs = [
            '--reg', reg_file,
            '--targ', recon_dir / subj / 'mri/brainmask.mgz',
            '--mov', mov_file,
            '--inv',
            '--o', mask_path]
        sh.mri_vol2vol(*shargs, _out=sys.stdout)

        binmask_path = preprocess_dir / subj / 'bold' / run / f'{subj}.brainmask.bin.nii.gz'
        shargs = [
            '--i', mask_path,
            '--o', binmask_path,
            '--min', 0.0001]
        sh.mri_binarize(*shargs, _out=sys.stdout)


def setenv_smooth_downsampling():
    subjects_dir = Path(os.environ['SUBJECTS_DIR'])
    fsaverage6_dir = subjects_dir / 'fsaverage6'
    if not fsaverage6_dir.exists():
        src_fsaverage6_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage6'
        os.symlink(src_fsaverage6_dir, fsaverage6_dir)

    fsaverage5_dir = subjects_dir / 'fsaverage5'
    if not fsaverage5_dir.exists():
        src_fsaverage5_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage5'
        os.symlink(src_fsaverage5_dir, fsaverage5_dir)

    fsaverage4_dir = subjects_dir / 'fsaverage4'
    if not fsaverage4_dir.exists():
        src_fsaverage4_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage4'
        os.symlink(src_fsaverage4_dir, fsaverage4_dir)


@timing_func
def smooth_downsampling(preprocess_dir, bold_path, bldrun, subject):
    logs_path = preprocess_dir / subject / 'logs'
    logs_path.mkdir(exist_ok=True)

    surf_path = preprocess_dir / subject / 'surf'
    surf_path.mkdir(exist_ok=True)

    bldrun_path = bold_path / bldrun
    reg_name = '%s_bld_rest_reorient_skip_faln_mc.register.dat' % (subject)
    reg_path = bldrun_path / reg_name

    resid_name = '%s_bld_rest_reorient_skip_faln' % (subject)
    resid_name += '_mc_g1000000000_bpss_resid.nii.gz'
    resid_path = bldrun_path / resid_name

    for hemi in ['lh', 'rh']:
        fs6_path = sp.indi_to_fs6(surf_path, subject, resid_path, reg_path, hemi)
        sm6_path = sp.smooth_fs6(fs6_path, hemi)
        sp.downsample_fs6_to_fs4(sm6_path, hemi)


@timing_func
def preprocess_rest(layout, bids_bolds, preprocess_dir, subj):
    fcmri_dir = preprocess_dir / subj / 'fcmri'
    fcmri_dir.mkdir(exist_ok=True)
    bold_dir = preprocess_dir / subj / 'bold'

    for idx, bids_bold in enumerate(bids_bolds):
        entities = dict(bids_bold.entities)
        if 'RepetitionTime' in entities:
            TR = entities['RepetitionTime']
        else:
            bold = ants.image_read(bids_bold.path)
            TR = bold.spacing[3]
        run = f'{idx + 1:03}'
        mc_path = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'
        gauss_path = gauss_nifti(str(mc_path), 1000000000)

        # bandpass_filtering
        bpss_path = bandpass_nifti(gauss_path, TR)

        # compile_regressors, regression
        all_regressors_path = compile_regressors(preprocess_dir, bold_dir, run, subj, fcmri_dir, bpss_path)
        regression(bpss_path, all_regressors_path)

        # smooth_downsampling
        # smooth_downsampling(preprocess_dir, bold_dir, run, subj)


@timing_func
def preprocess(layout, bids_bolds, subj, deepprep_subj_path, preprocess_dir):
    subj_bold_dir = preprocess_dir / f'sub-{subj}' / 'bold'
    subj_bold_dir.mkdir(parents=True, exist_ok=True)
    for idx, bids_bold in enumerate(bids_bolds):
        bids_file = Path(bids_bold.path)
        run = f'{idx + 1:03}'
        (subj_bold_dir / run).mkdir(exist_ok=True)
        shutil.copy(bids_file, subj_bold_dir / run / f'sub-{subj}_bld{run}_rest.nii.gz')
    preprocess_common(preprocess_dir, f'sub-{subj}')
    preprocess_rest(layout, bids_bolds, preprocess_dir, f'sub-{subj}')
    setenv_smooth_downsampling()
    for idx, bids_bold in enumerate(bids_bolds):
        run = f"{idx + 1:03}"
        entities = dict(bids_bold.entities)
        subj = entities['subject']
        file_prefix = Path(bids_bold.path).name.replace('.nii.gz', '')
        if 'session' in entities:
            subj_func_path = deepprep_subj_path / f"ses-{entities['session']}" / 'func'
            subj_surf_path = deepprep_subj_path / f"ses-{entities['session']}" / 'surf'
        else:
            subj_func_path = deepprep_subj_path / 'func'
            subj_surf_path = deepprep_subj_path / 'surf'
        subj_surf_path.mkdir(exist_ok=True)
        src_resid_file = subj_bold_dir / run / f'sub-{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
        dst_resid_file = subj_func_path / f'{file_prefix}_resid.nii.gz'
        shutil.copy(src_resid_file, dst_resid_file)
        src_mc_file = subj_bold_dir / run / f'sub-{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'
        dst_mc_file = subj_func_path / f'{file_prefix}_mc.nii.gz'
        shutil.copy(src_mc_file, dst_mc_file)
        src_reg_file = subj_bold_dir / run / f'sub-{subj}_bld_rest_reorient_skip_faln_mc.register.dat'
        dst_reg_file = subj_func_path / f'{file_prefix}_bbregister.register.dat'
        shutil.copy(src_reg_file, dst_reg_file)

        # smooth_downsampling
        for hemi in ['lh', 'rh']:
            fs6_path = sp.indi_to_fs6(subj_surf_path, f'sub-{subj}', dst_resid_file, dst_reg_file, hemi)
            sm6_path = sp.smooth_fs6(fs6_path, hemi)
            sp.downsample_fs6_to_fs4(sm6_path, hemi)


def parse_args():
    def _drop_sub(value):
        return value[4:] if value.startswith("sub-") else value

    parser = argparse.ArgumentParser()
    parser.add_argument('--bd', required=True, help='directory of bids type')
    parser.add_argument('--fsd', default=os.environ.get('FREESURFER_HOME'),
                        help='Output directory $FREESURFER_HOME (pass via environment or here)')
    parser.add_argument("-t", "--task", action='store', nargs='+',
                        help='a space delimited list of tasks identifiers or a single task')
    parser.add_argument("-p", "--preprocess", required=True, help='choose the pre-processing method(rest or task)')
    parser.add_argument("-s", "--subject", action="store", nargs="+", type=_drop_sub,
                        help="a space delimited list of subject identifiers or a single "
                             "identifier (the sub- prefix can be removed)")
    args = parser.parse_args()
    args_dict = vars(args)

    if args.fsd is None:
        args_dict['fsd'] = '/usr/local/freesurfer'

    return argparse.Namespace(**args_dict)


if __name__ == '__main__':
    args = parse_args()

    data_path = Path(args.bd)
    preprocess_method = args.preprocess
    if preprocess_method in ['rest', 'task']:
        print(f'preprocess method : {preprocess_method}')
    else:
        print(f'preprocess method error!')
        exit()
    devices = tf.config.list_physical_devices('GPU')

    layout = bids.BIDSLayout(str(data_path), derivatives=False)
    if args.subject is None:
        subjs = sorted(layout.get_subjects())
    else:
        subjs = args.subject

    # DeepPrep dataset_description    devices = tf.config.list_physical_devices('GPU')
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep'
    derivative_deepprep_path.mkdir(exist_ok=True)
    dataset_description_file = derivative_deepprep_path / 'dataset_description.json'
    if not os.path.exists(dataset_description_file):
        dataset_description = dict()
        dataset_description['Name'] = 'DeepPrep Outputs'
        dataset_description['BIDSVersion'] = '1.4.0'
        dataset_description['DatasetType'] = 'derivative'
        dataset_description['GeneratedBy'] = [{'Name': 'deepprep', 'Version': '0.0.1'}]

        with open(dataset_description_file, 'w') as jf:
            json.dump(dataset_description, jf, indent=4)

    set_envrion()
    freesurfer_subjects_path = derivative_deepprep_path / 'Recon'
    os.environ['SUBJECTS_DIR'] = str(freesurfer_subjects_path)
    atlas_type = 'MNI152_T1_2mm'
    for subj in subjs:

        deepprep_subj_path = derivative_deepprep_path / f'sub-{subj}'
        deepprep_subj_path.mkdir(exist_ok=True)

        tmpdir = deepprep_subj_path / 'tmp'
        tmpdir.mkdir(exist_ok=True)
        # T1 to MNI152 2mm
        model_file = Path(__file__).parent / 'model' / 'voxelmorph' / atlas_type / 'model.h5'
        norm_file = freesurfer_subjects_path / f'sub-{subj}' / 'mri' / 'norm.mgz'
        trf_file = tmpdir / f'sub-{subj}_affine.mat'
        warp_file = tmpdir / f'sub-{subj}_warp.nii.gz'
        warped_file = tmpdir / f'sub-{subj}_warped.nii.gz'
        vxm_registraion(atlas_type, model_file, tmpdir, norm_file, trf_file, warp_file, warped_file)

        for task in args.task:
            # temp dir
            workdir = deepprep_subj_path / 'tmp' / f'task-{task}'  # issues#1
            workdir.mkdir(exist_ok=True)

            sess = layout.get_session(subject=subj)
            if len(sess) == 0:
                subj_func_path = deepprep_subj_path / 'func'
                subj_func_path.mkdir(exist_ok=True)
                shutil.copy(trf_file, subj_func_path / f'sub-{subj}_affine.mat')
                shutil.copy(warp_file, subj_func_path / f'sub-{subj}_warp.nii.gz')
                shutil.copy(warped_file, subj_func_path / f'sub-{subj}_warped.nii.gz')
            else:
                for ses in sess:
                    if args.task is None:
                        bids_bolds = layout.get(subject=subj, session=ses, suffix='bold', extension='.nii.gz')
                    else:
                        bids_bolds = layout.get(subject=subj, session=ses, task=task, suffix='bold',
                                                extension='.nii.gz')
                    if len(bids_bolds) == 0:
                        continue
                    subj_func_path = deepprep_subj_path / f'ses-{ses}' / 'func'
                    subj_func_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy(trf_file, subj_func_path / f'sub-{subj}_affine.mat')
                    shutil.copy(warp_file, subj_func_path / f'sub-{subj}_warp.nii.gz')
                    shutil.copy(warped_file, subj_func_path / f'sub-{subj}_warped.nii.gz')
            if args.task is None:
                bids_bolds = layout.get(subject=subj, suffix='bold', extension='.nii.gz')
            else:
                bids_bolds = layout.get(subject=subj, task=args.task, suffix='bold', extension='.nii.gz')
            # native bold preprocess
            preprocess(layout, bids_bolds, subj, deepprep_subj_path, workdir)

            for bids_bold in bids_bolds:
                entities = dict(bids_bold.entities)
                subj = entities['subject']
                file_prefix = Path(bids_bold.path).name.replace('.nii.gz', '')
                if 'session' in entities:
                    ses = entities['session']
                    subj_func_path = deepprep_subj_path / f'ses-{ses}' / 'func'
                else:
                    subj_func_path = deepprep_subj_path / 'func'

                # native bold to MNI152 2mm
                if preprocess_method == 'rest':
                    bold_file = subj_func_path / f'{file_prefix}_resid.nii.gz'
                    save_file = subj_func_path / f'{file_prefix}_resid_MIN2mm_sm6.nii.gz'
                else:
                    bold_file = subj_func_path / f'{file_prefix}_mc.nii.gz'
                    save_file = subj_func_path / f'{file_prefix}_mc_MIN2mm.nii.gz'

                reg_file = subj_func_path / f'{file_prefix}_bbregister.register.dat'
                bold_t1_file = subj_func_path / f'{subj}_native_t1_2mm.nii.gz'
                bold_t1 = native_bold_to_T1_2mm_ants(bold_file, subj, norm_file, reg_file, bold_t1_file, workdir,
                                                     verbose=False)

                warp_file = subj_func_path / f'sub-{subj}_warp.nii.gz'
                affine_file = subj_func_path / f'sub-{subj}_affine.mat'
                warped_file = subj_func_path / f'sub-{subj}_MNI2mm.nii.gz'
                warped_img = vxm_warp_bold_2mm(bold_t1, affine_file, warp_file, warped_file, verbose=False)
                if preprocess_method == 'rest':
                    temp_file = workdir / f'{file_prefix}_MNI2mm_sm6_temp.nii.gz'
                    bold_smooth_6_ants(warped_img, save_file, temp_file, bold_file, verbose=True)
                else:
                    temp_file = workdir / f'{file_prefix}_MNI2mm_temp.nii.gz'
                    save_bold(warped_img, temp_file, bold_file, save_file)
            # shutil.rmtree(workdir)
