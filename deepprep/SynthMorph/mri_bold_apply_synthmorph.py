#!/usr/bin/env python3
import os
import argparse
import tensorflow.keras.backend as K
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm
from pathlib import Path
import math


def vxm_batch_transform(vol, loc_shift,
                        batch_size=None, interp_method='linear', indexing='ij', fill_value=None):
    """ apply transform along batch. Compared to _single_transform, reshape inputs to move the
    batch axis to the feature/channel axis, then essentially apply single transform, and
    finally reshape back. Need to know/fix batch_size.

    Important: loc_shift is currently implemented only for shape [B, *new_vol_shape, C, D].
        to implement loc_shift size [B, *new_vol_shape, D] (as transform() supports),
        we need to figure out how to deal with the second-last dimension.

    Other Notes:
    - we couldn't use ne.utils.flatten_axes() because that computes the axes size from tf.shape(),
      whereas we get the batch size as an input to avoid 'None'

    Args:
        vol (Tensor): volume with size vol_shape or [B, *vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [B, *new_vol_shape, C, D]
            where C is the number of channels, and D is the dimentionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # input management
    ndim = len(vol.shape) - 2
    assert ndim in range(1, 4), 'Dimension {} can only be in [1, 2, 3]'.format(ndim)
    vol_shape_tf = tf.shape(vol)

    if batch_size is None:
        batch_size = vol_shape_tf[0]
        assert batch_size is not None, 'batch_transform: provide batch_size or valid Tensor shape'
    else:
        tf.debugging.assert_equal(vol_shape_tf[0],
                                  batch_size,
                                  message='Tensor has wrong batch size '
                                          '{} instead of {}'.format(vol_shape_tf[0], batch_size))
    BC = batch_size * vol.shape[-1]

    assert len(loc_shift.shape) == ndim + 3, \
        'vol dim {} and loc dim {} are not appropriate'.format(ndim + 2, len(loc_shift.shape))
    assert loc_shift.shape[-1] == ndim, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(ndim, vol.shape[:-1], loc_shift.shape[-1])

    vol_reshape = K.permute_dimensions(vol, list(range(1, ndim + 2)) + [0])
    vol_reshape = K.reshape(vol_reshape, list(vol.shape[1:ndim + 1]) + [BC])

    loc_reshape = K.permute_dimensions(loc_shift, list(range(1, ndim + 2)) + [0] + [ndim + 2])
    loc_reshape_shape = list(loc_shift.shape[1:ndim + 1]) + [BC] + [loc_shift.shape[ndim + 2]]
    loc_reshape = K.reshape(loc_reshape, loc_reshape_shape)

    vol_trf = vxm.utils.transform(vol_reshape, loc_reshape,
                        interp_method=interp_method, indexing=indexing, fill_value=fill_value)

    vol_trf_reshape = tf.expand_dims(vol_trf, axis=-2)
    return K.permute_dimensions(vol_trf_reshape, [ndim + 1] + list(range(ndim + 1)))


def batch_transform(image, trans, normalize=False):
    if isinstance(image, nib.filebasedimages.FileBasedImage):
        image = image.get_fdata(dtype=np.float32)
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=-1)
    trans = tf.expand_dims(trans, axis=0)
    trans = tf.expand_dims(trans, axis=-2)
    sliced_im_i = tf.transpose(image, perm=(3, 0, 1, 2))
    if tf.rank(sliced_im_i) == 4:
        sliced_im_i = sliced_im_i[..., tf.newaxis]

    sliced_trans_i = tf.tile(trans, [sliced_im_i.shape[0], 1, 1, 1, 1, 1])
    out_i = vxm_batch_transform(
        sliced_im_i, sliced_trans_i, batch_size=sliced_im_i.shape[0], fill_value=0)
    if normalize:
        out_i -= tf.reduce_min(out_i)
        out_i /= tf.reduce_max(out_i)
    return out_i[tf.newaxis, ...]


def bold_save(path, fframe_bold_path, num, data, affine, header, ori_header, dtype=None):
    """Save image file.

    Helper function for saving a spatial image using NiBabel. Removes singleton
    dimensions and sets the data type, world matrix, and header units.

    Parameters
    ----------
    path : str
        File system path to write the image to.
    data : NiBabel image or NumPy array or TensorFlow tensor.
        Image data to save. Except for the data type, the header information of
        a NiBabel image object will be ignored.
    affine : (4, 4) array-like
        World matrix of the image, describing the voxel-to-RAS transform.
    dtype : None or dtype, optional
        Output data type. None means the original type of the image buffer.

    """
    # Use NiBabel's caching functionality to avoid re-reading from disk.
    if isinstance(data, nib.filebasedimages.FileBasedImage):
        if dtype is None:
            dtype = data.dataobj.dtype
        data = data.get_fdata(dtype=np.float32)

    data = np.squeeze(data)
    data = np.asarray(data, int)
    data = data.transpose((1, 2, 3, 0))
    # Use Nifti1Image instead of MGHImage for FP64 support. Set units to avoid
    # warnings when reading with FreeSurfer.
    out = nib.Nifti1Image(data, affine=affine, header=ori_header)
    nib.save(out, filename=path)
    if num == 0:
        fframe_out = nib.Nifti1Image(data[..., 0], affine=affine, header=ori_header)
        nib.save(fframe_out, filename=fframe_bold_path)


def save(path, dat, affine, dtype=None):
    """Save image file.

    Helper function for saving a spatial image using NiBabel. Removes singleton
    dimensions and sets the data type, world matrix, and header units.

    Parameters
    ----------
    path : str
        File system path to write the image to.
    dat : NiBabel image or NumPy array or TensorFlow tensor.
        Image data to save. Except for the data type, the header information of
        a NiBabel image object will be ignored.
    affine : (4, 4) array-like
        World matrix of the image, describing the voxel-to-RAS transform.
    dtype : None or dtype, optional
        Output data type. None means the original type of the image buffer.

    """
    # Use NiBabel's caching functionality to avoid re-reading from disk.
    if isinstance(dat, nib.filebasedimages.FileBasedImage):
        if dtype is None:
            dtype = dat.dataobj.dtype
        dat = dat.get_fdata(dtype=np.float32)

    dat = np.squeeze(dat)
    dat = np.asarray(dat, dtype)
    # Use Nifti1Image instead of MGHImage for FP64 support. Set units to avoid
    # warnings when reading with FreeSurfer.
    out = nib.Nifti1Image(dat, affine)
    out.header.set_xyzt_units(xyz='mm', t='sec')
    nib.save(out, filename=path)


def read_batch_nifty_file(next_datafiles):
    batch_size = next_datafiles.shape[0]
    data_array = None
    for i, file_path in enumerate(next_datafiles):
        img = nib.load(str(file_path.numpy(), encoding="utf-8"))
        data = img.get_fdata()
        if data_array is None:
            data_array = np.zeros((data.shape[0], data.shape[1], data.shape[2], batch_size))
        data_array[:, :, :, i] = data
    return data_array


p = argparse.ArgumentParser()
p.add_argument('moving', type=str, metavar='MOVING')
p.add_argument('fixed', type=str, metavar='FIXED')
p.add_argument('-j', '--threads', type=int)
p.add_argument('-fbo', '--fframe_bold_out', type=str, metavar='BOLD_OUT')
p.add_argument('-tv', '--trans_vox', type=str, metavar='TRANS VOXEL')
p.add_argument('-ob', '--orig_bold', type=str, metavar='ORIGINAL BOLD')
p.add_argument('-up', '--upsampled_path', type=str, metavar='UPSAMPLED_BOLD PATH')
p.add_argument('-bs', '--batch_size', type=int, metavar='BATCH SIZE')
p.add_argument('-ai', '--anat_input', type=str, default=None, metavar='ANAT INPUT IN T1w SPACE')
p.add_argument('-ao', '--anat_output', type=str, default=None, metavar='ANAT OUTPUT IN STAND SPACE')
arg = p.parse_args()

# Setup.
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

if arg.threads:
    tf.config.threading.set_inter_op_parallelism_threads(arg.threads)
    tf.config.threading.set_intra_op_parallelism_threads(arg.threads)

# Input data.
mov = nib.load(arg.moving)
fix = nib.load(arg.fixed)
assert len(mov.shape) == len(fix.shape) == 3, 'input images not single volumes'

trans_vox = tf.convert_to_tensor(np.load(f'{arg.trans_vox}')['arr_0'])
if arg.anat_input is not None:
    anat_input = nib.load(arg.anat_input)
    out_anat = batch_transform(anat_input, trans=trans_vox)
    save(arg.anat_output, out_anat, fix.affine)
else:
    orig_bold = nib.load(arg.orig_bold)
    batch_size = arg.batch_size
    transform_save_path = Path(arg.upsampled_path).parent / 'transform'
    transform_save_path.mkdir(parents=True, exist_ok=True)
    upsamping_file_list = sorted([os.path.join(arg.upsampled_path, f) for f in os.listdir(arg.upsampled_path) if f.endswith('.nii.gz')])
    datafiles = tf.data.Dataset.from_tensor_slices(upsamping_file_list)
    dataset = datafiles.batch(batch_size)
    iterator = iter(dataset)
    iter_nums = math.ceil(len(upsamping_file_list) / batch_size)
    for i in range(iter_nums):
        next_datafiles = iterator.get_next()
        bold = read_batch_nifty_file(next_datafiles)
        out_bold = batch_transform(bold, trans=trans_vox)
        bold_out = f'{transform_save_path}/t00{i}.nii.gz'
        bold_save(bold_out, arg.fframe_bold_out, num=i, data=out_bold, affine=fix.affine, header=fix.header, ori_header=orig_bold.header, dtype=mov.dataobj.dtype)
    assert os.path.exists(arg.fframe_bold_out), f'{arg.fframe_bold_out}'
