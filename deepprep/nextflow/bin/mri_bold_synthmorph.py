#!/usr/bin/env python3

# SynthMorph registration script using TensorFlow.

import os
import sys
import shutil
import textwrap
import argparse
import tensorflow.keras.backend as K
import time

import tensorflow as tf

# Settings.
default = {
    'model': 'deform',
    'smooth': 1,
    'extent': 256,
}
choices = {
    'model': ('deform', 'affine', 'rigid'),
    'smooth': (1,),
    'extent': (192, 256),
}
weights = {
    'deform': 'synthmorph_deform{smooth}.h5',
    'affine': 'synthmorph_affine.h5',
    'rigid': 'synthmorph_rigid.h5',
}


def rewrap(text, width=None, hard='\t\n', hard_indent=0):
    """Rewrap text so that lines fill the available horizontal space.

    Reformats individual paragraphs of a text body, considering as a paragraph
    subsequent lines with identical indentation. For unspecified width, the
    function will attempt to determine the extent of the current terminal.

    Parameters
    ----------
    text : str
        Input text to rewrap.
    width : int, optional
        Maximum line width. None means the width of the terminal as determined
        by `textwrap`, defaulting to 80 characters for background processes.
    hard : str, optional
        String interpreted as a hard break when terminating a line. Useful for
        inserting a line break without changing the indentation level. Must end
        with a line break and will be removed from the output text.
    hard_indent : int, optional
        Number of additional whitespace characters by which to indent the lines
        following a hard break. See `hard`.

    Returns
    -------
    out : str
        Reformatted text.

    """
    # Inputs.
    if width is None:
        width = shutil.get_terminal_size().columns
    lines = text.splitlines(keepends=True)

    # Merge lines to paragraphs.
    pad = []
    pad_hard = []
    par = []
    for i, line in enumerate(lines):
        ind = len(line) - len(line.lstrip())
        if i == 0 or ind != pad[-1] or lines[i - 1].endswith(hard):
            par.append('')
            pad.append(ind)
            pad_hard.append(ind)

        if line.endswith(hard):
            line = line.replace(hard, '\n')
            pad_hard[-1] += hard_indent
        par[-1] += line[ind:]

    # Reformat paragraphs.
    for i, _ in enumerate(par):
        par[i] = textwrap.fill(
            par[i], width,
            initial_indent=' ' * pad[i], subsequent_indent=' ' * pad_hard[i],
        )

    return '\n'.join(par)


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

def bold_save(path, data, affine, header, dtype=None):
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
    data = np.asarray(data, dtype)
    data = data.transpose((1, 2, 3, 0))
    # Use Nifti1Image instead of MGHImage for FP64 support. Set units to avoid
    # warnings when reading with FreeSurfer.
    out = nib.Nifti1Image(data, affine=affine, header=header)
    out.header.set_xyzt_units(xyz='mm', t='sec')
    nib.save(out, filename=path)


def ori_to_ori(old, new='LIA', old_shape=None, zero_center=False):
    """Construct transform between canonical image orientations.

    Constructs a matrix transforming coordinates from a voxel space with a new
    predominant anatomical axis orientation to an old orientation by swapping
    and flipping axes. The transform operates in zero-based index space unless
    the space is specified to be zero-centered.

    Parameters
    ----------
    old : str or NiBabel image or (4, 4) array-like
        Old canonical image orientation as a three-letter string (see
        `orientation`), a NiBabel image object, or a voxel-to-RAS matrix.
    new : str, optional
        New canonical image orientation. See `old`.
    old_shape : (3,) array-like, optional
        Spatial shape of the old image. The old shape must be specified if the
        old orientation is not a NiBabel image, ignored otherwise.
    zero_center : bool, optional
        Return a transform operating in zero-centered index space. If False,
        the transform will operate in zero-based index space.

    Returns
    -------
    mat : (4, 4) NumPy array
        Transform from the new to the old canonical voxel space.

    """

    def extract_ori(x):
        if isinstance(x, nib.filebasedimages.FileBasedImage):
            x = x.affine
        if isinstance(x, np.ndarray):
            return nib.orientations.io_orientation(x)
        if isinstance(x, str):
            return nib.orientations.axcodes2ornt(x)

    # Old shape.
    if zero_center:
        old_shape = (1, 1, 1)
    if old_shape is None:
        old_shape = old.shape

    # Transform from new to old coordinates.
    old = extract_ori(old)
    new = extract_ori(new)
    new_to_old = nib.orientations.ornt_transform(old, new)
    return nib.orientations.inv_ornt_aff(new_to_old, old_shape)


def net_to_vox(im, shape=None):
    """Construct transform from network space to the voxel space of an image.

    Construct a coordinate transform from the space the network will operate in
    to the original zero-based image index space. This network space will have
    isotropic 1-mm voxels, gross LIA (left-inferior-anterior) orientation, and
    be centered on the field of view. It thus is a scaled and shifted voxel
    space, not world space.

    Parameters
    ----------
    im : str or NiBabel image
        Input image to construct the transform for.
    shape : (3,) array-like
        Desired spatial shape of the network space to construct.

    Returns
    -------
    mat : (4, 4) NumPy array
        Transform from network to indexed input-image space.

    """
    if isinstance(im, str):
        im = nib.load(im)

    # Gross LIA to predominant anatomical orientation of input image.
    assert isinstance(im, nib.filebasedimages.FileBasedImage)
    lia_to_ori = ori_to_ori(im, new='LIA', old_shape=shape)

    # Scaling from millimeter to input voxels.
    vox_size = np.sqrt(np.sum(im.affine[:-1, :-1] ** 2, axis=0))
    scale = np.diag((*1 / vox_size, 1))

    # Shift from cen
    shift = np.eye(4)
    shift[:-1, -1] = 0.5 * (im.shape - shape / vox_size)

    # Composite transform.
    return shift @ scale @ lia_to_ori


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

    # reshape vol [B, *vol_shape, C] --> [*vol_shape, C * B]
    vol_reshape = K.permute_dimensions(vol, list(range(1, ndim + 2)) + [0])
    vol_reshape = K.reshape(vol_reshape, list(vol.shape[1:ndim + 1]) + [BC])

    # reshape loc_shift [B, *loc_shift_shape, C, D] --> [*loc_shift_shape, C * B, D]
    loc_reshape = K.permute_dimensions(loc_shift, list(range(1, ndim + 2)) + [0] + [ndim + 2])
    loc_reshape_shape = list(loc_shift.shape[1:ndim + 1]) + [BC] + [loc_shift.shape[ndim + 2]]
    loc_reshape = K.reshape(loc_reshape, loc_reshape_shape)

    # transform (output is [*loc_shift_shape, C*B])
    vol_trf = vxm.utils.transform(vol_reshape, loc_reshape,
                        interp_method=interp_method, indexing=indexing, fill_value=fill_value)

    # reshape vol back to [*vol_shape, C, B]
    # new_shape = tf.concat([vol_shape_tf[1:], vol_shape_tf[0:1]], 0)
    # vol_trf_reshape = K.reshape(vol_trf, new_shape)
    vol_trf_reshape = tf.expand_dims(vol_trf, axis=-2)
    # reshape back to [B, *vol_shape, C]
    return K.permute_dimensions(vol_trf_reshape, [ndim + 1] + list(range(ndim + 1)))


def transform(im, trans, shape=None, normalize=False):
    """Apply a spatial transform to image voxel data in N dimensions.

    Apply a transformation matrix operating in zero-based index space or a
    displacement field to an image buffer.

    Parameters
    ----------
    im : NiBabel image or NumPy array or TensorFlow tensor
        Input image to transform.
    trans : array-like
        Transformation to apply to the image. A matrix of shape ``(N, N + 1)``,
        a matrix of shape ``(N + 1, N + 1)``, or a displacement field of shape
        ``(*space, N)``, that is, without batch dimension.
    shape : (N,) array-like, optional
        Output shape used for converting matrices to dense transforms. None
        means the shape of the input image will be used.
    normalize : bool, optional
        Min-max normalize the image data into the interval [0, 1].

    Returns
    -------
    out : (1, ...) float TensorFlow tensor
        Transformed image with a leading singleton batch and a trailing
        feature dimension.

    """
    if isinstance(im, nib.filebasedimages.FileBasedImage):
        im = im.get_fdata(dtype=np.float32)

    # Add singleton feature dimension if needed.
    if tf.rank(im) == 3:
        im = im[..., tf.newaxis]

    # Remove last row of matrix transforms.
    if tf.rank(trans) == 2 and trans.shape[0] == trans.shape[1]:
        trans = trans[:-1, :]

    out = vxm.utils.transform(
        im, trans, fill_value=0, shift_center=False, shape=shape,
    )

    if normalize:
        out -= tf.reduce_min(out)
        out /= tf.reduce_max(out)
    return out[tf.newaxis, ...]


def batch_transform(im, trans, normalize=False):
    if isinstance(im, nib.filebasedimages.FileBasedImage):
        im = im.get_fdata(dtype=np.float32)
        # Add singleton feature dimension if needed.
    # im = im[:,:,:,np.newaxis]
    im = tf.transpose(im, perm=(3, 0, 1, 2))
    if tf.rank(im) == 4:
        im = im[..., tf.newaxis]

    trans = tf.expand_dims(trans, axis=0)
    trans = tf.expand_dims(trans, axis=-2)
    # trans = tf.tile(trans, [im.shape[0], 1, 1, 1, 1, 1])

    last_dim_size = im.shape[0]
    num_splits = last_dim_size // 40
    sliced_im = np.array_split(im, num_splits, axis=0)
    # sliced_trans = np.array_split(trans, num_splits, axis=0)
    print('len_num_splits: ', num_splits)
    frames = 0
    out_arr =np.zeros(im.shape[0] + trans.shape[1:5])
    for i in range(num_splits):
        start_time = time.time()
        sliced_im_i = sliced_im[i]
        sliced_trans_i = tf.tile(trans, [sliced_im_i.shape[0], 1, 1, 1, 1, 1])
        out_i = vxm_batch_transform(
            sliced_im_i, sliced_trans_i, batch_size=sliced_im_i.shape[0], fill_value=0)
        if i == 0:
            # out = out_i
            out_arr[:out_i.shape[0]] = out_i
            frames += out_i.shape[0]
        else:
            # out = np.concatenate([out, out_i], axis=0)
            out_arr[frames:frames+out_i.shape[0]] = out_i
            frames += out_i.shape[0]
        # print('slice_time: ', time.time() - start_time)
        # print('ep/och: ', i)
        # print('-num: ', num_splits-i)
    # print('done loop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    if normalize:
        out_arr -= tf.reduce_min(out_arr)
        out_arr /= tf.reduce_max(out_arr)
    return out_arr[tf.newaxis, ...]


def vm_affine(
        in_shape=None,
        in_model=None,
        num_key=64,
        enc_nf=(256,) * 4,
        dec_nf=(256,) * 0,
        add_nf=(256,) * 4,
        half_res=True,
        rigid=False,
):
    # Inputs.
    if in_model is None:
        source = tf.keras.Input(shape=(*in_shape, 1))
        target = tf.keras.Input(shape=(*in_shape, 1))
        in_model = tf.keras.Model(*[(source, target)] * 2)
    source, target = in_model.outputs[:2]

    in_shape = np.asarray(source.shape[1:-1])
    num_dim = len(in_shape)
    assert num_dim in (2, 3), 'only 2D and 3D supported'

    # Layers.
    down = getattr(tf.keras.layers, f'MaxPool{num_dim}D')()
    up = getattr(tf.keras.layers, f'UpSampling{num_dim}D')()
    act = tf.keras.layers.LeakyReLU(0.2)
    conv = getattr(tf.keras.layers, f'Conv{num_dim}D')
    prop = dict(kernel_size=3, padding='same')

    # Internal U-Net.
    inp = tf.keras.Input(shape=(*in_shape, 1))
    x = down(inp) if half_res else inp

    # Encoder.
    enc = []
    for n in enc_nf:
        x = conv(n, **prop)(x)
        x = act(x)
        enc.append(x)
        x = down(x)

    # Decoder.
    for n in dec_nf:
        x = conv(n, **prop)(x)
        x = act(x)
        x = tf.keras.layers.concatenate([up(x), enc.pop()])

    # Additional convolutions.
    for n in add_nf:
        x = conv(n, **prop)(x)
        x = act(x)

    # Features.
    x = conv(num_key, activation='relu', **prop)(x)
    net = tf.keras.Model(inp, outputs=x)
    key_1 = net(source)
    key_2 = net(target)

    # Barycenters.
    prop = dict(axes=range(1, num_dim + 1), normalize=True, shift_center=True)
    cen_1 = ne.utils.barycenter(key_1, **prop) * in_shape
    cen_2 = ne.utils.barycenter(key_2, **prop) * in_shape

    # Weights.
    axes = range(1, num_dim + 1)
    pow_1 = tf.reduce_sum(key_1, axis=axes)
    pow_2 = tf.reduce_sum(key_2, axis=axes)
    pow_1 /= tf.reduce_sum(pow_1, axis=-1, keepdims=True)
    pow_2 /= tf.reduce_sum(pow_2, axis=-1, keepdims=True)
    weights = pow_1 * pow_2

    # Least squares.
    out = vxm.utils.fit_affine(cen_1, cen_2, weights=weights)
    if rigid:
        out = vxm.utils.affine_matrix_to_params(out)
        out = out[:, :num_dim * (num_dim + 1) // 2]
        out = vxm.layers.ParamsToAffineMatrix(ndims=num_dim)(out)

    return tf.keras.Model(in_model.inputs, out)


def vm_dense(
        in_shape=None,
        input_model=None,
        enc_nf=(256,) * 4,
        dec_nf=(256,) * 4,
        add_nf=(256,) * 4,
        int_steps=7,
        upsample=True,
        half_res=True,
):
    if input_model is None:
        source = tf.keras.Input(shape=(*in_shape, 1))
        target = tf.keras.Input(shape=(*in_shape, 1))
        input_model = tf.keras.Model(*[(source, target)] * 2)
    source, target = input_model.outputs[:2]

    in_shape = np.asarray(source.shape[1:-1])
    num_dim = len(in_shape)
    assert num_dim in (2, 3), 'only 2D and 3D supported'

    down = getattr(tf.keras.layers, f'MaxPool{num_dim}D')()
    up = getattr(tf.keras.layers, f'UpSampling{num_dim}D')()
    act = tf.keras.layers.LeakyReLU(0.2)
    conv = getattr(tf.keras.layers, f'Conv{num_dim}D')
    prop = dict(kernel_size=3, padding='same')

    # Encoder.
    x = tf.keras.layers.concatenate((source, target))
    if half_res:
        x = down(x)
    enc = [x]
    for n in enc_nf:
        x = conv(n, **prop)(x)
        x = act(x)
        enc.append(x)
        x = down(x)

    # Decoder.
    for n in dec_nf:
        x = conv(n, **prop)(x)
        x = act(x)
        x = tf.keras.layers.concatenate([up(x), enc.pop()])

    # Additional convolutions.
    for n in add_nf:
        x = conv(n, **prop)(x)
        x = act(x)

    # Transform.
    x = conv(num_dim, **prop)(x)
    if int_steps > 0:
        x = vxm.layers.VecInt(method='ss', int_steps=int_steps)(x)

    # Rescaling.
    zoom = source.shape[1] // x.shape[1]
    if upsample and zoom > 1:
        x = vxm.layers.RescaleTransform(zoom)(x)

    return tf.keras.Model(input_model.inputs, outputs=x)


# Documentation.
prog = os.path.basename(sys.argv[0])
doc = f'''{prog}

NAME
        {prog} - register a landscape of unprocessed 3D brain images

SYNOPSIS
        {prog} [OPTIONS] MOVING FIXED

DESCRIPTION
        SynthMorph is an easy-to-use deep-learning (DL) tool for brain-specific
        registration of images without preprocessing right off the MRI scanner.
        In contrast to registration methods which are agnostic to the anatomy,
        SynthMorph can distinguish between anatomy of interest and irrelevant
        structures, removing the need for preprocessing to exclude content that
        would otherwise reduce the accuracy of anatomy-specific registration. 

ARGUMENTS
        MOVING
                Moving input image. See IMAGE FORMAT.

        FIXED
                Fixed input image. See IMAGE FORMAT.

OPTIONS
        -o, --moved MOVED
                Moved output image. See IMAGE FORMAT.

        -t, --trans TRANS
                Output transform. A text file for linear or an image file for
                deformable registration, including any initialization. See
                TRANSFORMS.

        -H, --header-only
                Set the output image orientation by adapting the voxel-to-world
                matrix instead of resampling. For linear registration only.

        -m, --model MODEL
                Transformation model {choices['model']}. Defaults to
                '{default['model']}'. Rigid is experimental.

        -i, --init INIT
                Linear transform to initialize with. See TRANSFORMS.

        -j, --threads THREADS
                Number of TensorFlow threads. Defaults to the number of cores.

        -g, --gpu
                Instead of the CPU, use the GPU specified by environment
                variable CUDA_VISIBLE_DEVICES or GPU 0 if unset or empty.

        -s, --smooth SMOOTH
                Regularization parameter for deformable registration
                {choices['smooth']}. Higher values mean smoother displacement
                fields. Defaults to {default['smooth']}.

        -e, --extent EXTENT
                Isotropic extent of the registration space in unit voxels
                {choices['extent']}. Lower values can improve speed and memory
                usage but may crop the anatomy of interest. Defaults to
                {default['extent']}.

        -w, --weights WEIGHTS
                Alternative model weights as an H5 file. The weights have to
                match the architecture of the specified registration model.

        --inspect DIR
                Save model inputs resampled into network space for inspection.
                Files existing in the folder may be overwritten.

        -h, --help
                Print this help text and exit.

IMAGE FORMAT
        The registration networks expect input images of cubic shape with
        isotropic 1-mm voxels min-max normalized into the interval [0, 1]. They
        also assume that the axes of the image-data array have approximate LIA
        orientation (left-inferior-anterior).

        Internally, the model converts images to LIA orientation based on the
        image-to-world matrices in their headers. This conversion assumes HFS
        (head-first-supine) subject positioning in the scanner. In other words,
        the brain must show the correct anatomical orientation in FreeView.

        Acceptable image formats are those supported by NiBabel and include:
        MGH format (.mgz) and NIfTI (.nii.gz, .nii). The model supports
        three-dimensional images with a single frame.

TRANSFORMS
        The output transforms of this script are defined in physical RAS space,
        transforming world coordinates of the fixed image to the moving image.
        Likewise, we will assume that any linear transform passed to initialize
        the registration operates in RAS space. We save linear transforms as a
        4-by-4 matrix in text format and non-linear displacements fields as a
        vector image. See IMAGE FORMAT.

        For converting, concatenating, and applying transforms to other images,
        consider the FreeSurfer tools listed under SEE ALSO.

EXAMPLES
        Deformable registration:
                {prog} -t def.mgz mov.nii.gz fix.nii

        Rigid registration:
                {prog} -m rigid -t rig.txt mov.mgz fix.mgz

        Affine registration updating only the output image header:
                {prog} -m affine --header-only -o out.mgz mov.mgz fix.mgz

        Initialize deformable registration with an affine transform:
                {prog} -i aff.txt -t def.nii.gz mov.mgz fix.mgz

SEE ALSO
        Useful FreeSurfer tools for applying and manipulating transforms
        include: mri_concatenate_lta, mri_concatenate_gcam, mri_warp_convert,
        lta_convert, mri_convert, mri_info.

        Convert a linear transform to FreeSurfer's LTA format:
                lta_convert --src mov.mgz --trg fix.nii.gz --inras aff.txt
                --outlta aff.lta

        Apply an LTA to another image:
                mri_convert -at aff.lta in.mgz out.mgz

        Convert a deformable transform to FreeSurfer's GCAM format (.m3z):
                mri_warp_convert -g mov.mgz --inras def.mgz --outm3z def.m3z

        Apply a GCAM (.m3z) to another image:
                mri_convert -at def.m3z in.nii.gz out.nii.gz

BUGS
        Report bugs to freesurfer@nmr.mgh.harvard.edu or at
        https://github.com/freesurfer/freesurfer/issues.

REFERENCES
        If you use SynthMorph in a publication, please cite:
'''

# References.
ref = '''
Hoffmann M, Billot B, Greve DN, Iglesias JE, Fischl B, Dalca AV\t
SynthMorph: learning contrast-invariant registration without acquired images\t
IEEE Transactions on Medical Imaging, 41 (3), 543-558, 2022\t
https://doi.org/10.1109/TMI.2021.3116879

Website: https://w3id.org/synthmorph
'''
doc += textwrap.indent(ref, prefix=' ' * 8)

# Command-line arguments.
p = argparse.ArgumentParser()
p.add_argument('moving', type=str, metavar='MOVING')
p.add_argument('fixed', type=str, metavar='FIXED')
p.add_argument('-o', '--moved', type=str)
p.add_argument('-t', '--trans', type=str)
p.add_argument('-H', '--header-only', action='store_true')
p.add_argument('-i', '--init', type=str, metavar='TRANS')
p.add_argument('-j', '--threads', type=int)
p.add_argument('-g', '--gpu', action='store_true')
p.add_argument('-s', '--smooth', choices=choices['smooth'], default=default['smooth'])
p.add_argument('-e', '--extent', type=int, choices=choices['extent'], default=default['extent'])
p.add_argument('-m', '--model', choices=choices['model'], default=default['model'])
p.add_argument('-w', '--weights', type=str)
p.add_argument('--inspect', type=str, metavar='OUT_DIR')
p.add_argument('-b', '--bold', type=str, metavar='BOLD')
p.add_argument('-bo', '--bold_out', type=str, metavar='BOLD_OUT')
p.add_argument('-mp', '--model_path', type=str, metavar='MODEL PATH')
p.add_argument('-mc', '--mc', type=str, metavar='TR_info')
p.add_argument('-a', '--apply', type=str, metavar='APPLY_TRANS')
p.add_argument('-ao', '--apply_out', type=str, metavar='APPLY_TRANS_OUT')

if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
    print(rewrap(doc), end='\n\n')
    exit(1)
arg = p.parse_args()

in_shape = (arg.extent,) * 3
is_linear = arg.model in ('affine', 'rigid')
assert arg.moved or arg.trans, 'no output specified with --moved or --trans'

# Third-party imports. Avoid waiting for TensorFlow just for documentation.
import numpy as np
import nibabel as nib
import tensorflow as tf
import neurite as ne
import voxelmorph as vxm
from pathlib import Path

# Setup.
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu if arg.gpu else ''

if arg.threads:
    tf.config.threading.set_inter_op_parallelism_threads(arg.threads)
    tf.config.threading.set_intra_op_parallelism_threads(arg.threads)

# Input data.
mov = nib.load(arg.moving)
fix = nib.load(arg.fixed)
assert len(mov.shape) == len(fix.shape) == 3, 'input images not single volumes'

# Coordinate transforms. We will need these to take the images from their
# native voxel spaces to network space. Voxel and network spaces are different
# for each image. We register in isotropic 1-mm spaces centered on the original
# images. Their axes are aligned with the original voxel data but flipped and
# swapped to gross LIA orientation, which the network will expect.
net_to_mov = net_to_vox(mov, shape=in_shape)
net_to_fix = net_to_vox(fix, shape=in_shape)
mov_to_net = np.linalg.inv(net_to_mov)
fix_to_net = np.linalg.inv(net_to_fix)

# Transforms from and to world space (RAS). There is only one world.
mov_to_ras = mov.affine
fix_to_ras = fix.affine
ras_to_mov = np.linalg.inv(mov_to_ras)
ras_to_fix = np.linalg.inv(fix_to_ras)

# Transforms between zero-centered and zero-based voxel spaces.
ind_to_cen = np.eye(4)
ind_to_cen[:-1, -1] = -0.5 * (np.asarray(in_shape) - 1)
cen_to_ind = np.eye(4)
cen_to_ind[:-1, -1] = +0.5 * (np.asarray(in_shape) - 1)

# Incorporate an initial linear transform operating in RAS. It goes from fixed
# to moving coordinates, so we start with fixed network space on the right.
if arg.init:
    aff = np.loadtxt(arg.init)
    net_to_mov = ras_to_mov @ aff @ fix_to_ras @ net_to_fix

# Take the input images to network space. When saving the moving image with the
# correct voxel-to-RAS matrix after incorporating an initial linear transform,
# an image viewer taking this matrix into account will show an unchanged image.
# However, the network only sees the voxel data, which have been moved.
a = transform(mov, net_to_mov, shape=in_shape, normalize=True)
inputs = (
    transform(mov, net_to_mov, shape=in_shape, normalize=True),
    transform(fix, net_to_fix, shape=in_shape, normalize=True),
)
if arg.inspect:
    os.makedirs(arg.inspect, exist_ok=True)
    input_1 = os.path.join(arg.inspect, 'input_1.mgz')
    input_2 = os.path.join(arg.inspect, 'input_2.mgz')
    save(path=input_1, dat=inputs[0], affine=mov.affine @ net_to_mov)
    save(path=input_2, dat=inputs[1], affine=fix.affine @ net_to_fix)

# Model.
if is_linear:
    model = vm_affine(in_shape, rigid=arg.model == 'rigid')
else:
    model = vm_dense(in_shape)

if not arg.weights:
    # fs = os.environ.get('FREESURFER_HOME')
    # assert fs, 'set environment variable FREESURFER_HOME or specify weights'

    f = weights[arg.model]
    if arg.model == 'deform':
        f = f.format(smooth=arg.smooth)

    # parent_path = os.getcwd()
    model_path = arg.model_path
    arg.weights = os.path.join(model_path, f)

model.load_weights(arg.weights)
trans = model(inputs)

# Add the last row to create a full matrix. Convert from zero-centered to
# zero-based indices. Then compute the transform from native fixed to native
# moving voxel spaces. Also compute a transform operating in RAS.
if is_linear:
    trans = np.concatenate((
        np.squeeze(trans),
        np.reshape((0, 0, 0, 1), newshape=(1, -1)),
    ))
    trans = cen_to_ind @ trans @ ind_to_cen
    trans_vox = net_to_mov @ trans @ fix_to_net
    trans_ras = mov_to_ras @ trans_vox @ ras_to_fix

else:
    # Construct grid of zero-based index coordinates and shape (3, N) in native
    # fixed voxel space, where N is the number of voxels.
    x_fix = (tf.range(x, dtype=tf.float32) for x in fix.shape)
    x_fix = tf.meshgrid(*x_fix, indexing='ij')
    x_fix = tf.stack(x_fix)
    x_fix = tf.reshape(x_fix, shape=(3, -1))

    # Transform fixed voxel coordinates to the fixed network space.
    x_out = fix_to_net[:-1, -1:] + (fix_to_net[:-1, :-1] @ x_fix)
    x_out = tf.transpose(x_out)

    # Add predicted warp to coordinates to go to the moving network space.
    trans = tf.squeeze(trans)
    x_out += ne.utils.interpn(trans, x_out, fill_value=0)
    x_out = tf.transpose(x_out)

    # Transform coordinates to the native moving voxel space. Subtract fixed
    # coordinates to obtain displacement from fixed to moving voxel space.
    x_out = net_to_mov[:-1, -1:] + (net_to_mov[:-1, :-1] @ x_out)
    trans_vox = tf.transpose(x_out - x_fix)
    trans_vox = tf.reshape(trans_vox, shape=(*fix.shape, -1))
    # Save trans voxel
    tv_save_path = Path(arg.moved).parent / Path(arg.moved).name.replace('.nii.gz', '_transvoxel.npz')
    if arg.gpu:
        np.savez(tv_save_path, trans_vox.cpu().numpy())
    else:
        np.savez(tv_save_path, trans_vox.numpy())

    # Displacement from fixed to moving RAS coordinates.
    x_ras = fix_to_ras[:-1, -1:] + (fix_to_ras[:-1, :-1] @ x_fix)
    x_out = mov_to_ras[:-1, -1:] + (mov_to_ras[:-1, :-1] @ x_out)
    trans_ras = tf.transpose(x_out - x_ras)
    trans_ras = tf.reshape(trans_ras, shape=(*fix.shape, -1))

# Output transforms operating in RAS.
if arg.trans:
    if is_linear:
        np.savetxt(fname=arg.trans, X=trans_ras, fmt='%.8f %.8f %.8f %.8f')

    else:
        save(arg.trans, dat=trans_ras, affine=fix.affine)

if arg.bold:
    bold = nib.load(arg.bold)
    out_bold = batch_transform(bold, trans=trans_vox)
    mc = nib.load(arg.mc)
    bold_save(arg.bold_out, data=out_bold, affine=fix.affine, header=mc.header, dtype=mov.dataobj.dtype)
# Output moved image. Keep same data type as the input.
if arg.moved:
    if arg.header_only:
        assert is_linear, '--header-only applies only to linear registration'
        mov_to_fix = np.linalg.inv(trans_ras)
        save(arg.moved, dat=mov, affine=mov_to_fix @ mov.affine)

    else:
        out = transform(mov, trans=trans_vox, shape=fix.shape)
        save(arg.moved, dat=out, affine=fix.affine, dtype=mov.dataobj.dtype)
        if arg.apply:
            apply_mov = nib.load(arg.apply)
            apply_out = transform(apply_mov, trans=trans_vox, shape=fix.shape)
            save(arg.apply_out, dat=apply_out, affine=fix.affine, dtype=apply_mov.dataobj.dtype)

# print('Thank you for using SynthMorph. Please cite:')
# print(rewrap(ref))
