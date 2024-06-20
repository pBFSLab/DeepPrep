#!/usr/bin/env python3

# Built-in modules. Import third-party modules further down.
import os
import sys
import shutil
import textwrap
import argparse
import nibabel as nib

# Settings.
default = {
    'model': 'joint',
    'hyper': 0.5,
    'extent': 192,
    'steps': 7,
}
choices = {
    'model': ('joint', 'deform', 'affine', 'rigid'),
    'extent': (192, 256),
}
limits = {
    'steps': 5,
}
weights = {
    'joint': ('synthmorph.affine.2.h5', 'synthmorph.deform.2.h5',),
    'deform': ('synthmorph.deform.2.h5',),
    'affine': ('synthmorph.affine.2.h5',),
    'rigid': ('synthmorph.rigid.1.h5',),
}


def rewrap(text, width=None, hard='\t\n', hard_indent=0):
    """Rewrap text such that lines fill the available horizontal space.

    Reformats individual paragraphs of a text body, considering subsequent
    lines with identical indentation as paragraphs. For unspecified width, the
    function will attempt to determine the extent of the terminal.

    Parameters
    ----------
    text : str
        Text to rewrap.
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


def network_space(im, shape, center=None):
    """Construct transform from network space to the voxel space of an image.

    Constructs a coordinate transform from the space the network will operate
    in to the zero-based image index space. The network space has isotropic
    1-mm voxels, left-inferior-anterior (LIA) orientation, and no shear. It is
    centered on the field of view, or that of a reference image. This space is
    an indexed voxel space, not world space.

    Parameters
    ----------
    im : surfa.Volume
        Input image to construct the transform for.
    shape : (3,) array-like
        Spatial shape of the network space.
    center : surfa.Volume, optional
        Center the network space on the center of a reference image.

    Returns
    -------
    out : tuple of (3, 4) NumPy arrays
        Transform from network to input-image space and its inverse, thinking
        coordinates.

    """
    old = im.geom
    new = sf.ImageGeometry(
        shape=shape,
        voxsize=1,
        rotation='LIA',
        center=old.center if center is None else center.geom.center,
        shear=None,
    )

    net_to_vox = old.world2vox @ new.vox2world
    vox_to_net = new.world2vox @ old.vox2world
    return np.float32(net_to_vox.matrix), np.float32(vox_to_net.matrix)


def transform(im, trans, shape=None, normalize=False, batch=False):
    """Apply a spatial transform to 3D image voxel data in dimensions.

    Applies a transformation matrix operating in zero-based index space or a
    displacement field to an image buffer.

    Parameters
    ----------
    im : surfa.Volume or NumPy array or TensorFlow tensor
        Input image to transform, without batch dimension.
    trans : array-like
        Transform to apply to the image. A matrix of shape (3, 4), a matrix
        of shape (4, 4), or a displacement field of shape (*space, 3),
        without batch dimension.
    shape : (3,) array-like, optional
        Output shape used for converting matrices to dense transforms. None
        means the shape of the input image will be used.
    normalize : bool, optional
        Min-max normalize the image intensities into the interval [0, 1].
    batch : bool, optional
        Prepend a singleton batch dimension to the output tensor.

    Returns
    -------
    out : float TensorFlow tensor
        Transformed image with with a trailing feature dimension.

    """
    # Add singleton feature dimension if needed.
    if tf.rank(im) == 3:
        im = im[..., tf.newaxis]

    out = vxm.utils.transform(
        im, trans, fill_value=0, shift_center=False, shape=shape,
    )

    if normalize:
        out -= tf.reduce_min(out)
        out /= tf.reduce_max(out)

    if batch:
        out = out[tf.newaxis, ...]

    return out


def load_weights(model, weights):
    """Load weights into model or submodel.

    Attempts to load weights into a model or its direct submodels and return on
    first success. Raises a `ValueError` if unsuccessful.

    Parameters
    ----------
    model : TensorFlow model
        Model to initialize.
    weights : str
        Path to weights file.

    """
    cand = (model, *(f for f in model.layers if isinstance(f, tf.keras.Model)))
    for c in cand:
        try:
            c.load_weights(weights)
            return
        except ValueError as e:
            if c is cand[-1]:
                raise e


def convert_to_ras(trans, source, target):
    """Convert a voxel-to-voxel transform to a world-to-world (RAS) transform.

    For input displacement fields, we want to output shifts that we can add to
    the RAS coordinates corresponding to indices (0, 1, ...) along each axis in
    the target voxel space - instead of adding the shifts to discrete RAS
    indices (0, 1, ...) along each axis. Therefore, we start with target voxel
    coordinates and subtract the corresponding RAS coordinates at the end.
    Naming transforms after their effect on coordinates,

        out = mov_to_ras @ fix_to_mov + x_fix - x_ras
        out = mov_to_ras @ fix_to_mov + x_fix - fix_to_ras @ x_fix
        out = mov_to_ras @ fix_to_mov + (identity - fix_to_ras) @ x_fix

    Parameters
    ----------
    trans : (3, 4) or (4, 4) or (*space, 3) array-like
        Matrix transform or displacement field.
    source : castable surfa.ImageGeometry
        Transform source (or moving) image geometry.
    target : castable surfa.ImageGeometry
        Transform target (or fixed) image geometry.

    Returns
    -------
    out: float TensorFlow tensor
        Converted world-space transform of shape (3, 4) if `trans` is a matrix
        or (*space, 3) if it is a displacement field.

    """
    mov = sf.transform.geometry.cast_image_geometry(source)
    fix = sf.transform.geometry.cast_image_geometry(target)
    mov_to_ras = np.float32(mov.vox2world.matrix)
    fix_to_ras = np.float32(fix.vox2world.matrix)
    ras_to_fix = np.float32(fix.world2vox.matrix)
    prop = dict(shift_center=False, shape=fix.shape)

    # Simple matrix multiplication.
    if vxm.utils.is_affine_shape(trans.shape):
        return vxm.utils.compose((mov_to_ras, trans, ras_to_fix), **prop)

    # Target voxel coordinate grid.
    x_fix = (tf.range(x, dtype=tf.float32) for x in fix.shape)
    x_fix = tf.meshgrid(*x_fix, indexing='ij')
    x_fix = tf.stack(x_fix)
    x_fix = tf.reshape(x_fix, shape=(3, -1))

    # We go from target voxels to source voxels to RAS, then subtract the RAS
    # coordinates corresponding to the target voxels we started at.
    mat = tf.eye(4) - fix_to_ras
    x = mat[:-1, -1:] + mat[:-1, :-1] @ x_fix
    x = tf.transpose(x)
    x = tf.reshape(x, shape=(*fix.shape, -1))
    return vxm.utils.compose((mov_to_ras, trans), **prop) + x


# Documentation.
n = '\033[0m' if sys.stdout.isatty() else ''
b = '\033[1m' if sys.stdout.isatty() else ''
u = '\033[4m' if sys.stdout.isatty() else ''
prog = os.path.basename(sys.argv[0])
doc = f'''{prog}

{b}NAME{n}
        {b}{prog}{n} - register 3D brain images without preprocessing

{b}SYNOPSIS{n}
        {b}{prog}{n} [options] {u}moving{n} {u}fixed{n}

{b}DESCRIPTION{n}
        SynthMorph is a deep-learning tool for symmetric, anatomy-aware, and
        acquisition-agnostic registration of any brain image right off the MRI
        scanner. In contrast to anatomy-agnostic methods, SynthMorph can
        distinguish anatomy of interest from irrelevant structures, removing
        the need for preprocessing that excludes content which would reduce the
        accuracy of anatomy-specific registration.

        SynthMorph registers a {u}moving{n} (source) to a {u}fixed{n} (target)
        image. The options are as follows:

        {b}-m{n} {u}model{n}
                Transformation model ({', '.join(choices['model'])}). Defaults
                to {default['model']}. Joint includes affine and deformable but
                differs from running both in sequence in that it applies the
                deformable step in an affine mid-space to guarantee symmetric
                joint transforms. Deformable assumes prior affine alignment or
                initialization with {b}-i{n}.

        {b}-o{n} {u}image{n}
                Save the moving image registered to the fixed image.

        {b}-O{n} {u}image{n}
                Save the fixed image registered to the moving image.

        {b}-H{n}
                Update the voxel-to-world matrix instead of resampling when
                saving images with {b}-o{n} and {b}-O{n}. For matrix transforms
                only. Not all software supports headers with shear from affine
                registration.

        {b}-t{n} {u}trans{n}
                Save the transform from the moving to the fixed image,
                including any initialization.

        {b}-T{n} {u}trans{n}
                Save the transform from the fixed to the moving image,
                including any initialization.

        {b}-i{n} {u}trans{n}
                Apply an initial matrix transform to the moving image before
                the registration.

        {b}-j{n} {u}threads{n}
                Number of TensorFlow threads. System default if unspecified.

        {b}-g{n}
                Use the GPU in environment variable CUDA_VISIBLE_DEVICES or GPU
                0 if the variable is unset or empty.

        {b}-r{n} {u}lambda{n}
                Regularization parameter in the open interval (0, 1) for
                deformable registration. Higher values lead to smoother warps.
                Defaults to {default['hyper']}.

        {b}-n{n} {u}steps{n}
                Integration steps for deformable registration. Lower numbers
                improve speed and memory use but can lead to inaccuracies and
                folding voxels. Defaults to {default['steps']}. Should not be
                less than {limits['steps']}.

        {b}-e{n} {u}extent{n}
                Isotropic extent of the registration space in unit voxels
                {choices['extent']}. Lower values improve speed and memory use
                but may crop the anatomy of interest. Defaults to
                {default['extent']}.

        {b}-w{n} {u}weights{n}
                Alternative weights for the selected registration model. Repeat
                the option to set submodel weights.

        {b}-h{n}
                Print this help text and exit.

{b}IMAGE FORMAT{n}
        The registration supports 3-dimensional single-frame images with any
        resolution and orientation. The accepted file formats include: MGH
        (.mgz) and NIfTI (.nii.gz, .nii).

        Internally, it converts image buffers to the format expected by the
        network: isotropic 1-mm voxels, intensities min-max normalized into the
        interval [0, 1], and left-inferior-anterior (LIA) axis orientation.
        This conversion requires intact image-to-world matrices in the image
        headers. The head must have the correct anatomical orientation in a
        viewer like FreeView.

{b}TRANSFORMS{n}
        Transforms operate in physical RAS space. We save matrix transforms in
        LTA text format (.lta) and displacement fields as images with three
        frames indicating shifts in RAS direction.

        For converting, composing, and applying transforms, consider the
        FreeSurfer tools lta_convert, mri_warp_convert, mri_concatenate_lta,
        mri_concatenate_gcam, mri_convert, mri_info.

        Convert FreeSurfer's LTA format to NiftyReg's matrix format:
                # lta_convert --src mov.mgz --trg fix.nii.gz --inlta aff.lta
                --outras aff.txt

        Apply an LTA to another image:
                # mri_convert -at aff.lta in.mgz out.mgz

        Convert a deformable transform to FreeSurfer's GCAM format (.m3z):
                # mri_warp_convert -g mov.mgz --inras def.mgz --outm3z def.m3z

        Apply a GCAM (.m3z) to another image:
                # mri_convert -at def.m3z in.nii.gz out.nii.gz

{b}ENVIRONMENT{n}
        The following environment variables affect {b}{prog}{n}:

        CUDA_VISIBLE_DEVICES
                Use a specific GPU. If unset or empty, passing {b}-g{n} will
                select GPU 0. Ignored without {b}-g{n}.

        FREESURFER_HOME
                Load model weights from directory {u}FREESURFER_HOME/models{n}.
                Ignored when specifying weights with {b}-w{n}.

        SUBJECTS_DIR
                Ignored unless {b}{prog}{n} runs inside a container. Mount the
                host directory SUBJECTS_DIR to {u}/mnt{n} inside the container.
                Defaults to the current working directory.

{b}EXAMPLES{n}
        Joint affine-deformable registration, saving the moved image:
                # {prog} -o out.nii mov.nii.gz fix.mgz

        Affine registration saving the transform:
                # {prog} -m affine -t aff.lta mov.mgz fix.mgz

        Deformable registration only, assuming prior affine alignment:
                # {prog} -m deform -t def.mgz mov.mgz fix.mgz

        Deformable registration initialized with an affine transform:
                # {prog} -m deform -i aff.lta -o out.mgz mov.mgz fix.mgz

        Rigid registration, updating the output image header (no resampling):
                # {prog} -m rigid -Ho out.mgz mov.mgz fix.mgz

{b}CONTACT{n}
        Reach out to freesurfer@nmr.mgh.harvard.edu or at
        https://github.com/voxelmorph/voxelmorph.

{b}REFERENCES{n}
        If you use SynthMorph in a publication, please cite us!
'''


# References.
ref = '''
Anatomy-specific acquisition-agnostic affine registration learned from fictitious images\t
Hoffmann M, Hoopes A, Fischl B*, Dalca AV* (*equal contribution)\t
SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023\t
https://doi.org/10.1117/12.2653251\t
https://malte.cz/#papers (PDF)

SynthMorph: learning contrast-invariant registration without acquired images\t
Hoffmann M, Billot B, Greve DN, Iglesias JE, Fischl B, Dalca AV\t
IEEE Transactions on Medical Imaging, 41 (3), 543-558, 2022\t
https://doi.org/10.1109/TMI.2021.3116879

Website: https://w3id.org/synthmorph
'''
doc += textwrap.indent(ref, prefix=' ' * 8)


# Command-line arguments.
p = argparse.ArgumentParser(add_help=False)
p.add_argument('moving')
p.add_argument('fixed')
p.add_argument('-m', dest='model', choices=choices['model'], default=default['model'])
p.add_argument('-o', dest='out_moving', metavar='image')
p.add_argument('-O', dest='out_fixed', metavar='image')
p.add_argument('-H', dest='header_only', action='store_true')
p.add_argument('-t', dest='trans', metavar='trans')
p.add_argument('-T', dest='inverse', metavar='trans')
p.add_argument('-i', dest='init', metavar='trans')
p.add_argument('-j', dest='threads', metavar='threads', type=int)
p.add_argument('-g', dest='gpu', action='store_true')
p.add_argument('-r', dest='hyper', metavar='lambda', type=float, default=default['hyper'])
p.add_argument('-n', dest='steps', metavar='steps', default=default['steps'], type=int)
p.add_argument('-e', dest='extent', choices=choices['extent'], default=default['extent'])
p.add_argument('-w', dest='weights', metavar='weights', action='append')
p.add_argument('-h', action='store_true')
p.add_argument('-v', dest='verbose', action='store_true')
p.add_argument('-d', dest='out_dir', metavar='dir')
p.add_argument('-mp', '--model_path', type=str, metavar='MODEL PATH')
p.add_argument('-a', '--apply', type=str, metavar='APPLY_TRANS')
p.add_argument('-ao', '--apply_out', type=str, metavar='APPLY_TRANS_OUT')

# Help.
if len(sys.argv) == 1:
    p.print_usage()
    exit(0)

if any(f[0] == '-' and 'h' in f for f in sys.argv):
    print(rewrap(doc), end='\n\n')
    exit(0)


# Parse arguments.
arg = p.parse_args()
in_shape = (arg.extent,) * 3
is_mat = arg.model in ('affine', 'rigid')

if arg.header_only and not is_mat:
    print('Error: -H is not compatible with deformable registration')
    exit(1)

if not 0 < arg.hyper < 1:
    print('Error: regularization strength not in open interval (0, 1)')
    exit(1)

if arg.steps < limits['steps']:
    print('Error: too few integration steps')
    exit(1)


# Third-party imports. Avoid waiting for TensorFlow just for documentation.
import numpy as np
import surfa as sf
import tensorflow as tf
import neurite as ne
import voxelmorph as vxm

# Setup.
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

# Threading.
if arg.threads:
    tf.config.threading.set_inter_op_parallelism_threads(arg.threads)
    tf.config.threading.set_intra_op_parallelism_threads(arg.threads)


# Input data.
mov = sf.load_volume(arg.moving)
fix = sf.load_volume(arg.fixed)
if not len(mov.shape) == len(fix.shape) == 3:
    sf.system.fatal('input images are not single-frame volumes')


# Transforms between native voxel and network coordinates. Voxel and network
# spaces differ for each image. The networks expect isotropic 1-mm LIA spaces.
# We center these on the original images, except for deformable registration:
# this assumes prior affine registration, so we center the moving network space
# on the fixed image, to take into account affine transforms applied by
# resampling, updating the header, or passed on the command line alike.
center = fix if arg.model == 'deform' else None
net_to_mov, mov_to_net = network_space(mov, shape=in_shape, center=center)
net_to_fix, fix_to_net = network_space(fix, shape=in_shape)

# Coordinate transforms from and to world space. There is only one world.
mov_to_ras = np.float32(mov.geom.vox2world.matrix)
fix_to_ras = np.float32(fix.geom.vox2world.matrix)
ras_to_mov = np.float32(mov.geom.world2vox.matrix)
ras_to_fix = np.float32(fix.geom.world2vox.matrix)


# Incorporate an initial matrix transform. It maps from fixed to moving world
# coordinates, so we start with fixed network space on the right. FreeSurfer
# LTAs store the inverse of the transform.
if arg.init:
    init = sf.load_affine(arg.init).convert(space='world')
    if init.ndim != 3 \
        or not sf.transform.image_geometry_equal(mov.geom, init.source, tol=1e-4) \
        or not sf.transform.image_geometry_equal(fix.geom, init.target, tol=1e-4):
        sf.system.fatal('initial transform geometry does not match images')

    net_to_mov = np.float32(ras_to_mov @ init.inv() @ fix_to_ras @ net_to_fix)
    mov_to_new = np.float32(fix_to_net @ ras_to_fix @ init @ mov_to_ras)


# Take the input images to network space. When saving the moving image with the
# correct voxel-to-RAS matrix after incorporating an initial matrix transform,
# an image viewer taking this matrix into account will show an unchanged image.
# However, the networks only see the voxel data, which have been moved.
inputs = (
    transform(mov, net_to_mov, shape=in_shape, normalize=True, batch=True),
    transform(fix, net_to_fix, shape=in_shape, normalize=True, batch=True),
)
if arg.out_dir:
    os.makedirs(arg.out_dir, exist_ok=True)
    inp_1 = os.path.join(arg.out_dir, 'inp_1.mgz')
    inp_2 = os.path.join(arg.out_dir, 'inp_2.mgz')
    geom_1 = sf.ImageGeometry(in_shape, vox2world=mov_to_ras @ net_to_mov)
    geom_2 = sf.ImageGeometry(in_shape, vox2world=fix_to_ras @ net_to_fix)
    sf.Volume(inputs[0][0], geom_1).save(inp_1)
    sf.Volume(inputs[1][0], geom_2).save(inp_2)


# Network.
prop = dict(in_shape=in_shape, bidir=True)
if is_mat:
    prop.update(make_dense=False, rigid=arg.model == 'rigid')
    model = vxm.networks.VxmAffineFeatureDetector(**prop)

else:
    prop.update(mid_space=True, int_steps=arg.steps, skip_affine=arg.model == 'deform')
    model = vxm.networks.HyperVxmJoint(**prop)
    inputs = (np.asarray([arg.hyper]), *inputs)


# Weights.
arg.weights = [os.path.join(arg.model_path, f) for f in weights[arg.model]]

for f in arg.weights:
    load_weights(model, weights=f)


# Inference. The first transform maps from the moving to the fixed image, or
# equivalently, from fixed to moving coordinates. The second is the inverse.
vox_1, vox_2 = map(tf.squeeze, model(inputs))


# Convert transforms between moving and fixed network spaces to transforms
# between the original voxel spaces. Also compute transforms operating in RAS.
prop = dict(shift_center=False, shape=fix.shape)
vox_1 = vxm.utils.compose((net_to_mov, vox_1, fix_to_net), **prop)
ras_1 = convert_to_ras(vox_1, source=mov, target=fix)

prop = dict(shift_center=False, shape=mov.shape)
vox_2 = vxm.utils.compose((net_to_fix, vox_2, mov_to_net), **prop)
ras_2 = convert_to_ras(vox_2, source=fix, target=mov)


# Save transform from moving to fixed image. FreeSurfer LTAs store the inverse.
if arg.trans:
    if is_mat:
        out = sf.Affine(ras_2, source=mov, target=fix, space='world')
    else:
        out = fix.new(ras_1)
    out.save(arg.trans)


# Save transform from fixed to moving image. FreeSurfer LTAs store the inverse.
if arg.inverse:
    if is_mat:
        out = sf.Affine(ras_1, source=fix, target=mov, space='world')
    else:
        out = mov.new(ras_2)
    out.save(arg.inverse)


# Save moving image registered to fixed image.
if arg.out_moving:
    if arg.header_only:
        out = mov.copy()
        out.geom.update(vox2world=ras_2 @ mov.geom.vox2world)
    else:
        out = transform(mov, trans=vox_1, shape=fix.shape)
        out = fix.new(out)
    out.save(arg.out_moving)


# Save fixed image registered to moving image.
if arg.out_fixed:
    if arg.header_only:
        out = fix.copy()
        out.geom.update(vox2world=ras_1 @ fix.geom.vox2world)
    else:
        out = transform(fix, trans=vox_2, shape=mov.shape)
        out = mov.new(out)
    out.save(arg.out_fixed)
if arg.apply:
    apply_mov = sf.load_volume(arg.apply)
    apply_out = transform(apply_mov, trans=vox_1, shape=fix.shape)
    apply_out = fix.new(apply_out)
    apply_out.save(arg.apply_out)

print('Thank you for choosing SynthMorph. Please cite us!')
print(rewrap(ref))
