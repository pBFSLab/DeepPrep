import nibabel as nib
import numpy as np
import numpy
from typing import Iterable
from numpy.core.multiarray import normalize_axis_index
from . import _nd_image, butt1d


def _normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(input, str)
    if not is_str and isinstance(input, Iterable):
        normalized = list(input)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized


def _get_output(output, input, shape=None):
    if shape is None:
        shape = input.shape
    if output is None:
        output = np.zeros(shape, dtype=input.dtype.name)
    elif isinstance(output, (type, np.dtype)):
        # Classes (like `np.float32`) and dtypes are interpreted as dtype
        output = np.zeros(shape, dtype=output)
    elif isinstance(output, str):
        output = np.typeDict[output]
        output = np.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    return output


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = numpy.arange(order + 1)
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius + 1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = numpy.zeros(order + 1)
        q[0] = 1
        D = numpy.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = numpy.diag(numpy.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def _invalid_origin(origin, lenw):
    return (origin < -(lenw // 2)) or (origin > (lenw - 1) // 2)


def _extend_mode_to_code(mode):
    """Convert an extension mode to the corresponding integer code.
    """
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode == 'reflect':
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    else:
        raise RuntimeError('boundary mode not supported')


def correlate1d(input, weights, axis=-1, output=None, mode="reflect",
                cval=0.0, origin=0):
    """Calculate a 1-D correlation along the given axis.

    The lines of the array along the given axis are correlated with the
    given weights.

    Parameters
    ----------
    %(input)s
    weights : array
        1-D sequence of numbers.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s

    Examples
    --------
    >>> from scipy.ndimage import correlate1d
    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([ 8, 26,  8, 12,  7, 28, 36,  9])
    """
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    output = _get_output(output, input)
    weights = numpy.asarray(weights, dtype=numpy.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = normalize_axis_index(axis, input.ndim)
    if _invalid_origin(origin, len(weights)):
        raise ValueError('Invalid origin; origin must satisfy '
                         '-(len(weights) // 2) <= origin <= '
                         '(len(weights)-1) // 2')
    mode = _extend_mode_to_code(mode)
    _nd_image.correlate1d(input, weights, axis, output, mode, cval,
                          origin)
    return output


def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    """1-D Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    gaussian_filter1d : ndarray

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter1d
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
    >>> import matplotlib.pyplot as plt
    >>> np.random.seed(280490)
    >>> x = np.random.randn(101).cumsum()
    >>> y3 = gaussian_filter1d(x, 3)
    >>> y6 = gaussian_filter1d(x, 6)
    >>> plt.plot(x, 'k', label='original data')
    >>> plt.plot(y3, '--', label='filtered, sigma=3')
    >>> plt.plot(y6, ':', label='filtered, sigma=6')
    >>> plt.legend()
    >>> plt.grid()
    >>> plt.show()
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)


def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    """Multidimensional Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : int or sequence of ints, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. A positive order
        corresponds to convolution with that derivative of a Gaussian.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    truncate : float
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    gaussian_filter : ndarray
        Returned array of same shape as `input`.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter
    >>> a = np.arange(50, step=2).reshape((5,5))
    >>> a
    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28],
           [30, 32, 34, 36, 38],
           [40, 42, 44, 46, 48]])
    >>> gaussian_filter(a, sigma=1)
    array([[ 4,  6,  8,  9, 11],
           [10, 12, 14, 15, 17],
           [20, 22, 24, 25, 27],
           [29, 31, 33, 34, 36],
           [35, 37, 39, 40, 42]])

    >>> from scipy import misc
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = misc.ascent()
    >>> result = gaussian_filter(ascent, sigma=5)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)
    output = _get_output(output, input)
    orders = _normalize_sequence(order, input.ndim)
    sigmas = _normalize_sequence(sigma, input.ndim)
    modes = _normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            gaussian_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate)
            input = output
    else:
        output[...] = input[...]
    return output


def gauss_nifti(mc_path, scale):
    '''
    Perform a 3d Gaussian filter on each frame of a nifti file.

    mc_path  - path. Path of bold after motion correction process.
    scale    - float. Scale value.

    Nifti files are created with the same name as those in the input list file,
    with an appended suffix of _g<int(scale)>.
    '''

    sigma = 1.0 / scale

    img = nib.load(mc_path)
    data = img.get_data()
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[3]):
        filtered_data[:, :, :, i] = gaussian_filter(
            data[:, :, :, i], sigma)

    new_img = nib.Nifti1Image(
        filtered_data, img.affine, img.header)
    gauss_path = mc_path.replace('.nii.gz', '_g%d.nii.gz' % scale)
    nib.save(new_img, gauss_path)

    del data
    del img
    del filtered_data
    del new_img
    return gauss_path


def _bandpass_nifti(fname, bpss_path, nskip, order, band, tr):
    # Parameters.
    margin = 16

    # Read source data.
    src = nib.load(fname)
    vol = src.get_data()
    imgt = vol.copy()
    V_DIM = vol.shape[3]

    # Allocate time series buffers.
    tdim = V_DIM - nskip
    tdim_pad = butt1d.npad(tdim, margin)
    padlen = tdim_pad - tdim

    # butt1db in butt1d only accept np.float32, but np.float64 default in python.
    tpad = np.zeros(tdim_pad, np.float32)
    x = -1.0 + 2.0 * np.arange(tdim) / (tdim - 1.0)
    sxx = (tdim * (tdim + 1.0)) / (3.0 * (tdim - 1.0))

    # Process all voxels of one run.
    for iz in range(vol.shape[2]):
        for iy in range(vol.shape[1]):
            for ix in range(vol.shape[0]):
                tpad[:tdim] = imgt[ix, iy, iz, nskip:]

                # Remove DC and linear trend.
                sy = np.sum(tpad[:tdim])
                sxy = np.sum(tpad[:tdim] * x)
                a = [sy / tdim, sxy / sxx]
                tpad[:tdim] -= a[0] + x * a[1]

                # Circularly connect time series.
                q = (tpad[0] - tpad[tdim - 1]) / (padlen + 1)
                tpad[tdim:] = tpad[tdim - 1] + q * np.arange(1, padlen + 1)
                # in Fortran, if parameters contain array, the last parameter will be thought as the dimension.
                # https://www.numfys.net/howto/F2PY/
                butt1d.butt1db(tpad, tr, band[0], 0, band[1], order, tdim_pad)

                # Force unpadded timeseries to zero mean and put filtered
                # results back in the image.
                q = np.sum(tpad[:tdim]) / tdim
                imgt[ix, iy, iz, nskip:] = tpad[:tdim] - q
                imgt[ix, iy, iz, :nskip] -= a[0]  # make the way the same to INDI_lab

    # Save the output.
    output = nib.Nifti1Image(imgt, src.affine, header=src.header)
    nib.save(output, bpss_path)


def bandpass_nifti(gauss_path, tr):
    '''
    Perform a bandpass filter on each voxel in each .nii.gz file listed,
    as lines, in :arg listfile:.

    gauss_path  - path. Path of bold after gauss filter process.
    tr        - float. TR period.

    Creates files with 'bpss' in their filenames.
    return:
        bpss_path - path. Path of bold after bpass process.
    '''

    # Get and set parameters.
    nskip = 0
    fhalf_lo = 0.0  # 1e-16  # make parameter the same to INDI_lab
    fhalf_hi = 0.08
    band = [fhalf_lo, fhalf_hi]
    order = 2
    bpss_path = gauss_path.replace('.nii.gz', '_bpss.nii.gz')
    _bandpass_nifti(gauss_path, bpss_path, nskip, order, band, tr)
    return bpss_path
