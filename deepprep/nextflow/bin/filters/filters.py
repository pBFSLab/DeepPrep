import nibabel as nib
import numpy as np
from . import butt1d
from scipy.ndimage import gaussian_filter


def bandpass_nifti(vol, nskip, order, band, tr):
    # Parameters.
    margin = 16

    # Read source data.
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
    return imgt


def _bandpass_nifti(gauss_path, bpss_path, tr):
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
    fhalf_lo = 0.01  # 1e-16  # make parameter the same to INDI_lab
    fhalf_hi = 0.08
    band = [fhalf_lo, fhalf_hi]
    order = 2
    _bandpass_nifti(gauss_path, bpss_path, nskip, order, band, tr)
    return bpss_path
