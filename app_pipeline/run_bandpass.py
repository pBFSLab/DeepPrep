from filters.filters import _bandpass_nifti


# Get and set parameters.
nskip = 0
fhalf_lo = 0.01  # 1e-16  # make parameter the same to INDI_lab
fhalf_hi = 1
band = [fhalf_lo, fhalf_hi]
order = 2
tr = 3000
gauss_path = '/home/anning/Downloads/NGCASD031_LJ_bld_rest_reorient_skip_faln_mc_g1000000000.nii.gz'
bpss_path = '/home/anning/Downloads/NGCASD031_LJ_bld_rest_reorient_skip_faln_mc_g1000000000_bpass-001-1_3000.nii.gz'
_bandpass_nifti(gauss_path, bpss_path, nskip, order, band, tr)
