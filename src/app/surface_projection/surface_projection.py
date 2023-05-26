import os
import sh
import sys

from pathlib import Path


def _path_extension(fpath):
    '''
    .nii.gz friendly extension getter.
    '''

    fname, ext = os.path.splitext(fpath)
    if ext == '.gz':
        ext = os.path.splitext(fname)[-1] + '.gz'
    return ext


def indi_to_fs6(surf_path, subject, input_path, reg_path, hemi):
    '''
    Project a volume (e.g., residuals) in native space onto the
    fsaverage6 surface.

    subject     - str. Subject ID.
    input_path  - Path. Volume to project.
    reg_path    - Path. Registaration .dat file.
    hemi        - str from {'lh', 'rh'}.

    Output file created at $DATA_DIR/surf/
     input_path.name.replace(input_path.ext, '_fsaverage6' + input_path.ext))
    Path object pointing to this file is returned.
    '''
    ext = _path_extension(input_path)
    input_name = input_path.name.replace(ext, '')

    outpath = surf_path / (hemi + '.' + input_name + '_fsaverage6' + ext)
    sh.mri_vol2surf(
        '--mov', input_path,
        '--reg', reg_path,
        '--hemi', hemi,
        '--projfrac', 0.5,
        '--trgsubject', 'fsaverage6',
        '--o', outpath,
        '--reshape',
        '--interp', 'trilinear',
        _out=sys.stdout
    )
    return outpath


def t1_to_native_surf(subject, input_path, out_path, hemi):
    '''
    Project a volume (e.g., residuals) to native surface space.

    subject     - str. Subject ID.
    input_path  - Path. Volume to project.
    out_path    - Path. Surface .gz or .mgh files.
    hemi        - str from {'lh', 'rh'}.
    '''

    sh.mri_vol2surf(
        '--mov', input_path,
        '--hemi', hemi,
        '--projfrac', 1,
        '--trgsubject', subject,
        '--o', out_path,
        '--reshape',
        '--interp', 'nearest',
        '--regheader', subject,
        _out=sys.stdout
    )
    return out_path


def native_to_fs4_surf(subject, input_path, outpath, hemi):
    sh.mri_surf2surf(
        '--srcsubject', subject,
        '--sval', input_path,
        '--trgsubject', 'fsaverage4',
        '--tval', outpath,
        '--hemi', hemi,
        _out=sys.stdout
    )


def smooth_fs6(input_path, hemi):
    '''
    Smooth values on the fsaverage6 surface.

    input_path - Path. Surface to smooth.
    hemi       - str from {'lh', 'rh'}.

    Output file created at
    input_path.replace(input_path.ext, '_sm6' + input_path.ext).
    Path object pointing to this file is returned.
    '''
    ext = _path_extension(input_path)
    outpath = Path(str(input_path).replace(ext, '_sm6' + ext))
    sh.mri_surf2surf(
        '--hemi', hemi,
        '--s', 'fsaverage6',
        '--sval', input_path,
        '--label-src', hemi + '.cortex.label',
        '--fwhm-trg', 6,
        '--tval', outpath,
        '--reshape',
        _out=sys.stdout
    )
    return outpath


def downsample_fs6_to_fs4(input_path, hemi):
    '''
    Downsample from the fsaverage6 surface to the fsaverage4 surface.

    input_path - Path. Surface to smooth.
    hemi       - str from {'lh', 'rh'}.

    Output file created at
    input_path.replace(input_path.ext, '_fsaverage4' + input_path.ext).
    Path object pointing to this file is returned.

    fsprojresolution - fsaverage6
    fsresolution - fsaverage4
    '''
    ext = _path_extension(input_path)
    fs5_path = Path(str(input_path).replace(ext, '_fsaverage5' + ext))
    sh.mri_surf2surf(
        '--hemi', hemi,
        '--srcsubject', 'fsaverage6',
        '--sval', input_path,
        '--label-src', hemi + '.cortex.label',
        '--nsmooth-in', 1,
        '--trgsubject', 'fsaverage5',
        '--tval', fs5_path,
        '--reshape',
        _out=sys.stdout
    )

    outpath = Path(str(input_path).replace(ext, '_fsaverage4' + ext))
    sh.mri_surf2surf(
        '--hemi', hemi,
        '--srcsubject', 'fsaverage5',
        '--sval', fs5_path,
        '--label-src', hemi + '.cortex.label',
        '--nsmooth-in', 1,
        '--trgsubject', 'fsaverage4',
        '--tval', outpath,
        '--reshape',
        _out=sys.stdout
    )

    return outpath


def fs2mm_vol_to_surf(input_path: Path, output_path: Path, hemi: str, surf_space: str) -> Path:
    '''
    Project a volume in FS2mm space into a surface

    input_path  - Path. Volume to project.
    out_path    - Path. Surface .gz or .mgh files.
    hemi        - str. from {'lh', 'rh'}.
    surf_space  - str.
    '''
    sh.mri_vol2surf(
        '--mov', input_path,
        '--hemi', hemi,
        '--projfrac', 0.5,
        '--regheader', 'FS2mm',
        '--trgsubject', surf_space,
        '--o', output_path,
        '--reshape',
        '--interp', 'nearest',
        _out=sys.stdout
    )

    return output_path
