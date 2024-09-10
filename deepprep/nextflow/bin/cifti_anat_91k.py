#! /usr/bin/env python3
import argparse
import json
import os
import templateflow.api as tf
import nibabel as nb
import typing
import numpy as np

from pathlib import Path
from nibabel import cifti2 as ci
from neuromaps import transforms

grayord_key = {
    "91k": {
        "surface-den": "32k",
        "tf-res": "02",
        "grayords": "91,282",
        "res-mm": "2mm"
    }
}

def set_environ(freesurfer_home):
    # FreeSurfer
    os.environ['FREESURFER_HOME'] = freesurfer_home
    os.environ['SUBJECTS_DIR'] = f'{freesurfer_home}/subjects'
    os.environ['PATH'] = f'{freesurfer_home}/bin:' + '/usr/local/workbench/bin_linux64:' + os.environ['PATH']


def shape_nii2gii(shape, structure, nii_file, gii_file):
    hemi_white = str(nii_file).replace(shape, 'white')
    os.system(f'mris_convert -c {nii_file} {hemi_white} {gii_file}')
    os.system(f'wb_command -set-structure {gii_file} {structure}')
    if shape == 'thickness':
        os.system(f'wb_command -metric-math "(abs(thickness))" {gii_file} -var thickness {gii_file}')
    else:
        os.system(f'wb_command -metric-math "(var * -1)" {gii_file} -var var {gii_file}')

def surf_nii2gii(nii_surf, gii_surf):
    if 'sphere' in os.path.basename(str(nii_surf)):
        os.system(f'mris_convert  {nii_surf} {gii_surf}')
    else:
        os.system(f'mris_convert --to-scanner {nii_surf} {gii_surf}')


def resample_nii2gii(metric_in, current_sphere, new_sphere, metric_out, method):
    if metric_out.exists() is False:
        os.system(f'wb_command -metric-resample {metric_in} {current_sphere} {new_sphere} {method} {metric_out}')


def save_json(metadata, json_oupath):
    with open(json_oupath, 'w') as f:
        json.dump(metadata, f, indent=4)
        f.close()


def prepare_cifti_getsurf(grayordinates: str, surf_name: str) -> [list]:
    # Fetch templates
    surface_files = [
        str(
            tf.get(
                'fsaverage',
                density=grayordinates,
                hemi=hemi,
                suffix=surf_name
            )
        )
        for hemi in ('L', 'R')
    ]

    return surface_files


def _prepare_cifti(grayordinates: str) -> tuple[list, dict]:
    """
    Fetch the required templates needed for CIFTI-2 generation, based on input surface density.

    Parameters
    ----------
    grayordinates :
        Total CIFTI grayordinates (91k)

    Returns
    -------
    surface_labels
        Surface label files for vertex inclusion/exclusion.
    metadata
        Dictionary with BIDS metadata.

    Examples
    --------
    >>> surface_labels, metadata = _prepare_cifti('91k')
    >>> surface_labels  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../tpl-fsLR_hemi-L_den-32k_desc-nomedialwall_dparc.label.gii',
     '.../tpl-fsLR_hemi-R_den-32k_desc-nomedialwall_dparc.label.gii']
    >>> metadata # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    {'Density': '91,282 grayordinates corresponding to all of the grey matter sampled at a \
2mm average vertex spacing...', 'SpatialReference': {'CIFTI_STRUCTURE_CORTEX_LEFT': ...}}

    """

    grayord_key = {
        '91k': {
            'surface-den': '32k',
            'tf-res': '02',
            'grayords': '91,282',
            'res-mm': '2mm',
        }
    }
    if grayordinates not in grayord_key:
        raise NotImplementedError(f'Grayordinates {grayordinates} is not supported.')

    total_grayords = grayord_key[grayordinates]['grayords']
    res_mm = grayord_key[grayordinates]['res-mm']
    surface_density = grayord_key[grayordinates]['surface-den']
    # Fetch templates
    surface_labels = [
        str(
            tf.get(
                'fsLR',
                density=surface_density,
                hemi=hemi,
                desc='nomedialwall',
                suffix='dparc',
                raise_empty=True,
            )
        )
        for hemi in ('L', 'R')
    ]

    tf_url = 'https://templateflow.s3.amazonaws.com'
    surfaces_url = (  # midthickness is the default, but varying levels of inflation are all valid
        f'{tf_url}/tpl-fsLR/tpl-fsLR_den-{surface_density}_hemi-%s_midthickness.surf.gii'
    )
    metadata = {
        'Density': (
            f'{total_grayords} grayordinates corresponding to all of the grey matter sampled at a '
            f'{res_mm} average vertex spacing on the surface'
        ),
        'SpatialReference': {
            'CIFTI_STRUCTURE_CORTEX_LEFT': surfaces_url % 'L',
            'CIFTI_STRUCTURE_CORTEX_RIGHT': surfaces_url % 'R',
        },
    }
    return surface_labels, metadata

def _create_cifti_image(
    scalar_surfs: tuple[str, str],
    surface_labels: tuple[str, str],
    scalar_name: str,
    metadata: typing.Optional[dict] = None,
    cifti_savepath: Path = None,
):
    """
    Generate CIFTI image in target space.

    Parameters
    ----------
    scalar_surfs
        Surface scalar files (L,R)
    surface_labels
        Surface label files used to remove medial wall (L,R)
    metadata
        Metadata to include in CIFTI header
    scalar_name
        Name to apply to scalar map

    Returns
    -------
    out :
        BOLD data saved as CIFTI dtseries
    """
    brainmodels = []
    arrays = []

    for idx, hemi in enumerate(('left', 'right')):
        labels = nb.load(surface_labels[idx])
        mask = np.bool_(labels.darrays[0].data)

        struct = f'cortex_{hemi}'
        brainmodels.append(
            ci.BrainModelAxis(struct, vertex=np.nonzero(mask)[0], nvertices={struct: len(mask)})
        )

        morph_scalar = nb.load(scalar_surfs[idx])
        arrays.append(morph_scalar.darrays[0].data[mask].astype('float32'))

    # provide some metadata to CIFTI matrix
    if not metadata:
        metadata = {
            'surface': 'fsLR',
        }

    # generate and save CIFTI image
    hdr = ci.Cifti2Header.from_axes(
        (ci.ScalarAxis([scalar_name]), brainmodels[0] + brainmodels[1])
    )
    hdr.matrix.metadata = ci.Cifti2MetaData(metadata)

    img = ci.Cifti2Image(dataobj=np.atleast_2d(np.concatenate(arrays)), header=hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS')

    stem = Path(scalar_surfs[0]).name.split('.')[0]
    cifti_stem = '_'.join(ent for ent in stem.split('_') if not ent.startswith('hemi-'))
    out_file = cifti_savepath / f'{cifti_stem}.dscalar.nii'
    json_oupath = cifti_savepath / f'{cifti_stem}.json'
    save_json(metadata, json_oupath)
    img.to_filename(out_file)
    return out_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create anat cifti")
    parser.add_argument('--subject_id', help='subjects id', required=True)
    parser.add_argument('--output_dir', help='DeepPrep output dir', required=True)
    parser.add_argument('--grayordinates', help='Total CIFTI grayordinates', default='91k', required=True)
    parser.add_argument('--freesurfer_home', help='freesurfer home', required=True)
    args = parser.parse_args()

    '''
    Run Example:
    --subject_id sub-MSC05
    --output_dir /mnt/ngshare2/to_them/me/cifit/DP24_41K
    --grayordinates 91k
    --freesurfer_home /usr/local/freesurfer
    '''

    DP_preprocess_dir = Path(args.output_dir)
    work_dir = DP_preprocess_dir / 'WorkDir'
    subject_id = args.subject_id
    grayordinates = args.grayordinates
    freesurfer_home = args.freesurfer_home
    set_environ(freesurfer_home)

    input_surfdir = DP_preprocess_dir / 'Recon' / subject_id / 'surf'
    output_giidir = DP_preprocess_dir / 'BOLD' / subject_id / 'anat'
    output_giidir.mkdir(exist_ok=True, parents=True)

    shape_list = ['sulc', 'curv', 'thickness']
    surf_list = ['white', 'pial', 'sphere', 'sphere.reg']

    # convert nii to gii
    for hemi, structure in [('lh','CORTEX_LEFT'), ('rh','CORTEX_RIGHT')]:
        hemi_ = 'L' if hemi == 'lh' else 'R'
        for shape in shape_list:
            hemi_shape = input_surfdir / f'{hemi}.{shape}'
            gii_hemi_shape = output_giidir / f'{subject_id}_hemi-{hemi_}_{shape}.shape.gii'
            shape_nii2gii(shape, structure, hemi_shape, gii_hemi_shape)
        for surf in surf_list:
            hemi_surf = input_surfdir / f'{hemi}.{surf}'
            if surf == 'sphere.reg':
                gii_hemi_surf = output_giidir / f'{subject_id}_hemi-{hemi_}_desc-reg_sphere.surf.gii'
            else:
                gii_hemi_surf = output_giidir / f'{subject_id}_hemi-{hemi_}_{surf}.surf.gii'
            surf_nii2gii(hemi_surf, gii_hemi_surf)

    # resample native to fsaverage
    surface_spheres = prepare_cifti_getsurf('41k', surf_name='sphere')
    cifti_workidr = work_dir / 'cifti' / subject_id / 'anat'
    cifti_workidr.mkdir(exist_ok=True, parents=True)
    for fs6_sphere in surface_spheres:
        hemi = 'L' if 'L' in fs6_sphere else 'R'
        native_gii_hemi_sphere = output_giidir / f'{subject_id}_hemi-{hemi}_desc-reg_sphere.surf.gii'
        for shape in shape_list:
            native_gii_hemi_shape = output_giidir / f'{subject_id}_hemi-{hemi}_{shape}.shape.gii'
            fs6_gii_hemi_shape = cifti_workidr / f'{subject_id}_space-fsaverage_den-41k_hemi-{hemi}_{shape}.shape.gii'
            resample_nii2gii(native_gii_hemi_shape, native_gii_hemi_sphere, fs6_sphere, fs6_gii_hemi_shape,
                             method='BARYCENTRIC')
            fslr_data = transforms.fsaverage_to_fslr(fs6_gii_hemi_shape, '32k', hemi=hemi)
            fslr_gii_hemi_shape = cifti_workidr / f'{subject_id}_space-fsLR_den-{grayordinates}_hemi-{hemi}_{shape}.shape.gii'
            nb.save(fslr_data[0], fslr_gii_hemi_shape)
    # create cifti image
    surface_labels, metadata = _prepare_cifti(grayordinates)
    for shape in shape_list:
        scalar_surfs = (str(cifti_workidr / f'{subject_id}_space-fsLR_den-{grayordinates}_hemi-L_{shape}.shape.gii'),
                        str(cifti_workidr / f'{subject_id}_space-fsLR_den-{grayordinates}_hemi-R_{shape}.shape.gii'))
        _create_cifti_image(scalar_surfs, surface_labels, shape, metadata, output_giidir)

