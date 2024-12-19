#! /usr/bin/env python3
import os
import templateflow.api as tf
import typing
import nibabel as nb
import warnings
import numpy as np
import json
import argparse

from pathlib import Path
from nibabel import cifti2 as ci
from nilearn.image import resample_to_img
from neuromaps import transforms

CIFTI_STRUCT_WITH_LABELS = {  # CITFI structures with corresponding labels
    # SURFACES
    "CIFTI_STRUCTURE_CORTEX_LEFT": None,
    "CIFTI_STRUCTURE_CORTEX_RIGHT": None,
    # SUBCORTICAL
    "CIFTI_STRUCTURE_ACCUMBENS_LEFT": (26,),
    "CIFTI_STRUCTURE_ACCUMBENS_RIGHT": (58,),
    "CIFTI_STRUCTURE_AMYGDALA_LEFT": (18,),
    "CIFTI_STRUCTURE_AMYGDALA_RIGHT": (54,),
    "CIFTI_STRUCTURE_BRAIN_STEM": (16,),
    "CIFTI_STRUCTURE_CAUDATE_LEFT": (11,),
    "CIFTI_STRUCTURE_CAUDATE_RIGHT": (50,),
    "CIFTI_STRUCTURE_CEREBELLUM_LEFT": (8,),  # HCP MNI152
    "CIFTI_STRUCTURE_CEREBELLUM_RIGHT": (47,),  # HCP MNI152
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT": (28,),
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT": (60,),
    "CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT": (17,),
    "CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT": (53,),
    "CIFTI_STRUCTURE_PALLIDUM_LEFT": (13,),
    "CIFTI_STRUCTURE_PALLIDUM_RIGHT": (52,),
    "CIFTI_STRUCTURE_PUTAMEN_LEFT": (12,),
    "CIFTI_STRUCTURE_PUTAMEN_RIGHT": (51,),
    "CIFTI_STRUCTURE_THALAMUS_LEFT": (10,),
    "CIFTI_STRUCTURE_THALAMUS_RIGHT": (49,),
}


def set_environ(subjects_dir, freesurfer_home, workbench_home):
    # FreeSurfer
    os.environ['FREESURFER_HOME'] = freesurfer_home
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    os.environ['PATH'] = f'{freesurfer_home}/bin:' + f'{workbench_home}/bin_linux64:' + os.environ['PATH']


def save_json(metadata, json_oupath):
    with open(json_oupath, 'w') as f:
        json.dump(metadata, f, indent=4)


def reorient_image(img: nb.spatialimages.SpatialImage, target_ornt: str):
    """Reorient an image in memory."""

    img_axcodes = nb.aff2axcodes(img.affine)
    in_ornt = nb.orientations.axcodes2ornt(img_axcodes)
    out_ornt = nb.orientations.axcodes2ornt(target_ornt)
    ornt_xfm = nb.orientations.ornt_transform(in_ornt, out_ornt)
    r_img = img.as_reoriented(ornt_xfm)
    return r_img


def _prepare_cifti(grayordinates: str) -> typing.Tuple[list, str, dict]:
    """
    Fetch the required templates needed for CIFTI-2 generation, based on input surface density.

    Parameters
    ----------
    grayordinates :
        Total CIFTI grayordinates (91k, 170k)

    Returns
    -------
    surface_labels
        Surface label files for vertex inclusion/exclusion.
    volume_label
        Volumetric label file of subcortical structures.
    metadata
        Dictionary with BIDS metadata.
    """

    grayord_key = {
        "91k": {
            "surface-den": "32k",
            "tf-res": "02",
            "grayords": "91,282",
            "res-mm": "2mm"
        },
        "170k": {
            "surface-den": "59k",
            "tf-res": "06",
            "grayords": "170,494",
            "res-mm": "1.6mm"
        }
    }
    if grayordinates not in grayord_key:
        raise NotImplementedError(f"Grayordinates {grayordinates} is not supported.")

    tf_vol_res = grayord_key[grayordinates]['tf-res']
    total_grayords = grayord_key[grayordinates]['grayords']
    res_mm = grayord_key[grayordinates]['res-mm']
    surface_density = grayord_key[grayordinates]['surface-den']
    # Fetch templates
    surface_labels = [
        str(
            tf.get(
                "fsLR",
                density=surface_density,
                hemi=hemi,
                desc="nomedialwall",
                suffix="dparc",
                raise_empty=True,
            )
        )
        for hemi in ("L", "R")
    ]
    volume_label = str(
        tf.get(
            "MNI152NLin6Asym",
            suffix="dseg",
            atlas="HCP",
            resolution=tf_vol_res,
            raise_empty=True
        )
    )

    volume_url = str(
        tf.get(
            'MNI152NLin6Asym',
            resolution=tf_vol_res,
            suffix='T1w',
            desc=None,
            raise_empty=True,
        )
    )
    surfaces_url = [
        str(
            tf.get(
                'fsLR',
                density=surface_density,
                hemi=hemi,
                suffix='midthickness',
                desc=None,
                raise_empty=True,
            )
        )
        for hemi in ('L', 'R')
    ]
    metadata = {
        "Density": (
            f"{total_grayords} grayordinates corresponding to all of the grey matter sampled at a "
            f"{res_mm} average vertex spacing on the surface and as {res_mm} voxels subcortically"
        ),
        "SpatialReference": {
            "VolumeReference": volume_url,
            'CIFTI_STRUCTURE_CORTEX_LEFT': surfaces_url[0],
            'CIFTI_STRUCTURE_CORTEX_RIGHT': surfaces_url[1],
        }
    }
    return surface_labels, volume_label, metadata


def _create_cifti_image(
        bold_file: Path,
        volume_label: str,
        bold_surfs: typing.Tuple[str, str],
        surface_labels: typing.Tuple[str, str],
        tr: float,
        metadata: typing.Optional[dict] = None,
        cifti_savepath: Path = None,
):
    """
    Generate CIFTI image in target space.

    Parameters
    ----------
    bold_file
        BOLD volumetric timeseries
    volume_label
        Subcortical label file
    bold_surfs
        BOLD surface timeseries (L,R)
    surface_labels
        Surface label files used to remove medial wall (L,R)
    tr
        BOLD repetition time
    metadata
        Metadata to include in CIFTI header

    Returns
    -------
    out :
        BOLD data saved as CIFTI dtseries
    """
    bold_img = nb.load(bold_file)
    label_img = nb.load(volume_label)
    if label_img.shape != bold_img.shape[:3]:
        warnings.warn("Resampling bold volume to match label dimensions")
        bold_img = resample_to_img(bold_img, label_img)

    # ensure images match HCP orientation (LAS)
    bold_img = reorient_image(bold_img, target_ornt="LAS")
    label_img = reorient_image(label_img, target_ornt="LAS")

    bold_data = bold_img.get_fdata(dtype="float32")
    timepoints = bold_img.shape[3]
    label_data = np.asanyarray(label_img.dataobj).astype("int16")

    # Create brain models
    idx_offset = 0
    brainmodels = []
    bm_ts = np.empty((timepoints, 0), dtype="float32")

    for structure, labels in CIFTI_STRUCT_WITH_LABELS.items():
        if labels is None:  # surface model
            model_type = "CIFTI_MODEL_TYPE_SURFACE"
            # use the corresponding annotation
            hemi = structure.split("_")[-1]
            # currently only supports L/R cortex
            fs6_surface_bold = bold_surfs[hemi == "RIGHT"]
            hemi_lower = 'R' if hemi == "RIGHT" else 'L'
            surf_ts = transforms.fsaverage_to_fslr(fs6_surface_bold, '32k', hemi=hemi_lower)[0]
            surf_verts = len(surf_ts.darrays[0].data)
            labels = nb.load(surface_labels[hemi == "RIGHT"])
            medial = np.nonzero(labels.darrays[0].data)[0]
            # extract values across volumes
            ts = np.array([tsarr.data[medial] for tsarr in surf_ts.darrays])

            vert_idx = ci.Cifti2VertexIndices(medial)
            bm = ci.Cifti2BrainModel(
                index_offset=idx_offset,
                index_count=len(vert_idx),
                model_type=model_type,
                brain_structure=structure,
                vertex_indices=vert_idx,
                n_surface_vertices=surf_verts,
            )
            idx_offset += len(vert_idx)
            bm_ts = np.column_stack((bm_ts, ts))
        else:
            model_type = "CIFTI_MODEL_TYPE_VOXELS"
            vox = []
            ts = []
            for label in labels:
                # nonzero returns indices in row-major (C) order
                # NIfTI uses column-major (Fortran) order, so HCP generates indices in F order
                # Therefore flip the data and label the indices backwards
                k, j, i = np.nonzero(label_data.T == label)
                if k.size == 0:  # skip label if nothing matches
                    continue
                ts.append(bold_data[i, j, k])
                vox.append(np.stack([i, j, k]).T)

            vox_indices_ijk = ci.Cifti2VoxelIndicesIJK(np.concatenate(vox))
            bm = ci.Cifti2BrainModel(
                index_offset=idx_offset,
                index_count=len(vox_indices_ijk),
                model_type=model_type,
                brain_structure=structure,
                voxel_indices_ijk=vox_indices_ijk,
            )
            idx_offset += len(vox_indices_ijk)
            bm_ts = np.column_stack((bm_ts, np.concatenate(ts).T))
        # add each brain structure to list
        brainmodels.append(bm)

    # add volume information
    brainmodels.append(
        ci.Cifti2Volume(
            bold_img.shape[:3],
            ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, bold_img.affine),
        )
    )

    # generate Matrix information
    series_map = ci.Cifti2MatrixIndicesMap(
        (0,),
        "CIFTI_INDEX_TYPE_SERIES",
        number_of_series_points=timepoints,
        series_exponent=0,
        series_start=0.0,
        series_step=tr,
        series_unit="SECOND",
    )
    geometry_map = ci.Cifti2MatrixIndicesMap(
        (1,), "CIFTI_INDEX_TYPE_BRAIN_MODELS", maps=brainmodels
    )
    # provide some metadata to CIFTI matrix
    if not metadata:
        metadata = {
            "surface": "fsLR",
            "volume": "MNI152NLin6Asym",
        }
    # generate and save CIFTI image
    matrix = ci.Cifti2Matrix()
    matrix.append(series_map)
    matrix.append(geometry_map)
    matrix.metadata = ci.Cifti2MetaData(metadata)
    hdr = ci.Cifti2Header(matrix)
    img = ci.Cifti2Image(dataobj=bm_ts, header=hdr)
    img.set_data_dtype(bold_img.get_data_dtype())
    img.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES")
    json_oupath = str(cifti_savepath).replace('dtseries.nii', 'json')
    save_json(metadata, json_oupath)
    ci.save(img, str(cifti_savepath))
    return cifti_savepath


def get_space_bold(subject_id, bids_preproc, bold_orig_file):
    from bids import BIDSLayout
    assert subject_id.startswith('sub-')
    layout_preproc = BIDSLayout(str(os.path.join(bids_preproc, subject_id)),
                                config=['bids', 'derivatives'], validate=False)
    info = layout_preproc.parse_file_entities(bold_orig_file)

    left_bold_info = info.copy()
    left_bold_info['space'] = 'fsaverage6'
    left_bold_info['suffix'] = 'bold'
    left_bold_info['hemi'] = 'L'
    left_bold_info['extension'] = 'func.gii'
    left_bold_file = layout_preproc.get(**left_bold_info)[0]

    right_bold_info = info.copy()
    right_bold_info['space'] = 'fsaverage6'
    right_bold_info['suffix'] = 'bold'
    right_bold_info['hemi'] = 'R'
    right_bold_info['extension'] = 'func.gii'
    right_bold_file = layout_preproc.get(**right_bold_info)[0]

    # space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz
    mni152_bold_info = info.copy()
    mni152_bold_info['space'] = 'MNI152NLin6Asym'
    mni152_bold_info['res'] = '02'
    mni152_bold_info['desc'] = 'preproc'
    mni152_bold_info['suffix'] = 'bold'
    mni152_bold_file = layout_preproc.get(**mni152_bold_info)[0]

    repetition_time = layout_preproc.get_metadata(right_bold_file.path)['RepetitionTime']

    return left_bold_file, right_bold_file, mni152_bold_file, repetition_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create func cifti")

    parser.add_argument('--subject_id', help='subjects id', required=True)
    parser.add_argument("--bold_id", help='bold id', required=True)
    parser.add_argument("--bold_file", required=True)
    parser.add_argument('--bold_preprocess_dir', help='bold_preprocess dir', required=True)
    parser.add_argument('--subjects_dir', help='subjects dir', required=True)
    parser.add_argument('--work_dir', help='DeepPrep output dir', required=True)

    parser.add_argument('--TR', help='TR', required=False)

    parser.add_argument('--grayordinates', help='Total CIFTI grayordinates', default='91k', required=True)
    parser.add_argument('--freesurfer_home', help='freesurfer home', required=True)
    parser.add_argument('--workbench_home', help='HCP WorkBench home', default='/opt/workbench', required=False)
    args = parser.parse_args()

    '''
    Run Example:
    --subject_id sub-MSC05
    --bold_id sub-MSC05_ses-func04_task-motor_run-01_bold
    --bold_file sub-MSC05_ses-func04_task-motor_run-01_bold.nii.gz
    --bold_preprocess_dir /mnt/ngshare/DeepPrep/light_sample_output/BOLD
    --subjects_dir /mnt/ngshare/DeepPrep/light_sample_output/Recon
    --work_dir /mnt/ngshare/DeepPrep/light_sample_output_test/WorkDir
    --grayordinates 91k
    --freesurfer_home /opt/freesurfer
    --workbench_home /opt/workbench 
    '''

    bold_preprocess_dir = Path(args.bold_preprocess_dir)
    subjects_dir = Path(args.subjects_dir)
    work_dir = Path(args.work_dir)

    subject_id = args.subject_id
    grayordinates = args.grayordinates
    freesurfer_home = args.freesurfer_home
    workbench_home = args.workbench_home
    set_environ(subjects_dir, freesurfer_home, workbench_home)

    with open(args.bold_file, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    bold_orig_file = data[1]

    fs6_surface_bold_l, fs6_surface_bold_r, mni152_bold, TR = get_space_bold(subject_id=args.subject_id,
                                                                             bids_preproc=args.bold_preprocess_dir,
                                                                             bold_orig_file=bold_orig_file)

    fs6_surface_bolds = [Path(fs6_surface_bold_l.path), Path(fs6_surface_bold_r.path)]
    mni152_bold = Path(mni152_bold.path)
    output_dir = fs6_surface_bolds[0].parent
    cifti_savepath = output_dir / fs6_surface_bolds[0].name.replace('hemi-L_space-fsaverage6_bold.func.gii',
                                                                    f'space-fsLR_den-{grayordinates}_bold.dtseries.nii')

    # resample bold surf
    cifti_workidr = work_dir / 'cifti' / subject_id / 'func'
    cifti_workidr.mkdir(exist_ok=True, parents=True)

    # create bold cifti
    surface_labels, volume_labels, metadata = _prepare_cifti(grayordinates)
    print('volume_labels: ', volume_labels)
    print('surface_labels: ', surface_labels)
    out_file = _create_cifti_image(mni152_bold, volume_labels, fs6_surface_bolds, surface_labels, TR, metadata,
                                   cifti_savepath)
