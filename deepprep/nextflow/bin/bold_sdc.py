#! /usr/bin/env python3
import shutil
from typing import Union, Tuple, List
import argparse
from pathlib import Path

import bids
from nipype.pipeline import engine as pe

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.utility import KeySelect


# reference : https://github.com/nipreps/fmriprep/blob/24002ef0f88e1560b26b61e30330ba570d333d37/fmriprep/utils/bids.py#L330
def dismiss_echo(entities=None):
    """Set entities to dismiss in a DerivativesDataSink."""
    if entities is None:
        entities = []

    # echo_idx = config.execution.echo_idx
    # if echo_idx is None or len(listify(echo_idx)) > 2:
    #     entities.append("echo")

    entities.append("echo")

    return entities


# reference : https://github.com/nipreps/fmriprep/blob/24002ef0f88e1560b26b61e30330ba570d333d37/fmriprep/workflows/bold/outputs.py#L469
from niworkflows.interfaces.bids import DerivativesDataSink as _DDSink


class DerivativesDataSink(_DDSink):
    out_path_base = ""


DEFAULT_MEMORY_MIN_GB = 0.01


def init_ds_registration_wf(
    *,
    bids_root: str,
    output_dir: str,
    source: str,
    dest: str,
    name: str,
) -> pe.Workflow:
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from smriprep.workflows.outputs import _bids_relative

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["source_files", "xform"]),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["xform"]), name="outputnode")

    raw_sources = pe.Node(niu.Function(function=_bids_relative), name="raw_sources")
    raw_sources.inputs.bids_root = bids_root

    ds_xform = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            mode='image',
            suffix='xfm',
            extension='.txt',
            dismiss_entities=dismiss_echo(["part"]),
            **{'from': source, 'to': dest},
        ),
        name='ds_xform',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # fmt:off
    workflow.connect([
        (inputnode, raw_sources, [('source_files', 'in_files')]),
        (inputnode, ds_xform, [('xform', 'in_file'),
                               ('source_files', 'source_file')]),
        (raw_sources, ds_xform, [('out', 'RawSources')]),
        (ds_xform, outputnode, [('out_file', 'xform')]),
    ])
    # fmt:on

    return workflow


def get_estimator(layout, fname):
    field_source = layout.get_metadata(fname).get("B0FieldSource")
    if isinstance(field_source, str):
        field_source = (field_source,)

    if field_source is None:
        import re
        from pathlib import Path

        from sdcflows.fieldmaps import get_identifier

        # Fallback to IntendedFor
        intended_rel = re.sub(r"^sub-[a-zA-Z0-9]*/", "", str(Path(fname).relative_to(layout.root)))
        field_source = get_identifier(intended_rel)

    return field_source


def map_fieldmap_estimation(
    layout: bids.BIDSLayout,
    subject_id: str,
    bold_data: List[List[str]],
    ignore_fieldmaps: bool,
    use_syn: Union[bool, str],
    force_syn: bool,
) -> Tuple[list, dict]:
    if not any((not ignore_fieldmaps, use_syn, force_syn)):
        return [], {}

    from sdcflows import fieldmaps as fm
    from sdcflows.utils.wrangler import find_estimators

    # In the case where fieldmaps are ignored and `--use-syn-sdc` is requested,
    # SDCFlows `find_estimators` still receives a full layout (which includes the fmap modality)
    # and will not calculate fmapless schemes.
    # Similarly, if fieldmaps are ignored and `--force-syn` is requested,
    # `fmapless` should be set to True to ensure BOLD targets are found to be corrected.
    fmap_estimators = find_estimators(
        layout=layout,
        subject=subject_id,
        fmapless=bool(use_syn) or ignore_fieldmaps and force_syn,
        force_fmapless=force_syn or ignore_fieldmaps and use_syn,
    )

    if not fmap_estimators:
        if use_syn:
            message = (
                "Fieldmap-less (SyN) estimation was requested, but PhaseEncodingDirection "
                "information appears to be absent."
            )
            print(message)
            if use_syn == "error":
                raise ValueError(message)
        return [], {}

    if ignore_fieldmaps and any(f.method == fm.EstimatorType.ANAT for f in fmap_estimators):
        print(
            'Option "--ignore fieldmaps" was set, but either "--use-syn-sdc" '
            'or "--force-syn" were given, so fieldmap-less estimation will be executed.'
        )
        fmap_estimators = [f for f in fmap_estimators if f.method == fm.EstimatorType.ANAT]

    # Pare down estimators to those that are actually used
    # If fmap_estimators == [], all loops/comprehensions terminate immediately
    all_ids = {fmap.bids_id for fmap in fmap_estimators}
    bold_files = (bold_series[0] for bold_series in bold_data)

    all_estimators = {
        bold_file: [fmap_id for fmap_id in get_estimator(layout, bold_file) if fmap_id in all_ids]
        for bold_file in bold_files
    }

    for bold_file, estimator_key in all_estimators.items():
        if len(estimator_key) > 1:
            print(
                f"Several fieldmaps <{', '.join(estimator_key)}> are "
                f"'IntendedFor' <{bold_file}>, using {estimator_key[0]}"
            )
            estimator_key[1:] = []

    # Final, 1-1 map, dropping uncorrected BOLD
    estimator_map = {
        bold_file: estimator_key[0]
        for bold_file, estimator_key in all_estimators.items()
        if estimator_key
    }

    fmap_estimators = [f for f in fmap_estimators if f.bids_id in estimator_map.values()]

    return fmap_estimators, estimator_map


def main(bids_path, subject_id, bold_id, bold_file_name, bold_input_file, boldref_file, hmc_xfm_file, tmp_dir, sdc_file):
    layout = bids.BIDSLayout(bids_path)
    use_syn_sdc = False
    force_syn = False
    omp_nthreads = 1

    print('bold_file_name : ', bold_file_name)
    bold_file = None
    for _f in layout.get('file', subject=subject_id, suffix="bold", extension='.nii.gz'):
        if bold_file_name in _f:
            bold_file = _f
            print('bold_file : ', bold_file)
    if not bold_file:
        print('No bold file found for subject')
        return

    bold_runs = [[bold_file]]
    ignore = []
    debug = []

    print(tmp_dir)
    workflow = Workflow(name=f'{bold_id}_wf')

    metadata = layout.get_metadata(bold_file)

    fmap_estimators, estimator_map = map_fieldmap_estimation(
        layout=layout,
        subject_id=subject_id,
        bold_data=bold_runs,
        ignore_fieldmaps="fieldmaps" in ignore,
        use_syn=use_syn_sdc,
        force_syn=force_syn,
    )

    if fmap_estimators:
        print(
            "B0 field inhomogeneity map will be estimated with the following "
            f"{len(fmap_estimators)} estimator(s): "
            f"{[e.method for e in fmap_estimators]}."
        )

        from sdcflows import fieldmaps as fm
        from sdcflows.workflows.base import init_fmap_preproc_wf

        fmap_wf = init_fmap_preproc_wf(
            debug="fieldmaps" in debug,
            estimators=fmap_estimators,
            omp_nthreads=omp_nthreads,
            output_dir=tmp_dir,
            subject=subject_id,
        )
        fmap_wf.__desc__ = f"""

    Preprocessing of B<sub>0</sub> inhomogeneity mappings

    : A total of {len(fmap_estimators)} fieldmaps were found available within the input
    BIDS structure for this particular subject.
    """

        # Overwrite ``out_path_base`` of sdcflows's DataSinks
        for node in fmap_wf.list_node_names():
            if node.split(".")[-1].startswith("ds_"):
                fmap_wf.get_node(node).interface.out_path_base = ""

        # fmap_select_std = pe.Node(
        #     KeySelect(fields=["std2anat_xfm"], key="MNI152NLin2009cAsym"),
        #     name="fmap_select_std",
        #     run_without_submitting=True,
        # )
        # if any(estimator.method == fm.EstimatorType.ANAT for estimator in fmap_estimators):
        #     # fmt:off
        #     workflow.connect([
        #         (anat_fit_wf, fmap_select_std, [
        #             ("outputnode.std2anat_xfm", "std2anat_xfm"),
        #             ("outputnode.template", "keys")]),
        #     ])
        #     # fmt:on

        for estimator in fmap_estimators:
            print(
                f"""\
    Setting-up fieldmap "{estimator.bids_id}" ({estimator.method}) with \
    <{', '.join(s.path.name for s in estimator.sources)}>"""
            )

            # Mapped and phasediff can be connected internally by SDCFlows
            if estimator.method in (fm.EstimatorType.MAPPED, fm.EstimatorType.PHASEDIFF):
                continue

            suffices = [s.suffix for s in estimator.sources]

            if estimator.method == fm.EstimatorType.PEPOLAR:
                if len(suffices) == 2 and all(suf in ("epi", "bold", "sbref") for suf in suffices):
                    wf_inputs = getattr(fmap_wf.inputs, f"in_{estimator.bids_id}")
                    wf_inputs.in_data = [str(s.path) for s in estimator.sources]
                    wf_inputs.metadata = [s.metadata for s in estimator.sources]
                else:
                    raise NotImplementedError("Sophisticated PEPOLAR schemes are unsupported.")

            # elif estimator.method == fm.EstimatorType.ANAT:
            #     from sdcflows.workflows.fit.syn import init_syn_preprocessing_wf
            #
            #     sources = [str(s.path) for s in estimator.sources if s.suffix in ("bold", "sbref")]
            #     source_meta = [
            #         s.metadata for s in estimator.sources if s.suffix in ("bold", "sbref")
            #     ]
            #     syn_preprocessing_wf = init_syn_preprocessing_wf(
            #         omp_nthreads=omp_nthreads,
            #         debug=False,
            #         auto_bold_nss=True,
            #         t1w_inversion=False,
            #         name=f"syn_preprocessing_{estimator.bids_id}",
            #     )
            #     syn_preprocessing_wf.inputs.inputnode.in_epis = sources
            #     syn_preprocessing_wf.inputs.inputnode.in_meta = source_meta
            #
            #     # fmt:off
            #     workflow.connect([
            #         (anat_fit_wf, syn_preprocessing_wf, [
            #             ("outputnode.t1w_preproc", "inputnode.in_anat"),
            #             ("outputnode.t1w_mask", "inputnode.mask_anat"),
            #         ]),
            #         (fmap_select_std, syn_preprocessing_wf, [
            #             ("std2anat_xfm", "inputnode.std2anat_xfm"),
            #         ]),
            #         (syn_preprocessing_wf, fmap_wf, [
            #             ("outputnode.epi_ref", f"in_{estimator.bids_id}.epi_ref"),
            #             ("outputnode.epi_mask", f"in_{estimator.bids_id}.epi_mask"),
            #             ("outputnode.anat_ref", f"in_{estimator.bids_id}.anat_ref"),
            #             ("outputnode.anat_mask", f"in_{estimator.bids_id}.anat_mask"),
            #             ("outputnode.sd_prior", f"in_{estimator.bids_id}.sd_prior"),
            #         ]),
            #     ])

        # fmap_wf.base_dir = tmp_dir
        # fmap_wf.run()

        fieldmap_id = estimator_map.get(bold_file)

        # https://github.com/nipreps/fmriprep/blob/24002ef0f88e1560b26b61e30330ba570d333d37/fmriprep/workflows/bold/fit.py#L860
        from bold_resampling import ReconstructFieldmap
        boldref_fmap = pe.Node(ReconstructFieldmap(inverse=[True]), name="boldref_fmap", mem_gb=1)

        # https://github.com/nipreps/fmriprep/blob/24002ef0f88e1560b26b61e30330ba570d333d37/fmriprep/workflows/bold/fit.py#L492
        from sdcflows.workflows.apply.registration import init_coeff2epi_wf
        fmapreg_wf = init_coeff2epi_wf(
            omp_nthreads=omp_nthreads,
            name="fmapreg_wf",
        )
        from niworkflows.interfaces.nitransforms import ConcatenateXFMs
        itk_mat2txt = pe.Node(ConcatenateXFMs(out_fmt="itk"), name="itk_mat2txt")

        ds_fmapreg_wf = init_ds_registration_wf(
            bids_root=layout.root,
            output_dir=tmp_dir,
            source="boldref",
            dest=fieldmap_id.replace('_', ''),
            name="ds_fmapreg_wf",
        )
        ds_fmapreg_wf.inputs.inputnode.source_files = [bold_input_file]

        # https://github.com/nipreps/fmriprep/blob/24002ef0f88e1560b26b61e30330ba570d333d37/fmriprep/workflows/bold/fit.py#L462
        from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
        enhance_boldref_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads)
        enhance_boldref_wf.inputs.inputnode.in_file = boldref_file
        # enhance_boldref_wf.inputs.inputnode.pre_mask = False
        # enhance_boldref_wf.run()
        # https://github.com/nipreps/fmriprep/blob/24002ef0f88e1560b26b61e30330ba570d333d37/fmriprep/workflows/bold/fit.py#L304
        from nipype.interfaces import utility as niu
        fmapreg_buffer = pe.Node(
            niu.IdentityInterface(fields=["boldref2fmap_xfm"]), name="fmapreg_buffer"
        )

        # https://github.com/nipreps/fmriprep/blob/24002ef0f88e1560b26b61e30330ba570d333d37/fmriprep/workflows/bold/fit.py#L817
        fmap_select = pe.Node(
            KeySelect(fields=["fmap_ref", "fmap_coeff"], key=fieldmap_id, keys=[fieldmap_id]),
            name="fmap_select",
            run_without_submitting=True,
        )

        from bold_resampling import DistortionParameters
        distortion_params = pe.Node(
            DistortionParameters(metadata=metadata, in_file=bold_input_file),
            name="distortion_params",
            run_without_submitting=True,
        )

        # https://github.com/nipreps/fmriprep/blob/24002ef0f88e1560b26b61e30330ba570d333d37/fmriprep/workflows/bold/fit.py#L840
        from bold_resampling import ResampleSeries
        boldref_bold = pe.Node(
            ResampleSeries(jacobian="fmap-jacobian" not in []),
            name="boldref_bold",
            n_procs=omp_nthreads,
            mem_gb='2GB',
        )
        boldref_bold.inputs.in_file = bold_input_file  # boldbuffer.bold_file
        boldref_bold.inputs.transforms = [hmc_xfm_file]  # bold_native_wf.inputnode.motion_xfm

        # fmt:off
        workflow.connect([
            # https://github.com/nipreps/fmriprep/blob/master/fmriprep/workflows/bold/fit.py#L511
            (enhance_boldref_wf, fmapreg_wf, [
                ('outputnode.bias_corrected_file', 'inputnode.target_ref'),  # subject_boldref
                ('outputnode.mask_file', 'inputnode.target_mask'),
            ]),
            (fmap_wf, fmapreg_wf, [
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask"),
            ]),
            (fmapreg_wf, itk_mat2txt, [('outputnode.target2fmap_xfm', 'in_xfms')]),
            (itk_mat2txt, ds_fmapreg_wf, [('out_xfm', 'inputnode.xform')]),
            (ds_fmapreg_wf, fmapreg_buffer, [('outputnode.xform', 'boldref2fmap_xfm')]),

            # 猜测: 得到的校正后的原始bold为boldref，作为配准到fmap的target_ref_file
            (enhance_boldref_wf, fmapreg_buffer, [('outputnode.bias_corrected_file', 'boldref')]),
            # 猜测: end

            (fmap_wf, fmap_select, [
                ("outputnode.fmap_coeff", "fmap_coeff"),
                ("outputnode.fmap_ref", "fmap_ref"),  # fmap_ref  filedmap 文件中的某个文件
            ]),
            (fmapreg_buffer, boldref_fmap, [
                ("boldref", "target_ref_file"),
                ("boldref2fmap_xfm", "transforms"),
            ]),
            (fmap_select, boldref_fmap, [
                ("fmap_coeff", "in_coeffs"),
                ("fmap_ref", "fmap_ref_file"),
            ]),

            (fmapreg_buffer, boldref_bold, [("boldref", "ref_file")]),

            (distortion_params, boldref_bold, [
                ("readout_time", "ro_time"),
                ("pe_direction", "pe_dir"),
            ]),

            (boldref_fmap, boldref_bold, [("out_file", "fieldmap")]),

        ])
        # fmt:on
        print('tmp_dirtmp_dirtmp_dirtmp_dirtmp_dir', tmp_dir)
        workflow.base_dir = tmp_dir
        workflow.run()
        sdc_resampled_file = list(Path(tmp_dir).glob('*/boldref_bold/sub-*.nii.gz'))[0]
        print(sdc_resampled_file)
        shutil.move(sdc_resampled_file, sdc_file)

    return


tranform_base = """#Transform {}
Transform: MatrixOffsetTransformBase_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 0 0 0
FixedParameters: 0 0 0
"""


def create_hmc_xfm_file(xfm_file, count):
    with open(xfm_file, 'w') as f:
        f.write('#Insight Transform File V1.0\n')
        for num in range(count):
            f.write(tranform_base.format(num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: Bold PreProcessing workflows -- BoldSkipReorient"
    )

    parser.add_argument("--bids_dir", required=True)
    parser.add_argument("--bold_preprocess_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--bold_id", required=True)
    parser.add_argument("--bold_file", required=True)
    parser.add_argument("--boldref_file", required=True)
    parser.add_argument("--hmc_xfm_file", required=True)
    parser.add_argument("--sdc_file", required=True, default='False')
    args = parser.parse_args()

    bids_path = args.bids_dir
    bold_preprocess_dir = args.bold_preprocess_dir
    subject_id = args.subject_id
    bold_id = args.bold_id
    bold_input_file = args.bold_file
    boldref_file = args.boldref_file
    hmc_xfm_file = args.hmc_xfm_file
    sdc_file = args.sdc_file

    sdc_tmp_dir = Path(bold_preprocess_dir) / subject_id / 'tmp' / 'sdc' / bold_id
    if not sdc_tmp_dir.exists():
        sdc_tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(hmc_xfm_file, 'r') as f:
        hmc_xfm_count = len(f.readlines())

    hmc_xfm_file = Path(bold_preprocess_dir) / subject_id / 'tmp' / 'sdc_wf' / bold_id / 'hmc_xfm.txt'
    if not hmc_xfm_file.parent.exists():
        hmc_xfm_file.parent.mkdir(parents=True, exist_ok=True)
    create_hmc_xfm_file(hmc_xfm_file, hmc_xfm_count)

    bold_file_name = Path(bold_input_file).name.replace('_space-mc', '')
    main(bids_path, subject_id.split('-')[1], bold_id, bold_file_name, bold_input_file, boldref_file, hmc_xfm_file, sdc_tmp_dir, sdc_file)
