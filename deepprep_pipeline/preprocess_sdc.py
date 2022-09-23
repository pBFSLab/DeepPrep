import sdcconfig as config
from pathlib import Path
import nibabel as nb
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.utils.connections import pop_file, listify


class BidsLayout:
    bids_filters = None
    _layout = None
    layout = None
    bids_database_dir = ''
    work_dir = None
    run_uuid = '1'
    bids_dir = None

    @classmethod
    def set_cls_feature(cls, bids_dir: str, work_dir: Path):
        cls.bids_dir = bids_dir
        cls.work_dir = work_dir

    @classmethod
    def init(cls):
        if cls._layout is None:
            import re
            from bids.layout.index import BIDSLayoutIndexer
            from bids.layout import BIDSLayout

            _db_path = cls.bids_database_dir or (
                    cls.work_dir / "bids_db"
            )
            _db_path.mkdir(exist_ok=True, parents=True)

            # Recommended after PyBIDS 12.1
            _indexer = BIDSLayoutIndexer(
                validate=False,
                ignore=(
                    "code",
                    "stimuli",
                    "sourcedata",
                    "models",
                    re.compile(r"^\."),
                    re.compile(
                        r"sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|dwi|eeg|ieeg|meg|perf)"
                    ),
                ),
            )
            cls._layout = BIDSLayout(
                str(cls.bids_dir),
                database_path=_db_path,
                reset_database=cls.bids_database_dir is None,
                indexer=_indexer,
            )
            cls.bids_database_dir = _db_path
        cls.layout = cls._layout
        if cls.bids_filters:
            from bids.layout import Query

            # unserialize pybids Query enum values
            for acq, filters in cls.bids_filters.items():
                cls.bids_filters[acq] = {
                    k: getattr(Query, v[7:-4])
                    if not isinstance(v, Query) and "Query" in v
                    else v
                    for k, v in filters.items()
                }


def set_config(config):
    config.workflow.ignore = []
    config.workflow.use_syn_sdc = False
    config.workflow.force_syn = False
    config.execution.task_id = 'rest'
    config.execution.echo_idx = None
    config.nipype.omp_nthreads = 1


def extract_entities(file_list):
    """
    Return a dictionary of common entities given a list of files.

    # Examples
    # --------
    # >>> extract_entities("sub-01/anat/sub-01_T1w.nii.gz")
    # {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    # >>> extract_entities(["sub-01/anat/sub-01_T1w.nii.gz"] * 2)
    # {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    # >>> extract_entities(["sub-01/anat/sub-01_run-1_T1w.nii.gz",
    # ...                   "sub-01/anat/sub-01_run-2_T1w.nii.gz"])
    # {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat',
    #  'extension': '.nii.gz'}

    """
    from collections import defaultdict
    from bids.layout import parse_file_entities

    entities = defaultdict(list)
    for e, v in [
        ev_pair
        for f in listify(file_list)
        for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}


def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024 ** 3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gb = {
        "filesize": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "largemem": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return bold_tlen, mem_gb


def _get_wf_name(bold_fname):
    """
    Derive the workflow name for supplied BOLD file.

    # >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz")
    # 'func_preproc_task_nback_wf'
    # >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz")
    # 'func_preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = "_".join(fname.split("_")[1:])
    name = "func_preproc_" + fname_nosub.replace(".", "_").replace(" ", "").replace(
        "-", "_"
    ).replace("_bold", "_wf")

    return name


def get_img_orientation(imgf):
    """Return the image orientation as a string"""
    img = nb.load(imgf)
    return "".join(nb.aff2axcodes(img.affine))


def init_func_sdc_wf(layout, bold_file, bold_mc=None, bold_mask=None, ref_image=None, has_fieldmap=True):
    from niworkflows.func.util import init_bold_reference_wf

    # Extract BIDS entities and metadata from BOLD file(s)
    entities = extract_entities(bold_file)

    # Extract metadata
    all_metadata = [layout.get_metadata(fname) for fname in listify(bold_file)]

    # Take first file as reference
    ref_file = pop_file(bold_file)
    metadata = all_metadata[0]
    # get original image orientation # TODO fMRIPrep默认都是使用的RAS
    ref_orientation = get_img_orientation(ref_file)

    echo_idxs = listify(entities.get("echo", []))
    multiecho = len(echo_idxs) > 2
    if len(echo_idxs) == 1:
        config.loggers.workflow.warning(
            f"Running a single echo <{ref_file}> from a seemingly multi-echo dataset."
        )
        bold_file = ref_file  # Just in case - drop the list

    if len(echo_idxs) == 2:
        raise RuntimeError(
            "Multi-echo processing requires at least three different echos (found two)."
        )

    if multiecho:
        # Drop echo entity for future queries, have a boolean shorthand
        entities.pop("echo", None)
        # reorder echoes from shortest to largest
        tes, bold_file = zip(
            *sorted([(layout.get_metadata(bf)["EchoTime"], bf) for bf in bold_file])
        )
        ref_file = bold_file[0]  # Reset reference to be the shortest TE

    if os.path.isfile(ref_file):
        bold_tlen, mem_gb = _create_mem_gb(ref_file)
        config.loggers.workflow.debug(
            "Creating bold processing workflow for <%s> (%.2f GB / %d TRs). "
            "Memory resampled/largemem=%.2f/%.2f GB.",
            ref_file,
            mem_gb["filesize"],
            bold_tlen,
            mem_gb["resampled"],
            mem_gb["largemem"],
        )

    wf_name = _get_wf_name(ref_file)
    workflow = Workflow(name=wf_name)

    # Find associated sbref, if possible
    entities["suffix"] = "sbref"
    entities["extension"] = [".nii", ".nii.gz"]  # Overwrite extensions
    sbref_files = layout.get(return_type="file", **entities)

    sbref_msg = f"No single-band-reference found for {os.path.basename(ref_file)}."
    if sbref_files and "sbref" in config.workflow.ignore:
        sbref_msg = "Single-band reference file(s) found and ignored."
        sbref_files = []
    elif sbref_files:
        sbref_msg = "Using single-band reference file(s) {}.".format(
            ",".join([os.path.basename(sbf) for sbf in sbref_files])
        )
    config.loggers.workflow.info(sbref_msg)

    if has_fieldmap:
        # First check if specified via B0FieldSource
        estimator_key = listify(metadata.get("B0FieldSource"))

        if not estimator_key:
            from pathlib import Path
            from sdcflows.fieldmaps import get_identifier

            # issuse#2 sdcflows 不能正常获取 fmap 估计器的bug
            # Fallback to estimators
            func_dir = Path(os.path.dirname(bold_file))
            fmap_dir = func_dir.parent / 'fmap'
            fmap_files = [str(i) for i in fmap_dir.iterdir()]
            estimator_key = get_identifier(fmap_files[0], by="sources")

        if not estimator_key:
            has_fieldmap = False
            config.loggers.workflow.critical(
                f"None of the available B0 fieldmaps are associated to <{bold_file}>"
            )
        else:
            config.loggers.workflow.info(
                f"Found usable B0-map (fieldmap) estimator(s) <{', '.join(estimator_key)}> "
                f"to correct <{bold_file}> for susceptibility-derived distortions.")
    if not has_fieldmap:
        return

    from niworkflows.interfaces.utility import KeySelect
    from sdcflows.workflows.apply.registration import init_coeff2epi_wf
    from sdcflows.workflows.apply.correction import init_unwarp_wf

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "subjects_dir",
                "subject_id",
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "fmap_id",
                "sdc_method",
            ]
        ),
        name="inputnode",
    )

    coeff2epi_wf = init_coeff2epi_wf(
        debug="fieldmaps" in config.execution.debug,
        omp_nthreads=config.nipype.omp_nthreads,
        write_coeff=True,
    )
    unwarp_wf = init_unwarp_wf(
        debug="fieldmaps" in config.execution.debug,
        omp_nthreads=config.nipype.omp_nthreads,
    )
    unwarp_wf.inputs.inputnode.metadata = metadata
    # unwarp_wf.inputs.inputnode.hmc_xforms = None  # use the already mc bold file

    # Top-level BOLD splitter
    from nipype.interfaces.fsl import Split as FSLSplit
    bold_split = pe.Node(
        FSLSplit(dimension="t"), name="bold_split", mem_gb=mem_gb["filesize"] * 3
    )
    bold_split.inputs.in_file = bold_mc
    bold_split.inputs.out_base_name = os.path.basename(bold_mc).split('.')[0]

    output_select = pe.Node(
        KeySelect(fields=["fmap", "fmap_ref", "fmap_coeff", "fmap_mask", "sdc_method"]),
        name="output_select",
        run_without_submitting=True,
    )
    output_select.inputs.key = estimator_key[0] if (len(estimator_key) > 0) else 'auto_00000'
    # if len(estimator_key) > 1:
    #     config.loggers.workflow.warning(
    #         f"Several fieldmaps <{', '.join(estimator_key)}> are "
    #         f"'IntendedFor' <{bold_file}>, using {estimator_key[0]}"
    #     )

    # sdc_report = pe.Node(
    #     SimpleBeforeAfter(
    #         before_label="Distorted",
    #         after_label="Corrected",
    #         dismiss_affine=True,
    #     ),
    #     name="sdc_report",
    #     mem_gb=0.1,
    # )
    #
    # ds_report_sdc = pe.Node(
    #     DerivativesDataSink(
    #         base_directory=fmriprep_dir,
    #         desc="sdc",
    #         suffix="bold",
    #         datatype="figures",
    #         dismiss_entities=("echo",),
    #     ),
    #     name="ds_report_sdc",
    #     run_without_submitting=True,
    # )

    bold_final = pe.Node(
        niu.IdentityInterface(fields=["bold"]),
        name="bold_final"
    )

    from nipype.interfaces.io import DataSink
    # Create DataSink object
    ds_sdc_wf = pe.Node(DataSink(), name='ds_sdc_wf')
    # Name of the output folder
    ds_sdc_wf.inputs.base_directory = os.path.dirname(bold_mc)
    # Define substitution strings
    substitutions = [('0000_unwarped_merged', '_sdc'),
                     ]
    # Feed the substitution strings to the DataSink node
    ds_sdc_wf.inputs.substitutions = substitutions

    if ref_image is not None and bold_mask is not None:
        coeff2epi_wf.inputs.inputnode.target_ref = ref_image
        coeff2epi_wf.inputs.inputnode.target_mask = bold_mask
    else:
        # Generate a tentative boldref
        initial_boldref_wf = init_bold_reference_wf(
            name="initial_boldref_wf",
            omp_nthreads=config.nipype.omp_nthreads,
            bold_file=bold_file,
            sbref_files=sbref_files,
            multiecho=multiecho,
        )
        initial_boldref_wf.inputs.inputnode.dummy_scans = config.workflow.dummy_scans
        workflow.connect([
            (initial_boldref_wf, coeff2epi_wf, [
                ("outputnode.ref_image", "inputnode.target_ref"),
                ("outputnode.bold_mask", "inputnode.target_mask")]),
        ])
    # fmt:off
    workflow.connect([
        (inputnode, output_select, [("fmap", "fmap"),
                                    ("fmap_ref", "fmap_ref"),
                                    ("fmap_coeff", "fmap_coeff"),
                                    ("fmap_mask", "fmap_mask"),
                                    ("sdc_method", "sdc_method"),
                                    ("fmap_id", "keys")]),
        (output_select, coeff2epi_wf, [
            ("fmap_ref", "inputnode.fmap_ref"),
            ("fmap_coeff", "inputnode.fmap_coeff"),
            ("fmap_mask", "inputnode.fmap_mask")]),
        # (output_select, summary, [("sdc_method", "distortion_correction")]),
        # (initial_boldref_wf, coeff2epi_wf, [
        #     ("outputnode.ref_image", "inputnode.target_ref"),
        #     ("outputnode.bold_mask", "inputnode.target_mask")]),
        (coeff2epi_wf, unwarp_wf, [
            ("outputnode.fmap_coeff", "inputnode.fmap_coeff")]),
        # (bold_hmc_wf, unwarp_wf, [
        #     ("outputnode.xforms", "inputnode.hmc_xforms")]),
        # (initial_boldref_wf, sdc_report, [
        #     ("outputnode.ref_image", "before")]),
        (bold_split, unwarp_wf, [
            ("out_files", "inputnode.distorted")]),
        # (final_boldref_wf, sdc_report, [
        #     ("outputnode.ref_image", "after"),
        #     ("outputnode.bold_mask", "wm_seg")]),
        # (inputnode, ds_report_sdc, [("bold_file", "source_file")]),
        # (sdc_report, ds_report_sdc, [("out_report", "in_file")]),
        (unwarp_wf, bold_final, [("outputnode.corrected", "bold")]),
        (unwarp_wf, ds_sdc_wf, [('outputnode.corrected', "sdc")]),  # Connect DataSink with the relevant nodes
    ])
    # fmt:on

    # fmt:off
    # workflow.connect([
    #     (unwarp_wf, bold_final, [("outputnode.corrected", "bold")]),
    #     # remaining workflow connections
    #     # (unwarp_wf, final_boldref_wf, [
    #     #     ("outputnode.corrected", "inputnode.bold_file"),
    #     # ]),
    #     # (unwarp_wf, bold_t1_trans_wf, [
    #     #     # TEMPORARY: For the moment we can't use frame-wise fieldmaps
    #     #     (("outputnode.fieldwarp", pop_file), "inputnode.fieldwarp"),
    #     # ]),
    #     # (unwarp_wf, bold_std_trans_wf, [
    #     #     # TEMPORARY: For the moment we can't use frame-wise fieldmaps
    #     #     (("outputnode.fieldwarp", pop_file), "inputnode.fieldwarp"),
    #     # ]),
    # ])
    # fmt:on

    return workflow


def sdc_single_subject_wf(layout, bids_bolds, preprocess_dir: Path, result_dir: Path, subject_id):
    from sdcflows import fieldmaps as fm

    fmap_estimators = None

    name = "single_subject_%s_wf" % subject_id
    workflow = Workflow(name=name)

    if any(("fieldmaps" not in config.workflow.ignore,
            config.workflow.use_syn_sdc,
            config.workflow.force_syn)):
        from sdcflows.utils.wrangler import find_estimators

        # SDC Step 1: Run basic heuristics to identify available data for fieldmap estimation
        # For now, no fmapless
        fmap_estimators = find_estimators(
            layout=layout,
            subject=subject_id,
            fmapless=bool(config.workflow.use_syn_sdc),
            force_fmapless=config.workflow.force_syn,
        )

        if config.workflow.use_syn_sdc and not fmap_estimators:
            message = ("Fieldmap-less (SyN) estimation was requested, but "
                       "PhaseEncodingDirection information appears to be "
                       "absent.")
            config.loggers.workflow.error(message)
            if config.workflow.use_syn_sdc == "error":
                raise ValueError(message)

        if (
            "fieldmaps" in config.workflow.ignore
            and [f for f in fmap_estimators
                 if f.method != fm.EstimatorType.ANAT]
        ):
            config.loggers.workflow.info(
                'Option "--ignore fieldmaps" was set, but either "--use-syn-sdc" '
                'or "--force-syn" were given, so fieldmap-less estimation will be executed.'
            )
            fmap_estimators = [f for f in fmap_estimators
                               if f.method == fm.EstimatorType.ANAT]

        if fmap_estimators:
            config.loggers.workflow.info(
                "B0 field inhomogeneity map will be estimated with "
                f" the following {len(fmap_estimators)} estimators: "
                f"{[e.method for e in fmap_estimators]}."
            )

    # Append the functional section to the existing anatomical exerpt
    # That way we do not need to stream down the number of bold datasets
    has_fieldmap = bool(fmap_estimators)
    if not has_fieldmap:
        return

    from sdcflows.workflows.base import init_fmap_preproc_wf

    fmap_wf = init_fmap_preproc_wf(
        debug="fieldmaps" in config.execution.debug,
        estimators=fmap_estimators,
        omp_nthreads=config.nipype.omp_nthreads,
        output_dir=result_dir,
        subject=subject_id,
    )
    fmap_wf.base_dir = config.execution.work_dir
    fmap_wf.__desc__ = f"""

Preprocessing of B<sub>0</sub> inhomogeneity mappings

: A total of {len(fmap_estimators)} fieldmaps were found available within the input
BIDS structure for this particular subject.
"""
    # from niworkflows.utils.bids import collect_data
    # subject_datas = collect_data(
    #     config.execution.layout,
    #     subject_id,
    #     task=config.execution.task_id,
    #     echo=config.execution.echo_idx,
    #     bids_filters=config.execution.bids_filters)[0]
    # func_sdc_wfs = []
    # has_fieldmap = bool(fmap_estimators)
    # for data in subject_datas['bold']:
    #     func_sdc_wf = init_func_sdc_wf(bold_file=data,
    #                                    bold_mask=None,
    #                                    ref_image=None,
    #                                    has_fieldmap=has_fieldmap)
    #     if func_sdc_wf is None:
    #         continue
    #     func_sdc_wfs.append(func_sdc_wf)
    #     break  # TODO tmp delete

    subject_datas = select_subject_bold_data(bids_bolds, preprocess_dir, subject_id)
    func_sdc_wfs = []
    has_fieldmap = bool(fmap_estimators)
    for data in subject_datas:
        func_sdc_wf = init_func_sdc_wf(layout,
                                       bold_file=data['bold_file'],
                                       bold_mc=data['bold_mc'],
                                       bold_mask=data['bold_mask'],
                                       ref_image=data['ref_image'],
                                       has_fieldmap=has_fieldmap)
        if func_sdc_wf is None:
            continue
        func_sdc_wfs.append(func_sdc_wf)
        # break  # TODO tmp delete

    for func_sdc_wf in func_sdc_wfs:
        # fmt: off
        workflow.connect([
            (fmap_wf, func_sdc_wf, [
                ("outputnode.fmap", "inputnode.fmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                ("outputnode.fmap_id", "inputnode.fmap_id"),
                ("outputnode.method", "inputnode.sdc_method"),
            ]),
        ])
        # fmt: on

    # Overwrite ``out_path_base`` of sdcflows's DataSinks
    for node in fmap_wf.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            fmap_wf.get_node(node).interface.out_path_base = ""

    # Step 3: Manually connect PEPOLAR
    for estimator in fmap_estimators:
        config.loggers.workflow.info(f"""\
Setting-up fieldmap "{estimator.bids_id}" ({estimator.method}) with \
<{', '.join(s.path.name for s in estimator.sources)}>""")

        # Mapped and phasediff can be connected internally by SDCFlows
        if estimator.method in (fm.EstimatorType.MAPPED, fm.EstimatorType.PHASEDIFF):
            continue

        suffices = [s.suffix for s in estimator.sources]

        if estimator.method == fm.EstimatorType.PEPOLAR:
            if set(suffices) == {"epi"} or sorted(suffices) == ["bold", "epi"]:
                wf_inputs = getattr(fmap_wf.inputs, f"in_{estimator.bids_id}")
                wf_inputs.in_data = [str(s.path) for s in estimator.sources]
                wf_inputs.metadata = [s.metadata for s in estimator.sources]

                # 21.0.x hack to change the number of volumes used
                # The default of 50 takes excessively long
                flatten = fmap_wf.get_node(f"wf_{estimator.bids_id}.flatten")
                flatten.inputs.max_trs = config.workflow.topup_max_vols
            else:
                raise NotImplementedError(
                    "Sophisticated PEPOLAR schemes are unsupported."
                )

    return workflow


def select_subject_bold_data(bids_bolds, preprocess_dir: Path, subject_id: str):
    """
    get bold mid data for sdc
    """
    subject_dir = preprocess_dir / f'sub-{subject_id}' / 'bold'
    datas = list()
    for idx, bids_bold in enumerate(bids_bolds):
        run = f'{idx + 1:03}'
        data = {
            'bold_file': bids_bold.path,
            'bold_mc': f'{subject_dir}/{run}/sub-{subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz',
            'bold_mask': f'{subject_dir}/{run}/sub-{subject_id}.brainmask.bin.nii.gz',
            'ref_image': f'{subject_dir}/{run}/template.nii.gz',
        }
        datas.append(data)
    return datas


def init_select_bold_wf(preprocess_dir, subject_id):
    from nipype import SelectFiles, Node

    # String template with {}-based strings
    templates = {
        'bold': '{subject_id}/bold/*/{subject_id}_bld_rest_reorient_skip_faln_mc.nii.gz',
        'bold_mask': '{subject_id}/bold/*/{subject_id}.brainmask.bin.nii.gz',
        'ref_image': '{subject_id}/bold/*/template.nii.gz',
                 }

    # Create SelectFiles node
    sf = Node(SelectFiles(templates),
              name='select_bold_wf')

    # Location of the dataset folder
    sf.inputs.base_directory = preprocess_dir

    # Feed {}-based placeholder strings with values
    sf.inputs.subject_id = subject_id

    return sf


if __name__ == '__main__':
    import os
    # FSL
    os.environ['PATH'] = '/usr/local/fsl/bin:' + os.environ['PATH']
    # ANFI
    os.environ['PATH'] = '/home/anning/abin:' + os.environ['PATH']
    # ANTs
    os.environ['PATH'] = '/usr/local/ANTs/bin:' + os.environ['PATH']
    # Convert3D
    os.environ['PATH'] = '/usr/local/c3d-1.1.0-Linux-x86_64/bin:' + os.environ['PATH']

    _bids_dir = '/mnt/ngshare/Data_Mirror/SDCFlows/MSC1'
    _preprocess_dir = Path('/mnt/ngshare/Data_Mirror/SDCFlows/MSC1/derivatives/deepprep/sub-MSC01/tmp')
    _subject_id = 'MSC01'
    set_config(config)

    _work_dir = _preprocess_dir / _subject_id / 'sdc_work_dir'
    _result_dir = _preprocess_dir / _subject_id / 'sdc_result_dir'
    bids_layout = BidsLayout()
    bids_layout.set_cls_feature(_bids_dir, _work_dir)
    bids_layout.init()

    config.execution.layout = bids_layout.layout

    _workflow = sdc_single_subject_wf(bids_layout, _preprocess_dir, _result_dir, _subject_id)
    _workflow.base_dir = _work_dir
    _workflow.write_graph(graph2use='flat', simple_form=False)
    _res = _workflow.run()
    _ = ''

    # _p = '/mnt/ngshare/DeepPrep/MSC/derivatives/deepprep/sub-MSC02/tmp'
    # _s = 'sub-MSC02'
    # _r = '001'
    # _workflow = init_select_boldref_wf(_p, _s, _r)
    # _res = _workflow.run()
    # _ = ''
