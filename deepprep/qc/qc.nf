
process bold_get_bold_file_in_bids {

    cpus 1
    memory { 500.MB * (task.attempt ** 2) }

    errorStrategy { task.exitStatus in 137..140 ? 'retry' : 'terminate' }
    maxRetries 3

    input:  // https://www.nextflow.io/docs/latest/process.html#inputs
    val(bids_dir)
    val(subjects_dir)
    val(subjects)
    val(bold_task_type)
    val(bold_only)

    output:
    path "sub-*" // emit: subject_boldfile_txt

    script:
    script_py = "bold_get_bold_file_in_bids.py"
    if (subjects.toString() == '') {
        """
        ${script_py} \
        --bids_dir ${bids_dir} \
        --subjects_dir ${subjects_dir} \
        --task_type ${bold_task_type} \
        --bold_only ${bold_only}
        """
    }
    else {
        """
        ${script_py} \
        --bids_dir ${bids_dir} \
        --subjects_dir ${subjects_dir} \
        --subject_ids ${subjects} \
        --task_type ${bold_task_type} \
        --bold_only ${bold_only}
        """
    }
}


process bold_head_motion {
    tag "${bold_id}"

    cpus 1
    memory '8 GB'
    label 'maxForks_10'

    input:
    tuple(val(subject_id), val(bold_id), val(bold_preproc_file))
    val(freesurfer_home)
    val(fsl_home)

    val(output_dir)
    val(work_dir)
    output:
    tuple(val(subject_id), val(bold_id), val("anything"))
    script:
    script_py = "bold_head_motion.py"
    if (freesurfer_home.toString() != '') {
        freesurfer_home = "--freesurfer_home ${freesurfer_home}"
    }
    if (fsl_home.toString() != '') {
        fsl_home = "--fsl_home ${fsl_home}"
    }
    """
    ${script_py} \
    --bold_preprocess_dir ${preprocess_bids_dir} \
    ${freesurfer_home} \
    ${fsl_home} \
    --output_dir ${output_dir} \
    --work_dir ${work_dir}
    """
}


workflow {

    bids_dir = params.bids_dir
    subjects_dir = params.subjects_dir

    freesurfer_home = params.freesurfer_home  // optional
    fsl_home = params.fsl_home  // optional

    task_id = params.task_id
    subject_id = params.subject_id  // optional
    bold_only = params.bold_only  // optional

    output_dir = params.output_dir

    qc_output_dir = "${output_dir}/QuickQC"
    work_dir = "${output_dir}/WorkDir"
    if (subjects_dir.toString().toUpperCase() == '') {
        subjects_dir = "${output_dir}/Recon"
    }
    println "INFO: subjects_dir       : ${subjects_dir}"

    subject_boldfile_txt = bold_get_bold_file_in_bids(bids_dir, subjects_dir, subject_id, task_id, bold_only)
    (subject_id, boldfile_id, subject_boldfile_txt) = subject_boldfile_txt.flatten().multiMap { it ->
                                                                                 a: it.name.split('_')[0]
                                                                                 c: it.name
                                                                                 b: [it.name.split('_')[0], it] }

    subject_id_boldfile_id = subject_id.merge(boldfile_id)
    // filter the recon result by subject_id from BOLD file.
    // this can suit for 'bold_only' or 'subjects to do Recon more than subjects to do BOLD preprocess'
    (subject_id_unique, boldfile_id_unique) = subject_id_boldfile_id.groupTuple(sort: true).multiMap { tuple ->
                                                                                                    a: tuple[0]
                                                                                                    b: tuple[1][0] }

    bold_head_motion_output = bold_head_motion(subject_boldfile_txt, qc_output_dir, work_dir, freesurfer_home, fsl_home)
}
