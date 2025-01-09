process bold_get_bold_file_in_bids {

    cpus 1
    memory { 500.MB * (task.attempt ** 2) }

    errorStrategy { task.exitStatus in 137..140 ? 'retry' : 'terminate' }
    maxRetries 3

    input:
    val(bids_dir)
    val(subjects_dir)
    val(output_dir)
    val(subject_id)
    val(task_id)
    val(bold_only)

    output:
    path "sub-*" // emit: subject_boldfile_txt

    script:
    script_py = "bold_get_bold_file_in_bids.py"
    """
    ${script_py} \
    --bids_dir ${bids_dir} \
    ${task_id ? "--task_id ${task_id}" : ""} \
    ${subject_id ? "--subject_id ${subject_id}" : ""} \
    --output_dir ${output_dir} \
    --work_dir ${output_dir}/WorkDir
    """
}

process bold_head_motion {
    tag "${bold_id}"

    cpus 1
    memory { 2.GB * (task.attempt ** 2) }

    errorStrategy { task.exitStatus in 137..140 ? 'retry' : 'terminate' }
    maxRetries 3

    input:
    tuple(val(subject_id), val(bold_id), val(bold_json))
    val(output_dir)
    val(work_dir)
    val(freesurfer_home)
    val(fsl_home)

    output:
    tuple(val(subject_id), val(bold_json)) // emit: head_motion_output

    script:
    script_py = "bold_head_motion.py"
    """
    ${script_py} \
    --bold_json ${bold_json} \
    --output_dir ${output_dir} \
    --work_dir ${work_dir} \
    ${freesurfer_home ? "--freesurfer_home ${freesurfer_home}" : ""} \
    ${fsl_home ? "--fsl_home ${fsl_home}" : ""}
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

    subject_boldfile_txt = bold_get_bold_file_in_bids(bids_dir, subjects_dir, qc_output_dir, subject_id, task_id, bold_only)
    (subject_id, boldfile_id, subject_boldfile_txt) = subject_boldfile_txt.flatten().multiMap { it ->
                                                                                 a: it.name.split('_')[0]
                                                                                 c: it.name
                                                                                 b: [it.name.split('_')[0], it.name.split('_bold.json')[0], it] }

    subject_id_boldfile_id = subject_id.merge(boldfile_id)
    // filter the recon result by subject_id from BOLD file.
    // this can suit for 'bold_only' or 'subjects to do Recon more than subjects to do BOLD preprocess'
    (subject_id_unique, boldfile_id_unique) = subject_id_boldfile_id.groupTuple(sort: true).multiMap { tuple ->
                                                                                                    a: tuple[0]
                                                                                                    b: tuple[1][0] }

    head_motion_output = bold_head_motion(subject_boldfile_txt, qc_output_dir, work_dir, freesurfer_home, fsl_home)
}