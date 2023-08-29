process get_bold_file_in_bids {
    input:  // https://www.nextflow.io/docs/latest/process.html#inputs
    path bids_dir
    path nextflow_bin_path
    val bold_task

    output:
    path "sub-*" // emit: subject_boldfile_txt

    script:

    script_py = "${nextflow_bin_path}/get_bold_file_in_bids.py"
    """
    python3 ${script_py} --bids-dir ${bids_dir} --task ${bold_task}
    """
}
process bold_skip_reorient{
    input:
    path bold_preprocess_path
    each path(subject_boldfile_txt)
    path nextflow_bin_path
    val nskip

    output:
    val subject_id

    script:
    subject_id =  subject_boldfile_txt.name.split('_')[0]
    script_py = "${nextflow_bin_path}/bold_skip_reorient.py"

    """
    python3 ${script_py} --bold_preproces_dir ${bold_preprocess_path} --boldfile_path ${subject_boldfile_txt} --nskip_frame ${nskip}
    """

}

workflow{
    bids_dir = params.bids_dir
    subjects_dir = params.subjects_dir
    nextflow_bin_path = params.nextflow_bin_path
    bold_task = params.bold_task
    bold_preprocess_path = params.bold_preprocess_path
    nskip = params.nskip

    subject_boldfile_txt = get_bold_file_in_bids(bids_dir, nextflow_bin_path, bold_task)
    subject_boldfile_txt.view()
    subject_id = bold_skip_reorient(bold_preprocess_path, subject_boldfile_txt, nextflow_bin_path, nskip)
    subject_id.view()


}