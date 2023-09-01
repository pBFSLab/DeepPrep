process anat_get_bold_file_in_bids {
    input:  // https://www.nextflow.io/docs/latest/process.html#inputs
    path bids_dir
    path nextflow_bin_path
    val bold_task
    output:
    path "sub-*" // emit: subject_boldfile_txt
    script:
    script_py = "${nextflow_bin_path}/anat_get_bold_file_in_bids.py"

    """
    python3 ${script_py} \
    --bids-dir ${bids_dir} \
    --task ${bold_task}
    """
}
process bold_skip_reorient {
    input:
    path bold_preprocess_path
    each path(subject_boldfile_txt)
    path nextflow_bin_path
    val nskip
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient.nii.gz")) // emit: skip_reorient
    script:
    bold_id = subject_boldfile_txt.name
    subject_id = bold_id.split('_')[0]
    script_py = "${nextflow_bin_path}/bold_skip_reorient.py"

    """
    python3 ${script_py} \
    --bold_preproces_dir ${bold_preprocess_path} \
    --boldfile_path ${subject_boldfile_txt} \
    --nskip_frame ${nskip}
    """

}
process bold_stc_mc {
    input:
    path bold_preprocess_path
    path nextflow_bin_path
    tuple(val(subject_id), val(bold_id), path(skip_reorient))
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_mc.nii.gz")) // emit: mc
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_boldref.nii.gz")) // emit: boldref
    script:
    script_py = "${nextflow_bin_path}/bold_stc_mc.py"

    """
    python3 ${script_py} \
    --bold_preproces_dir ${bold_preprocess_path} \
    --subject_id ${subject_id} \
    --skip_reorient ${skip_reorient} \
    --bold_id ${bold_id}
    """
}
process bold_bbregister {
    input:
    path subjects_dir
    path bold_preprocess_path
    path nextflow_bin_path
    tuple(val(subject_id), val(bold_id), path(mc))
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_mc_from_mc_to_fsnative_bbregister_rigid.dat")) // emit: bbregister_dat
    script:
    script_py = "${nextflow_bin_path}/bold_bbregister.py"

    """
    python3 ${script_py} \
    --subjects_dir ${subjects_dir} \
    --bold_preproces_dir ${bold_preprocess_path} \
    --subject_id ${subject_id} \
    --mc ${mc} \
    --bold_id ${bold_id}
    """

}
process bold_mkbrainmask {
    input:
    path subjects_dir
    path bold_preprocess_path
    path nextflow_bin_path
    tuple(val(subject_id), val(bold_id), path(mc), path(bbregister_dat))
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_mc.anat.wm.nii.gz")) // emit: anat_wm
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_mc.anat.csf.nii.gz")) // emit: anat_csf
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_mc.anat.aseg.nii.gz")) // emit: anat_aseg
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_mc.anat.ventricles.nii.gz")) // emit: anat_ventricles
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_mc.anat.brainmask.nii.gz")) // emit: anat_brainmask
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_mc.anat.brainmask.bin.nii.gz")) // emit: anat_brainmask_bin
    script:
    script_py = "${nextflow_bin_path}/bold_mkbrainmask.py"

    """
    python3 ${script_py} \
    --bold_preproces_dir ${bold_preprocess_path} \
    --subjects_dir ${subjects_dir} \
    --subject_id ${subject_id} \
    --mc ${mc} \
    --bbregister_dat ${bbregister_dat} \
    --bold_id ${bold_id}
    """
}
process bold_vxmregistraion{
    input:
    path subjects_dir
    path bold_preprocess_path
    path nextflow_bin_path
    each path(subject_boldfile_txt)
    val gpuid
    val atlas_type
    path vxm_model_path
    output:
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/anat/${subject_id}_norm_space-vxm${atlas_type}.nii.gz")) // emit: vxm_norm_nii
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/anat/${subject_id}_norm_space-${atlas_type}.nii.gz")) // emit: norm_nii
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/anat/${subject_id}_from_fsnative_to_vxm${atlas_type}_vxm_nonrigid.nii.gz")) // emit: vxm_nonrigid_nii
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/anat/${subject_id}_norm_affine_space-vxm${atlas_type}.npz")) // emit: vxm_affine_npz
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/anat/${subject_id}_from_fsnative_to_vxm${atlas_type}_ants_affine.mat")) // emit: vxm_fsnative_affine_mat
    script:
    subject_id = subject_boldfile_txt.name.split('_')[0]
    script_py = "${nextflow_bin_path}/bold_vxmregistraion.py"

    """
    python3 ${script_py} \
    --bold_preproces_dir ${bold_preprocess_path} \
    --subjects_dir ${subjects_dir} \
    --subject_id ${subject_id} \
    --atlas_type ${atlas_type} \
    --vxm_model_path ${vxm_model_path} \
    --gpuid ${gpuid}
    """
}
process bold_vxmregnormmni152{
    input:
    path bold_preprocess_path
    path subjects_dir
    val atlas_type
    path vxm_model_path
    path resource_dir
    path nextflow_bin_path
    val batch_size
    val gpuid
    val standard_space
    val fs_native_space
    tuple(val(subject_id), val(bold_id), path(mc), path(bbregister_dat), path(vxm_nonrigid_nii), path(vxm_fsnative_affine_mat))
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_skip_reorient_stc_mc_space-MNI152_T1_2mm.nii.gz")) // emit: bold_atlas_to_mni152
    script:
    script_py = "${nextflow_bin_path}/bold_vxmRegNormMNI152.py"

    """
    python3 ${script_py} \
    --bold_preproces_dir ${bold_preprocess_path} \
    --subjects_dir ${subjects_dir} \
    --subject_id ${subject_id} \
    --atlas_type ${atlas_type} \
    --vxm_model_path ${vxm_model_path} \
    --bold_id ${bold_id} \
    --mc ${mc} \
    --bbregister_dat ${bbregister_dat} \
    --resource_dir ${resource_dir} \
    --batch_size ${batch_size} \
    --ants_affine_trt ${vxm_fsnative_affine_mat} \
    --vxm_nonrigid_trt ${vxm_nonrigid_nii} \
    --gpuid ${gpuid} \
    --standard_space ${standard_space} \
    --fs_native_space ${fs_native_space}
    """
}

workflow {
    bids_dir = params.bids_dir
    subjects_dir = params.subjects_dir
    nextflow_bin_path = params.nextflow_bin_path
    bold_task = params.bold_task
    bold_preprocess_path = params.bold_preprocess_path
    vxm_model_path = params.vxm_model_path
    nskip = params.nskip
    atlas_type = params.atlas_type
    gpuid = params.device
    resource_dir = params.resource_dir
    batch_size = params.batch_size
    standard_space = params.standard_space
    fs_native_space = params.fs_native_space

    subject_boldfile_txt = anat_get_bold_file_in_bids(bids_dir, nextflow_bin_path, bold_task)
    skip_reorient = bold_skip_reorient(bold_preprocess_path, subject_boldfile_txt, nextflow_bin_path, nskip)
    (vxm_norm_nii, norm_nii, vxm_nonrigid_nii, vxm_affine_npz, vxm_fsnative_affine_mat) = bold_vxmregistraion(subjects_dir, bold_preprocess_path, nextflow_bin_path, subject_boldfile_txt, gpuid, atlas_type, vxm_model_path)
    (mc, boldref) = bold_stc_mc(bold_preprocess_path, nextflow_bin_path, skip_reorient)
    bbregister_dat = bold_bbregister(subjects_dir, bold_preprocess_path, nextflow_bin_path, mc)
    bold_aparaseg2mc_inputs = mc.join(bbregister_dat, by: [0,1])
    (anat_wm, anat_csf, anat_aseg, anat_ventricles, anat_brainmask, anat_brainmask_bin) = bold_mkbrainmask(subjects_dir, bold_preprocess_path, nextflow_bin_path, bold_aparaseg2mc_inputs)
    bold_vxmregnormmni152_inputs = mc.join(bbregister_dat, by: [0,1]).join(vxm_nonrigid_nii).join(vxm_fsnative_affine_mat)
    bold_vxmregnormmni152_inputs.view()
    (bold_atlas_to_mni152) = bold_vxmregnormmni152(bold_preprocess_path, subjects_dir, atlas_type, vxm_model_path, resource_dir, nextflow_bin_path, batch_size, gpuid, standard_space, fs_native_space, bold_vxmregnormmni152_inputs)
}