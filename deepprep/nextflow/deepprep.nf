process cp_fsaverage {
    cpus 1

    input:
    path subjects_dir
    path(freesurfer_fsaverage_dir)

    output:
    path("${subjects_dir}/fsaverage")

    shell:
    """
    #! /usr/bin/env python3
    import os
    if os.path.exists('${subjects_dir}/fsaverage'):
        os.system('rm -r ${subjects_dir}/fsaverage')
    os.system('cp -nr ${freesurfer_fsaverage_dir} ${subjects_dir}')
    """
}


process anat_get_t1w_file_in_bids {
    cpus 1

    input:  // https://www.nextflow.io/docs/latest/process.html#inputs
    path bids_dir
    path nextflow_bin_path

    output:
    path "sub-*"

    script:
    script_py = "${nextflow_bin_path}/anat_get_t1w_file_in_bids.py"
    """
    python3 ${script_py} --bids-dir ${bids_dir}
    """
}

process anat_create_subject_orig_dir {
    tag "${subject_id}"

    label "maxForks_10"
    cpus 1

    input:  // https://www.nextflow.io/docs/latest/process.html#inputs
    path(subjects_dir)
    each path(subject_t1wfile_txt)
    path(nextflow_bin_path)

    output:
    val(subject_id)

    script:
    subject_id =  subject_t1wfile_txt.name
    script_py = "${nextflow_bin_path}/anat_create_subject_orig_dir.py"
    """
    python3 ${script_py} --subjects-dir ${subjects_dir} --t1wfile-path ${subject_t1wfile_txt} --deepprep-version v0.0.1
    """
}

process anat_motioncor {
    tag "${subject_id}"

    cpus 1

    input:  // https://www.nextflow.io/docs/latest/process.html#inputs
    path(subjects_dir)
    val(subject_id)

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/orig.mgz")) // emit: orig_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/rawavg.mgz")) // emit: rawavg_mgz

    script:
    threads = 1
    """
    recon-all -sd ${subjects_dir} -subject ${subject_id} -motioncor -threads ${threads} -itkthreads ${threads} -no-isrunning
    """
}


process gpu_schedule_lock {

    cpus 1
    memory '100 MB'
    cache false

    input:
    path(nextflow_bin_path)

    output:
    val(output)

    script:
    script_py = "${nextflow_bin_path}/gpu_schedule_lock.py"
    output = "create-lock"
    """
    python3 ${script_py} ${task.executor}
    """
}


process anat_segment {
    tag "${subject_id}"

    label "with_gpu"
    cpus 3
    memory '7 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(orig_mgz))

    path(fastsurfer_home)
    path(nextflow_bin_path)

    val gpu_lock

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.deep.mgz")) // emit: seg_deep_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.orig.mgz")) // emit: seg_orig_mgz

    script:
    gpu_script_py = "${nextflow_bin_path}/gpu_schedule_run.py"
    script_py = "${fastsurfer_home}/FastSurferCNN/eval.py"

    seg_deep_mgz = "${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.deep.mgz"
    seg_orig_mgz = "${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.orig.mgz"

    network_sagittal_path = "${fastsurfer_home}/checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    network_coronal_path = "${fastsurfer_home}/checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    network_axial_path = "${fastsurfer_home}/checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    """
    python3 ${gpu_script_py} single ${task.executor} ${script_py} \
    --in_name ${orig_mgz} \
    --out_name ${seg_deep_mgz} \
    --conformed_name ${subjects_dir}/${subject_id}/mri/conformed.mgz \
    --order 1 \
    --network_sagittal_path ${network_sagittal_path} \
    --network_coronal_path ${network_coronal_path} \
    --network_axial_path ${network_axial_path} \
    --batch_size 1 --simple_run --run_viewagg_on check

    cp ${seg_deep_mgz} ${seg_orig_mgz}
    """
}

process anat_reduce_to_aseg {
    tag "${subject_id}"

    cpus 2
    memory '1 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(seg_deep_mgz))

    path fastsurfer_home

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aseg.auto_noCCseg.mgz")) // emit: aseg_auto_noccseg_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/mask.mgz")) // emit: mask_mgz

    script:
    aseg_auto_noccseg_mgz = "${subjects_dir}/${subject_id}/mri/aseg.auto_noCCseg.mgz"
    mask_mgz = "${subjects_dir}/${subject_id}/mri/mask.mgz"

    script_py = "${fastsurfer_home}/recon_surf/reduce_to_aseg.py"
    """
    python3 ${script_py} \
    -i ${seg_deep_mgz} \
    -o ${aseg_auto_noccseg_mgz} \
    --outmask ${mask_mgz} \
    --fixwm
    """
}


process anat_N4_bias_correct {
    tag "${subject_id}"

    cpus 1
    memory '350 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(orig_mgz), path(mask_mgz))

    path fastsurfer_home

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/orig_nu.mgz")) // emit: orig_nu_mgz

    script:
    orig_nu_mgz = "${subjects_dir}/${subject_id}/mri/orig_nu.mgz"

    script_py = "${fastsurfer_home}/recon_surf/N4_bias_correct.py"
    threads = 1
    """
    python3 ${script_py} \
    --in ${orig_mgz} \
    --out ${orig_nu_mgz} \
    --mask ${mask_mgz} \
    --threads ${threads}
    """
}


process anat_talairach_and_nu {
    tag "${subject_id}"

    cpus 1
    memory '200 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(orig_mgz), path(orig_nu_mgz))

    val freesurfer_home

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/nu.mgz")) // emit: nu_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/transforms/talairach.auto.xfm")) // emit: talairach_auto_xfm
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/transforms/talairach.xfm")) // emit: talairach_xfm
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/transforms/talairach.xfm.lta")) // emit: talairach_xfm_lta
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/transforms/talairach_with_skull.lta")) // emit: talairach_with_skull_lta
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/transforms/talairach.lta")) // emit: talairach_lta

    script:
    threads = 1

    nu_mgz = "${subjects_dir}/${subject_id}/mri/nu.mgz"
    talairach_auto_xfm = "${subjects_dir}/${subject_id}/mri/transforms/talairach.auto.xfm"
    talairach_xfm = "${subjects_dir}/${subject_id}/mri/transforms/talairach.xfm"
    talairach_xfm_lta = "${subjects_dir}/${subject_id}/mri/transforms/talairach.xfm.lta"
    talairach_with_skull_lta = "${subjects_dir}/${subject_id}/mri/transforms/talairach_with_skull.lta"
    talairach_lta = "${subjects_dir}/${subject_id}/mri/transforms/talairach.lta"

    """
    mkdir transforms

    talairach_avi --i ${orig_nu_mgz} --xfm ${talairach_auto_xfm}
    cp ${talairach_auto_xfm} ${talairach_xfm}

    lta_convert --src ${orig_mgz} --trg ${freesurfer_home}/average/mni305.cor.mgz \
    --inxfm ${talairach_xfm} --outlta ${talairach_xfm_lta} \
    --subject fsaverage --ltavox2vox

    cp ${talairach_xfm_lta} ${talairach_with_skull_lta}
    cp ${talairach_xfm_lta} ${talairach_lta}

    mri_add_xform_to_header -c ${talairach_xfm} ${orig_nu_mgz} ${nu_mgz}
    """
}


process anat_T1 {
    tag "${subject_id}"

    cpus 1
    memory '1 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(nu_mgz))

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/T1.mgz")) // emit: t1_mgz

    script:
    threads = 1

    t1_mgz = "${subjects_dir}/${subject_id}/mri/T1.mgz"

    """
    mri_normalize -seed 1234 -g 1 -mprage ${nu_mgz} ${t1_mgz}
    """
}


process anat_brainmask {
    tag "${subject_id}"

    cpus 1

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(nu_mgz), path(mask_mgz))

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/norm.mgz")) // emit: norm_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/brainmask.mgz")) // emit: brainmask_mgz

    script:
    threads = 1
    norm_mgz = "${subjects_dir}/${subject_id}/mri/norm.mgz"
    brainmask_mgz = "${subjects_dir}/${subject_id}/mri/brainmask.mgz"

    """
    mri_mask ${nu_mgz} ${mask_mgz} ${norm_mgz}
    cp ${norm_mgz} ${brainmask_mgz}
    """
}


process anat_paint_cc_to_aseg {
    tag "${subject_id}"

    cpus 1
    memory '200 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(norm_mgz), path(seg_deep_mgz), path(aseg_auto_noccseg_mgz))

    path fastsurfer_home

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aseg.auto.mgz")) // emit: aseg_auto_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.deep.withCC.mgz")) // emit: seg_deep_withcc_mgz

    script:
    aseg_auto_mgz = "${subjects_dir}/${subject_id}/mri/aseg.auto.mgz"
    seg_deep_withcc_mgz = "${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.deep.withCC.mgz"

    script_py = "${fastsurfer_home}/recon_surf/paint_cc_into_pred.py"
    threads = 1
    """
    mri_cc -aseg aseg.auto_noCCseg.mgz -norm norm.mgz -o aseg.auto.mgz -lta cc_up.lta -sdir ${subjects_dir} ${subject_id}

    python3 ${script_py} -in_cc ${aseg_auto_mgz} -in_pred ${seg_deep_mgz} -out ${seg_deep_withcc_mgz}
    """
}


process anat_fill {
    tag "${subject_id}"

    cpus 1
    memory '1.5 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(aseg_auto_mgz), path(norm_mgz), path(brainmask_mgz), path(talairach_lta))

    val fsthreads

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aseg.presurf.mgz")) // emit: aseg_presurf_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/brain.mgz")) // emit: brain_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/brain.finalsurfs.mgz")) // emit: brain_finalsurfs_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/wm.mgz")) // emit: wm_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/filled.mgz")) // emit: filled_mgz

    script:
    threads = 1
    """
    recon-all -sd ${subjects_dir} -subject ${subject_id} \
    -asegmerge -normalization2 -maskbfs -segmentation -fill \
    -threads ${threads} -itkthreads ${threads} -no-isrunning
    """
}


process subject_id_hemi_lh {
    tag "${subject_id}"

    cpus 1

    input:
    tuple(val(subject_id), path(aseg_presurf_mgz))
    output:
    tuple(val(subject_id), val('lh'))
    script:
    """
    echo ${subject_id} lh
    """
}
process subject_id_hemi_rh {
    tag "${subject_id}"

    cpus 1

    input:
    tuple(val(subject_id), path(aseg_presurf_mgz))
    output:
    tuple(val(subject_id), val('rh'))
    script:
    """
    echo ${subject_id} rh
    """
}
process split_hemi_orig_mgz {
    tag "${subject_id}"

    cpus 1

    input:
    each hemi
    tuple(val(subject_id), path(in_data))
    output:
    tuple(val(subject_id), val(hemi), path(in_data))
    script:
    """
    echo ${in_data}
    """
}
process split_hemi_rawavg_mgz {
    tag "${subject_id}"

    cpus 1

    input:
    each hemi
    tuple(val(subject_id), path(in_data))
    output:
    tuple(val(subject_id), val(hemi), path(in_data))
    script:
    """
    echo ${in_data}
    """
}
process split_hemi_brainmask_mgz {
    tag "${subject_id}"

    cpus 1

    input:
    each hemi
    tuple(val(subject_id), path(in_data))
    output:
    tuple(val(subject_id), val(hemi), path(in_data))
    script:
    """
    echo ${in_data}
    """
}
process split_hemi_aseg_presurf_mgz {
    tag "${subject_id}"

    cpus 1

    input:
    each hemi
    tuple(val(subject_id), path(in_data))
    output:
    tuple(val(subject_id), val(hemi), path(in_data))
    script:
    """
    echo ${in_data}
    """
}
process split_hemi_brain_finalsurfs_mgz {
    tag "${subject_id}"

    cpus 1

    input:
    each hemi
    tuple(val(subject_id), path(in_data))
    output:
    tuple(val(subject_id), val(hemi), path(in_data))
    script:
    """
    echo ${in_data}
    """
}
process split_hemi_wm_mgz {
    tag "${subject_id}"

    cpus 1

    input:
    each hemi
    tuple(val(subject_id), path(in_data))
    output:
    tuple(val(subject_id), val(hemi), path(in_data))
    script:
    """
    echo ${in_data}
    """
}
process split_hemi_filled_mgz {
    tag "${subject_id}"

    cpus 1

    input:
    each hemi
    tuple(val(subject_id), path(in_data))
    output:
    tuple(val(subject_id), val(hemi), path(in_data))
    script:
    """
    echo ${in_data}
    """
}


process anat_fastcsr_levelset {
    tag "${subject_id}"

    label "with_gpu"
    cpus 1
    memory '6 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(orig_mgz),path(filled_mgz))

    val(fastcsr_home)
    path(fastcsr_model_path)
    path(nextflow_bin_path)

    each hemi

    val(gpu_lock)

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/mri/${hemi}_levelset.nii.gz"))// emit: levelset

    script:
    gpu_script_py = "${nextflow_bin_path}/gpu_schedule_run.py"
    script_py = "${fastcsr_home}/fastcsr_model_infer.py"
    threads = 1

    """
    python3 ${gpu_script_py} single ${task.executor} ${script_py} \
    --fastcsr_subjects_dir ${subjects_dir} \
    --subj ${subject_id} \
    --hemi ${hemi} \
    --model-path ${fastcsr_model_path}
    """
}


process anat_fastcsr_mksurface {
    tag "${subject_id}"

    cpus 1
    memory '4 GB'

    input:
    path(subjects_dir)

    tuple(val(subject_id), val(hemi), path(levelset_nii), path(orig_mgz), path(brainmask_mgz), path(aseg_presurf_mgz))
    val fastcsr_home

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.orig")) // emit: orig_surf
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.orig.premesh")) // emit: orig_premesh_surf

    script:
    script_py = "${fastcsr_home}/levelset2surf.py"
    threads = 1

    """
    python3 ${script_py} \
    --fastcsr_subjects_dir ${subjects_dir} \
    --subj ${subject_id} \
    --hemi ${hemi} \
    --suffix orig

    cp ${subjects_dir}/${subject_id}/surf/${hemi}.orig ${subjects_dir}/${subject_id}/surf/${hemi}.orig.premesh
    """
}


process anat_autodet_gwstats {
    tag "${subject_id}"

    cpus 1
    memory '250 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(orig_premesh_surf), path(brain_finalsurfs_mgz), path(wm_mgz))

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/autodet.gw.stats.${hemi}.dat")) // emit: autodet_gwstats

    script:

    """
    mris_autodet_gwstats --o ${subjects_dir}/${subject_id}/surf/autodet.gw.stats.${hemi}.dat \
     --i ${brain_finalsurfs_mgz} --wm ${wm_mgz} --surf ${orig_premesh_surf}
    """
}


process anat_white_surface {
    tag "${subject_id}"

    cpus 1
    memory '2 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(orig_surf), path(autodet_gwstats), path(aseg_presurf_mgz), path(brain_finalsurfs_mgz), path(wm_mgz), path(filled_mgz))

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc")) // emit: white_preaparc_surf
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.white")) // emit: white_surf
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.curv")) // emit: curv_surf
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.area")) // emit: area_surf

//     errorStrategy 'ignore'

    script:
    threads = 1

//     """
//     mris_make_surfaces -aseg aseg.presurf -white white.preaparc -whiteonly \
//     -noaparc -mgz -T1 brain.finalsurfs -SDIR ${subjects_dir} ${subject_id} ${hemi}
//
//     cp ${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc ${subjects_dir}/${subject_id}/surf/${hemi}.white
//     """

    """
    mris_place_surface --adgws-in ${autodet_gwstats} --wm ${wm_mgz} \
    --threads ${threads} --invol ${brain_finalsurfs_mgz} --${hemi} --i ${orig_surf} \
    --o ${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc \
    --white --seg ${aseg_presurf_mgz} --nsmooth 5

    mris_place_surface --curv-map ${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc 2 10 ${subjects_dir}/${subject_id}/surf/${hemi}.curv
    mris_place_surface --area-map ${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc ${subjects_dir}/${subject_id}/surf/${hemi}.area

    cp ${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc ${subjects_dir}/${subject_id}/surf/${hemi}.white
    """
}


process anat_cortex_label {
    tag "${subject_id}"

    cpus 1
    memory '250 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(white_preaparc_surf), path(aseg_presurf_mgz))

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/label/${hemi}.cortex.label")) // emit: cortex_label

    script:
    threads = 1

    """
    mri_label2label --label-cortex ${white_preaparc_surf} ${aseg_presurf_mgz} 0 ${subjects_dir}/${subject_id}/label/${hemi}.cortex.label
    """
}


process anat_cortex_hipamyg_label {
    tag "${subject_id}"

    cpus 1
    memory '300 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(white_preaparc_surf), path(aseg_presurf_mgz))

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/label/${hemi}.cortex+hipamyg.label")) // emit: cortex_hipamyg_label

    script:
    threads = 1

    """
    mri_label2label --label-cortex ${white_preaparc_surf} ${aseg_presurf_mgz} 1 ${subjects_dir}/${subject_id}/label/${hemi}.cortex+hipamyg.label
    """
}


process anat_hyporelabel {
    tag "${subject_id}"

    cpus 1
    memory '500 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(aseg_presurf_mgz), path(lh_white_surf), path(rh_white_surf))

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aseg.presurf.hypos.mgz")) // emit: aseg_presurf_hypos_mgz

    script:
    threads = 1
//     """
//     recon-all -sd ${subjects_dir} -subject ${subject_id} -hyporelabel -threads ${threads} -itkthreads ${threads}
//     """
    """
    mri_relabel_hypointensities ${aseg_presurf_mgz} ${subjects_dir}/${subject_id}/surf ${subjects_dir}/${subject_id}/mri/aseg.presurf.hypos.mgz
    """
}


process anat_smooth_inflated {
    tag "${subject_id}"

    cpus 1
    memory '250 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(white_preaparc_surf))

    output:
    tuple(val(subject_id),val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.smoothwm")) // emit: smoothwm_surf
    tuple(val(subject_id),val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.inflated")) // emit: inflated_surf
    tuple(val(subject_id),val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.sulc")) // emit: sulc_surf

    script:
    threads = 1

    """
    mris_smooth -n 3 -nw  ${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc ${subjects_dir}/${subject_id}/surf/${hemi}.smoothwm
    mris_inflate ${subjects_dir}/${subject_id}/surf/${hemi}.smoothwm ${subjects_dir}/${subject_id}/surf/${hemi}.inflated
    """
}


process anat_curv_stats {
    tag "${subject_id}"

    cpus 1
    memory '200 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(smoothwm_surf), path(curv_surf), path(sulc_surf))

    output:
    tuple(val(hemi), path("${subjects_dir}/${subject_id}/stats/${hemi}.curv.stats")) // emit: curv_stats

    script:
    threads = 1
    """
    SUBJECTS_DIR=${subjects_dir} mris_curvature_stats -m --writeCurvatureFiles -G -o "${subjects_dir}/${subject_id}/stats/${hemi}.curv.stats" -F smoothwm ${subject_id} ${hemi} curv sulc
    """
//     """
//     recon-all -sd ${subjects_dir} -subject ${subject_id} -curvstats
//     """
}


process anat_sphere {
    tag "${subject_id}"

    cpus 1
    memory '300 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(inflated_surf))

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.sphere")) // emit: sphere_surf

    script:
    threads = 1
    """
    mris_sphere -seed 1234 ${subjects_dir}/${subject_id}/surf/${hemi}.inflated ${subjects_dir}/${subject_id}/surf/${hemi}.sphere
    """
}


process anat_sphere_register {
    tag "${subject_id}"

    label "with_gpu"
    cpus 1
    memory '5 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(curv_surf), path(sulc_surf), path(sphere_surf))

    path(surfreg_home)
    path(surfreg_model_path)
    val(freesurfer_home)

    path(nextflow_bin_path)
    val(gpu_lock)

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.sphere.reg")) // emit: sphere_reg_surf

    script:
    gpu_script_py = "${nextflow_bin_path}/gpu_schedule_run.py"
    script_py = "${surfreg_home}/predict.py"
    threads = 1
    device = "cuda"

    """
    python3 ${gpu_script_py} single ${task.executor} ${script_py} --sd ${subjects_dir} --sid ${subject_id} --fsd ${freesurfer_home} \
    --hemi ${hemi} --model_path ${surfreg_model_path} --device ${device}
    """
}


process anat_jacobian {
    tag "${subject_id}"

    cpus 1

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(white_preaparc_surf), path(sphere_reg_surf))

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.jacobian_white")) // emit: jacobian_white

    script:
    """
    mris_jacobian ${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc \
    ${subjects_dir}/${subject_id}/surf/${hemi}.sphere.reg \
    ${subjects_dir}/${subject_id}/surf/${hemi}.jacobian_white
    """
}


process anat_avgcurv {
    tag "${subject_id}"

    cpus 1

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(sphere_reg_surf))

    val freesurfer_home

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.avg_curv")) // emit: avg_curv

    script:
    """
    mrisp_paint -a 5 \
    ${freesurfer_home}/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 \
    ${subjects_dir}/${subject_id}/surf/${hemi}.sphere.reg ${subjects_dir}/${subject_id}/surf/${hemi}.avg_curv
    """
}


process anat_cortparc_aparc {
    tag "${subject_id}"

    cpus 1

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(cortex_label), path(sphere_reg_surf), path(smoothwm_surf), path(aseg_presurf_mgz))
    // smoothwm_surf is hidden needed
    val freesurfer_home

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/label/${hemi}.aparc.annot")) // emit: aparc_annot

    script:
    """
    mris_ca_label -SDIR ${subjects_dir} -l ${cortex_label} -aseg ${aseg_presurf_mgz} -seed 1234 ${subject_id} ${hemi} ${sphere_reg_surf} \
    ${freesurfer_home}/average/${hemi}.curvature.buckner40.filled.desikan_killiany.2010-03-25.gcs \
    ${subjects_dir}/${subject_id}/label/${hemi}.aparc.annot
    """
}


process anat_cortparc_aparc_a2009s {
    tag "${subject_id}"

    cpus 1

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(cortex_label), path(sphere_reg_surf), path(smoothwm_surf), path(aseg_presurf_mgz))
    // smoothwm_surf is hidden needed
    val freesurfer_home

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/label/${hemi}.aparc.a2009s.annot")) // emit: aparc_a2009s_annot

    script:
    """
    mris_ca_label -SDIR ${subjects_dir} -l ${cortex_label} -aseg ${aseg_presurf_mgz} -seed 1234 ${subject_id} ${hemi} ${sphere_reg_surf} \
    ${freesurfer_home}/average/${hemi}.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs \
    ${subjects_dir}/${subject_id}/label/${hemi}.aparc.a2009s.annot
    """
}


process anat_pial_surface {
    tag "${subject_id}"

    cpus 1
    memory '1.5 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(orig_surf), path(white_surf), path(autodet_gwstats), path(cortex_hipamyg_label), path(cortex_label), path(aparc_annot), path(aseg_presurf_mgz), path(wm_mgz), path(brain_finalsurfs_mgz))

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.pial")) // emit: pial_surf
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.pial.T1")) // emit: pial_t1_surf
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.curv.pial")) // emit: curv_pial_surf
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.area.pial")) // emit: area_pial_surf
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.thickness")) // emit: thickness_surf

    script:
    threads = 1
    """
    mris_place_surface --adgws-in ${autodet_gwstats} --seg ${aseg_presurf_mgz} --threads ${threads} --wm ${wm_mgz} \
    --invol ${brain_finalsurfs_mgz} --${hemi} --i ${white_surf} --o ${subjects_dir}/${subject_id}/surf/${hemi}.pial.T1 \
    --pial --nsmooth 0 --rip-label ${subjects_dir}/${subject_id}/label/${hemi}.cortex+hipamyg.label --pin-medial-wall ${cortex_label} --aparc ${aparc_annot} --repulse-surf ${white_surf} --white-surf ${white_surf}
    cp -f ${subjects_dir}/${subject_id}/surf/${hemi}.pial.T1 ${subjects_dir}/${subject_id}/surf/${hemi}.pial

    mris_place_surface --curv-map ${subjects_dir}/${subject_id}/surf/${hemi}.pial.T1 2 10 ${subjects_dir}/${subject_id}/surf/${hemi}.curv.pial
    mris_place_surface --area-map ${subjects_dir}/${subject_id}/surf/${hemi}.pial.T1 ${subjects_dir}/${subject_id}/surf/${hemi}.area.pial
    mris_place_surface --thickness ${white_surf} ${subjects_dir}/${subject_id}/surf/${hemi}.pial.T1 20 5 ${subjects_dir}/${subject_id}/surf/${hemi}.thickness

    mris_calc -o ${subjects_dir}/${subject_id}/surf/${hemi}.area.mid ${subjects_dir}/${subject_id}/surf/${hemi}.area add ${subjects_dir}/${subject_id}/surf/${hemi}.area.pial
    mris_calc -o ${subjects_dir}/${subject_id}/surf/${hemi}.area.mid ${subjects_dir}/${subject_id}/surf/${hemi}.area.mid div 2
    SUBJECTS_DIR=${subjects_dir} mris_convert --volume ${subject_id} ${hemi} ${subjects_dir}/${subject_id}/surf/${hemi}.volume
    """
//     """
//     recon-all -sd ${subjects_dir} -subject ${subject_id} -cortex-label
//     """

}


process qc_plot_volsurf {
    tag "${subject_id}"

    cpus 5

    input:
    path subjects_dir

    tuple(val(subject_id), path(lh_white_surf), path(lh_pial_surf), path(rh_white_surf), path(rh_pial_surf), path(t1_mgz))

    path nextflow_bin_path
    path qc_result_path
    val freesurfer_home

    output:
    tuple(val(subject_id), path("${qc_plot_aseg_fig_path}"))

    script:
    qc_plot_aseg_fig_path = "${qc_result_path}/${subject_id}/figures/${subject_id}_desc-volsurf_T1w.svg"

    script_py = "${nextflow_bin_path}/qc_anat_vol_surface.py"
    vol_Surface_scene = "${nextflow_bin_path}/qc_tool/Vol_Surface.scene"
    affine_mat = "${nextflow_bin_path}/qc_tool/affine.mat"

    """
    python3 ${script_py} \
    --subject_id ${subject_id} \
    --subjects_dir ${subjects_dir} \
    --qc_result_path ${qc_result_path} \
    --affine_mat ${affine_mat} \
    --scene_file ${vol_Surface_scene} \
    --svg_outpath ${qc_plot_aseg_fig_path} \
    --freesurfer_home ${freesurfer_home}

    """
}


process qc_plot_surfparc {
    tag "${subject_id}"

    cpus 1

    input:
    path subjects_dir

    tuple(val(subject_id), path(lh_aparc_annot), path(lh_white_surf), path(lh_pial_surf), path(rh_aparc_annot),path(rh_white_surf), path(rh_pial_surf))

    path nextflow_bin_path
    path qc_result_path
    val freesurfer_home

    output:
    tuple(val(subject_id), path("${qc_plot_aseg_fig_path}"))

    script:
    qc_plot_aseg_fig_path = "${qc_result_path}/${subject_id}/figures/${subject_id}_desc-surfparc_T1w.svg"

    script_py = "${nextflow_bin_path}/qc_anat_surface_parc.py"
    surface_parc_scene = "${nextflow_bin_path}/qc_tool/Surface_parc.scene"
    affine_mat = "${nextflow_bin_path}/qc_tool/affine.mat"

    """
    python3 ${script_py} \
    --subject_id ${subject_id} \
    --subjects_dir ${subjects_dir} \
    --qc_result_path ${qc_result_path} \
    --affine_mat ${affine_mat} \
    --scene_file ${surface_parc_scene} \
    --svg_outpath ${qc_plot_aseg_fig_path} \
    --freesurfer_home ${freesurfer_home}

    """
}


process anat_pctsurfcon {
    tag "${subject_id}"

    cpus 1
    memory '250 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(cortex_label), path(white_surf), path(thickness_surf), path(rawavg_mgz), path(orig_mgz))

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/surf/${hemi}.w-g.pct.mgh")) // emit: w_g_pct_mgh
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/stats/${hemi}.w-g.pct.stats")) // emit: w_g_pct_stats

    script:
    threads = 1
//     """
//     recon-all -sd ${subjects_dir} -subject ${subject_id} -pctsurfcon -threads ${threads} -itkthreads ${threads}
//     """
    """
    SUBJECTS_DIR=${subjects_dir} pctsurfcon --s ${subject_id} --${hemi}-only
    """
}


process anat_parcstats {
    tag "${subject_id}"

    cpus 1
    memory '500 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(lh_white_surf), path(lh_cortex_label), path(lh_pial_surf), path(lh_aparc_annot), path(rh_white_surf), path(rh_cortex_label), path(rh_pial_surf), path(rh_aparc_annot), path(wm_mgz))

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/stats/lh.aparc.pial.stats")) // emit: aparc_pial_stats
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/stats/rh.aparc.pial.stats")) // emit: aparc_pial_stats
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/stats/lh.aparc.stats")) // emit: aparc_stats
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/stats/rh.aparc.stats")) // emit: aparc_stats

    script:
//     """
//     recon-all -sd ${subjects_dir} -subject ${subject_id} -parcstats -threads ${threads} -itkthreads ${threads}
//     """
    """
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${lh_cortex_label} -f ${subjects_dir}/${subject_id}/stats/lh.aparc.pial.stats -b -a ${lh_aparc_annot} -c ${subjects_dir}/${subject_id}/label/aparc.annot.ctab ${subject_id} lh pial
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${lh_cortex_label} -f ${subjects_dir}/${subject_id}/stats/lh.aparc.stats -b -a ${lh_aparc_annot} -c ${subjects_dir}/${subject_id}/label/aparc.annot.ctab ${subject_id} lh white
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${rh_cortex_label} -f ${subjects_dir}/${subject_id}/stats/rh.aparc.pial.stats -b -a ${rh_aparc_annot} -c ${subjects_dir}/${subject_id}/label/aparc.annot.ctab ${subject_id} lh pial
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${rh_cortex_label} -f ${subjects_dir}/${subject_id}/stats/rh.aparc.stats -b -a ${rh_aparc_annot} -c ${subjects_dir}/${subject_id}/label/aparc.annot.ctab ${subject_id} lh white
    """
}


process anat_parcstats2 {
    tag "${subject_id}"

    cpus 1
    memory '500 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(lh_white_surf), path(lh_cortex_label), path(lh_pial_surf), path(lh_aparc_annot), path(rh_white_surf), path(rh_cortex_label), path(rh_pial_surf), path(rh_aparc_annot), path(wm_mgz))

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/stats/lh.aparc.a2009s.pial.stats")) // emit: aparc_pial_stats
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/stats/rh.aparc.a2009s.pial.stats")) // emit: aparc_pial_stats
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/stats/lh.aparc.a2009s.stats")) // emit: aparc_stats
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/stats/rh.aparc.a2009s.stats")) // emit: aparc_stats

    script:
//     """
//     recon-all -sd ${subjects_dir} -subject ${subject_id} -parcstats -threads ${threads} -itkthreads ${threads}
//     """
    """
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${lh_cortex_label} -f ${subjects_dir}/${subject_id}/stats/lh.aparc.a2009s.pial.stats -b -a ${lh_aparc_annot} -c ${subjects_dir}/${subject_id}/label/aparc.a2009s.annot.ctab ${subject_id} lh pial
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${lh_cortex_label} -f ${subjects_dir}/${subject_id}/stats/lh.aparc.a2009s.stats -b -a ${lh_aparc_annot} -c ${subjects_dir}/${subject_id}/label/aparc.a2009s.annot.ctab ${subject_id} lh white
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${rh_cortex_label} -f ${subjects_dir}/${subject_id}/stats/rh.aparc.a2009s.pial.stats -b -a ${rh_aparc_annot} -c ${subjects_dir}/${subject_id}/label/aparc.a2009s.annot.ctab ${subject_id} lh pial
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${rh_cortex_label} -f ${subjects_dir}/${subject_id}/stats/rh.aparc.a2009s.stats -b -a ${rh_aparc_annot} -c ${subjects_dir}/${subject_id}/label/aparc.a2009s.annot.ctab ${subject_id} lh white
    """
}


process lh_anat_ribbon_inputs {
    tag "${subject_id}"

    cpus 1
    memory '1 GB'

    input:
    tuple(val(subject_id), path(aseg_presurf_mgz))
    output:
    tuple(val(subject_id), val('lh'))
    script:
    """
    echo ${aseg_presurf_mgz}
    """
}
process rh_anat_ribbon_inputs {
    tag "${subject_id}"

    cpus 1

    input:
    tuple(val(subject_id), path(aseg_presurf_mgz))
    output:
    tuple(val(subject_id), val('rh'))
    script:
    """
    echo ${aseg_presurf_mgz}
    """
}


process anat_ribbon {
    tag "${subject_id}"

    cpus 1

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(aseg_presurf_mgz), path(lh_white_surf), path(lh_pial_surf), path(rh_white_surf), path(rh_pial_surf))

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/ribbon.mgz")) // emit: ribbon_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/lh.ribbon.mgz")) // emit: lh_ribbon_mgz
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/rh.ribbon.mgz")) // emit: rh_ribbon_mgz

    script:
    threads = 1
    """
    mris_volmask --sd ${subjects_dir} --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon ${subject_id}
    """
}


process anat_apas2aseg {
    tag "${subject_id}"

    cpus 3

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(aseg_presurf_hypos_mgz), path(ribbon_mgz), path(lh_white_surf), path(lh_pial_surf), path(lh_cortex_label), path(rh_white_surf), path(rh_pial_surf), path(rh_cortex_label))

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aseg.mgz")) // emit: parcstats
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/stats/brainvol.stats")) // emit: brainvol_stats

    script:
    threads = 1
    """
    SUBJECTS_DIR=${subjects_dir} mri_surf2volseg --o ${subjects_dir}/${subject_id}/mri/aseg.mgz --i ${aseg_presurf_hypos_mgz} \
    --fix-presurf-with-ribbon ${ribbon_mgz} \
    --threads ${threads} \
    --lh-cortex-mask lh.cortex.label --lh-white lh.white --lh-pial lh.pial \
    --rh-cortex-mask rh.cortex.label --rh-white rh.white --rh-pial rh.pial

    SUBJECTS_DIR=${subjects_dir} mri_brainvol_stats ${subject_id}
    """
}


process anat_aparc2aseg {
    tag "${subject_id}"

    cpus 1
    memory '1 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(aseg_mgz), path(ribbon_mgz), path(lh_white_surf), path(lh_pial_surf), path(lh_cortex_label), path(lh_aparc_annot), path(rh_white_surf), path(rh_pial_surf), path(rh_cortex_label), path(rh_aparc_annot))

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aparc+aseg.mgz")) // emit: parcstats

    script:
    threads = 1
    """
    SUBJECTS_DIR=${subjects_dir} mri_surf2volseg --o ${subjects_dir}/${subject_id}/mri/aparc+aseg.mgz --label-cortex --i ${aseg_mgz} \
    --threads ${threads} \
    --lh-annot lh.aparc.annot 1000 \
    --lh-cortex-mask lh.cortex.label --lh-white lh.white \
    --lh-pial lh.pial \
    --rh-annot rh.aparc.annot 2000 \
    --rh-cortex-mask rh.cortex.label --rh-white rh.white \
    --rh-pial rh.pial
    """
}


process qc_plot_aparc_aseg {
    tag "${subject_id}"

    cpus 5
    memory '1 GB'

    input:
    path subjects_dir
    tuple(val(subject_id), path(norm_mgz), path(aparc_aseg))

    path nextflow_bin_path
    path qc_result_path
    val freesurfer_home

    output:
    tuple(val(subject_id), path("${qc_plot_aseg_fig_path}"))

    script:
    qc_plot_aseg_fig_path = "${qc_result_path}/${subject_id}/figures/${subject_id}_desc-volparc_T1w.svg"

    script_py = "${nextflow_bin_path}/qc_anat_aparc_aseg.py"
    freesurfer_AllLut = "${nextflow_bin_path}/qc_tool/FreeSurferAllLut.txt"
    volume_parc_scene = "${nextflow_bin_path}/qc_tool/Volume_parc.scene"
    """
    python3 ${script_py} \
    --subject_id ${subject_id} \
    --subjects_dir ${subjects_dir} \
    --qc_result_path ${qc_result_path} \
    --dlabel_info ${freesurfer_AllLut} \
    --scene_file ${volume_parc_scene} \
    --svg_outpath ${qc_plot_aseg_fig_path} \
    --freesurfer_home ${freesurfer_home}

    """
}


process anat_aparc_a2009s2aseg {
    tag "${subject_id}"

    cpus 1
    memory '1 GB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), path(aseg_mgz), path(ribbon_mgz), path(lh_white_surf), path(lh_pial_surf), path(lh_cortex_label), path(lh_aparc_annot), path(rh_white_surf), path(rh_pial_surf), path(rh_cortex_label), path(rh_aparc_annot))

    output:
    tuple(val(subject_id), path("${subjects_dir}/${subject_id}/mri/aparc.a2009s+aseg.mgz")) // emit: parcstats

    script:
    threads = 1
    """
    SUBJECTS_DIR=${subjects_dir} mri_surf2volseg --o ${subjects_dir}/${subject_id}/mri/aparc.a2009s+aseg.mgz --label-cortex --i ${aseg_mgz} \
    --threads ${threads} \
    --lh-annot lh.aparc.a2009s.annot 11100 \
    --lh-cortex-mask lh.cortex.label --lh-white lh.white \
    --lh-pial lh.pial \
    --rh-annot rh.aparc.a2009s.annot 12100 \
    --rh-cortex-mask rh.cortex.label --rh-white rh.white \
    --rh-pial rh.pial
    """
}


process anat_balabels_lh {
    tag "${subject_id}"

    cpus 1
    memory '500 MB'

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(sphere_reg_surf), path(white_surf))
    each path(label)
    path(subjects_fsaverage_dir)

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/label/${label}")) // emit: balabel

    script:
    """
    SUBJECTS_DIR=${subjects_dir} mri_label2label --srcsubject fsaverage --srclabel ${label} --trgsubject ${subject_id} --trglabel ${subjects_dir}/${subject_id}/label/${label} --hemi lh --regmethod surface
    """
}


process anat_balabels_rh {
    tag "${subject_id}"

    cpus 1

    input:
    path(subjects_dir)
    tuple(val(subject_id), val(hemi), path(sphere_reg_surf), path(white_surf))
    each path(label)
    path(subjects_fsaverage_dir)

    output:
    tuple(val(subject_id), val(hemi), path("${subjects_dir}/${subject_id}/label/${label}")) // emit: balabel

    script:
    """
    SUBJECTS_DIR=${subjects_dir} mri_label2label --srcsubject fsaverage --srclabel ${label} --trgsubject ${subject_id} --trglabel ${subjects_dir}/${subject_id}/label/${label} --hemi rh --regmethod surface
    """
}


process make_bold_preprocess_dir {
    tag "${subject_id}"

    cpus 1

    input:
    path(dir_path)

    output:
    path(dir_path)

    shell:
    """
    #! /usr/bin/env python3

    import os
    from pathlib import Path
    sd = Path('${dir_path}')
    try:
        sd.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    """
}


process bold_get_bold_file_in_bids {

    cpus 1

    input:  // https://www.nextflow.io/docs/latest/process.html#inputs
    path bids_dir
    path nextflow_bin_path
    val bold_task_type
    output:
    path "sub-*" // emit: subject_boldfile_txt

    script:
    script_py = "${nextflow_bin_path}/bold_get_bold_file_in_bids.py"

    """
    python3 ${script_py} \
    --bids_dir ${bids_dir} \
    --task_type ${bold_task_type}
    """
}


process bold_T1_to_2mm {
    tag "${subject_id}"

    cpus 1

    input:
    path subjects_dir
    path bold_preprocess_path
    path nextflow_bin_path
    tuple(val(subject_id), path(t1_mgz), path(norm_mgz))
    output:
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/func/${subject_id}_space-T1w_res-2mm_desc-skull_T1w.nii.gz")) //emit: t1_native2mm
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/func/${subject_id}_space-T1w_res-2mm_desc-noskull_T1w.nii.gz")) //emit: norm_native2mm
    script:
    script_py = "${nextflow_bin_path}/bold_T1_to_2mm.py"

    """
    python3 ${script_py} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --subjects_dir ${subjects_dir} \
    --subject_id ${subject_id} \
    --t1_mgz ${t1_mgz} \
    --norm_mgz ${norm_mgz}
    """
}


process synthmorph_affine {
    tag "${subject_id}"

    label "with_gpu"
    cpus 1
    memory '5 GB'

    input:
    path(subjects_dir)
    path(bold_preprocess_path)
    path(nextflow_bin_path)
    tuple(val(subject_id), path(t1_native2mm), val(schedule_control), val(schedule), path(schedule_control)) // schedule_control for running this process in right time
    path(synth_model_path)
    path(synth_template_path)

    val(gpu_lock)

    output:
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/func/${subject_id}_space-MNI152_res-2mm_desc-affine_T1w.nii.gz")) //emit: affine_nii
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/func/${subject_id}_from-T1w_to-MNI152_desc-affine_xfm.txt")) //emit: affine_trans

    script:
    gpu_script_py = "${nextflow_bin_path}/gpu_schedule_run.py"
    script_py = "${nextflow_bin_path}/bold_synthmorph_affine.py"
    synth_script = "${nextflow_bin_path}/mri_bold_synthmorph.py"
    """
    python3 ${gpu_script_py} double ${task.executor} ${script_py} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --subject_id ${subject_id} \
    --synth_script ${synth_script} \
    --t1_native2mm ${t1_native2mm} \
    --synth_template_path ${synth_template_path} \
    --synth_model_path ${synth_model_path}
    """
}


process synthmorph_norigid {
    tag "${subject_id}"

    cpus 2
    label "with_gpu"
    memory '5 GB'

    input:
    path(subjects_dir)
    path(bold_preprocess_path)
    path(nextflow_bin_path)
    tuple(val(subject_id), path(t1_native2mm), path(norm_native2mm), path(affine_trans))
    path(synth_model_path)
    path(synth_template_path)

    val(gpu_lock)

    output:
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/func/${subject_id}_space-MNI152_res-2mm_desc-skull_T1w.nii.gz")) //emit: t1_norigid_nii
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/func/${subject_id}_space-MNI152_res-2mm_desc-noskull_T1w.nii.gz")) //emit: norm_norigid_nii
    tuple(val(subject_id), path("${bold_preprocess_path}/${subject_id}/func/${subject_id}_from-T1w_to_MNI152_desc-nonlinear_xfm.npz")) //emit: transvoxel

    script:
    gpu_script_py = "${nextflow_bin_path}/gpu_schedule_run.py"
    script_py = "${nextflow_bin_path}/bold_synthmorph_norigid.py"
    synth_script = "${nextflow_bin_path}/mri_bold_synthmorph.py"
    """
    python3 ${gpu_script_py} double ${task.executor} ${script_py} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --subject_id ${subject_id} \
    --synth_script ${synth_script} \
    --t1_native2mm ${t1_native2mm} \
    --norm_native2mm ${norm_native2mm} \
    --affine_trans ${affine_trans} \
    --synth_template_path ${synth_template_path} \
    --synth_model_path ${synth_model_path}
    """
}


process bold_get_bold_ref_in_bids {
    tag "${subject_id}"

    cpus 1

    input:  // https://www.nextflow.io/docs/latest/process.html#inputs
    path bold_preprocess_path
    path bids_dir
    path nextflow_bin_path
    val subject_id
    val bold_id


    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${subject_id}_boldref.nii.gz")) // emit: mc

    script:
    script_py = "${nextflow_bin_path}/bold_get_bold_ref_in_bids.py"
    """
    python3 ${script_py} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --bids_dir ${bids_dir} \
    --subject_id ${subject_id} \
    --bold_id ${bold_id}
    """
}

process split_subject_boldref_file {
    tag "${subject_id}"
    cpus 1

    input:
    val(bold_id)

    output:
    tuple(val(subject_id), val(bold_id))

    script:
    subject_id = bold_id.split('_')[0]
    """
    echo
    """
}


process bold_skip_reorient {
    tag "${subject_id}"

    label "maxForks_2"
    cpus 1

    input:
    path bold_preprocess_path
    path nextflow_bin_path
    path qc_result_path
    each path(subject_boldfile_txt)
    val reorient
    val skip_frame
    val sdc
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-reorient_bold.nii.gz")) // emit: mc
    script:
    bold_id = subject_boldfile_txt.name
    subject_id = bold_id.split('_')[0]
    script_py = "${nextflow_bin_path}/bold_skip_reorient.py"

    """
    python3 ${script_py} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --qc_report_dir ${qc_result_path} \
    --boldfile_path ${subject_boldfile_txt} \
    --reorient ${reorient} \
    --skip_frame ${skip_frame} \
    --sdc ${sdc}
    """

}


process bold_stc_mc {
    tag "${subject_id}"

    cpus 1

    input:
    path bold_preprocess_path
    path nextflow_bin_path
    tuple(val(subject_id), val(bold_id), path(reorient))
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-mc_bold.nii.gz")) // emit: mc
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_from-stc_to-mc_xfm.mcdat")) // emit: mcdat
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-mc_boldref.nii.gz")) // emit: boldref
    script:
    script_py = "${nextflow_bin_path}/bold_stc_mc.py"

    """
    python3 ${script_py} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --subject_id ${subject_id} \
    --bold_id ${bold_id} \
    --reorient ${reorient}
    """
}

process bold_add_subject_id_to_bold_file {
    tag "${subject_id}"
    cpus 1

    input:
    each path(bold_file_txt)

    output:
    tuple(val(subject_id), path(bold_file_txt))

    script:
    bold_id = bold_file_txt.name
    subject_id = bold_id.split('_')[0]
    """
    echo
    """
}
process split_bold_bbregister_from_anat_input {
    tag "${subject_id}"
    cpus 1

    input:
    each path(bold_file_txt)
    val(params)

    output:
    tuple(val(subject_id), val(bold_id), val(params))

    script:
    bold_id = bold_file_txt.name
    subject_id = bold_id.split('_')[0]
    """
    echo
    """
}
process bold_bbregister {
    tag "${subject_id}"

    cpus 1
    memory '1 GB'

    input:
    path subjects_dir
    path bold_preprocess_path
    path nextflow_bin_path
    tuple(val(subject_id), val(bold_id), path(aparc_aseg_mgz), path(mc))
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_from-mc_to-T1w_desc-rigid_xfm.dat")) // emit: bbregister_dat
    script:
    script_py = "${nextflow_bin_path}/bold_bbregister.py"

    """
    python3 ${script_py} \
    --subjects_dir ${subjects_dir} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --subject_id ${subject_id} \
    --mc ${mc} \
    --bold_id ${bold_id}
    """
}


process bold_bbregister_to_native {
    tag "${subject_id}"

    cpus 1
    label "maxForks_2"
    memory '18 GB'

    input:
    path subjects_dir
    path bold_preprocess_path
    path nextflow_bin_path
    tuple(val(subject_id), val(bold_id), path(boldref), path(mc), path(t1_native2mm), path(bbregister_dat))
    path freesurfer_home
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-T1w_res-2mm_desc-rigid_bold.nii.gz")) // emit: bbregister_native_2mm
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-T1w_res-2mm_desc-rigid_boldref.nii.gz")) // emit: bbregister_native_2mm_fframe
    script:
    script_py = "${nextflow_bin_path}/bold_bbregister_to_native.py"

    """
    python3 ${script_py} \
    --subjects_dir ${subjects_dir} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --subject_id ${subject_id} \
    --ref ${boldref} \
    --moving ${mc} \
    --fixed ${t1_native2mm} \
    --dat ${bbregister_dat} \
    --bold_id ${bold_id} \
    --freesurfer-home ${freesurfer_home}
    """
}


process bold_synthmorph_norigid_apply {
    tag "${subject_id}"

    cpus 2
    label "with_gpu"
    memory '15 GB'

    input:
    path(subjects_dir)
    path(bold_preprocess_path)
    path(nextflow_bin_path)
    tuple(val(subject_id), val(bold_id), path(t1_native2mm), path(mc), path(bbregister_native_2mm), path(transvoxel))
//     path synth_model_path
    path(synth_template_path)

    val(gpu_lock)

    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-MNI152_res-2mm_bold.nii.gz")) //emit: synthmorph_norigid_bold
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-MNI152_res-2mm_boldref.nii.gz")) //emit: synthmorph_norigid_bold_fframe

    script:
    gpu_script_py = "${nextflow_bin_path}/gpu_schedule_run.py"
    script_py = "${nextflow_bin_path}/bold_synthmorph_apply.py"
    synth_script = "${nextflow_bin_path}/mri_bold_apply_synthmorph.py"
    """
    python3 ${gpu_script_py} single ${task.executor} ${script_py} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --subject_id ${subject_id} \
    --bold_id ${bold_id} \
    --T1_file ${t1_native2mm} \
    --mc ${mc} \
    --bold ${bbregister_native_2mm} \
    --trans_vox ${transvoxel} \
    --synth_template_path ${synth_template_path} \
    --synth_script ${synth_script}
    """
}


process bold_mkbrainmask {
    tag "${subject_id}"

    cpus 1

    input:
    path subjects_dir
    path bold_preprocess_path
    path nextflow_bin_path
    tuple(val(subject_id), val(bold_id), path(aparc_aseg), path(mc), path(bbregister_dat))
    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-mc_desc-wm_mask.nii.gz")) // emit: anat_wm
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-mc_desc-csf_mask.nii.gz")) // emit: anat_csf
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-mc_desc-aparcaseg_dseg.nii.gz")) // emit: anat_aseg
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-mc_desc-ventricles_mask.nii.gz")) // emit: anat_ventricles
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-mc_desc-brain_mask.nii.gz")) // emit: anat_brainmask
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_space-mc_desc-brain_maskbin.nii.gz")) // emit: anat_brainmask_bin
    script:
    script_py = "${nextflow_bin_path}/bold_mkbrainmask.py"

    """
    python3 ${script_py} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --subjects_dir ${subjects_dir} \
    --subject_id ${subject_id} \
    --mc ${mc} \
    --bbregister_dat ${bbregister_dat} \
    --bold_id ${bold_id}
    """
}


process qc_plot_mctsnr {
    tag "${subject_id}"

    cpus 1
    memory '3 GB'

    input:
    tuple(val(subject_id), val(bold_id), path(mc), path(anat_brainmask))
    path bold_preprocess_path
    path nextflow_bin_path
    path qc_result_path

    output:
    tuple(val(subject_id), val(bold_id), path("${qc_plot_mctsnr_fig_path}"))

    script:
    qc_plot_mctsnr_fig_path = "${qc_result_path}/${subject_id}/figures/${bold_id}_desc-tsnr_bold.svg"
    script_py = "${nextflow_bin_path}/qc_bold_mc_tsnr.py"
    mctsnr_scene = "${nextflow_bin_path}/qc_tool/McTSNR.scene"
    color_bar_png = "${nextflow_bin_path}/qc_tool/color_bar.png"

    """
    python3 ${script_py} \
    --subject_id ${subject_id} \
    --bold_id ${bold_id} \
    --bold_preprocess_path  ${bold_preprocess_path} \
    --qc_result_path  ${qc_result_path} \
    --mc  ${mc} \
    --mc_brainmask  ${anat_brainmask} \
    --scene_file ${mctsnr_scene} \
    --color_bar_png ${color_bar_png} \
    --svg_outpath ${qc_plot_mctsnr_fig_path}
    """

}


process qc_plot_carpet {
    tag "${subject_id}"

    cpus 1
    memory '400 MB'

    input:
    path bold_preprocess_path
    path nextflow_bin_path
    path qc_result_path
    tuple(val(subject_id), val(bold_id), path(mc_nii), path(mcdat), path(anat_aseg_nii), path(anat_brainmask_nii), path(anat_brainmask_bin_nii), path(anat_wm_nii), path(anat_csf_nii))
    output:
    tuple(val(subject_id), val(bold_id), path("${qc_result_path}/${subject_id}/figures/${bold_id}_desc-carpet_bold.svg")) // emit: bold_carpet_svg
    script:
    script_py = "${nextflow_bin_path}/bold_averagesingnal.py"
    """
    python3 ${script_py} \
    --subject_id ${subject_id} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --qc_result_path ${qc_result_path} \
    --mc ${mc_nii} \
    --mcdat ${mcdat} \
    --anat_aseg ${anat_aseg_nii} \
    --anat_brainmask ${anat_brainmask_nii} \
    --anat_brainmaskbin ${anat_brainmask_bin_nii} \
    --anat_wm ${anat_wm_nii} \
    --anat_csf ${anat_csf_nii} \
    --bold_id ${bold_id} \
    """
}


process bold_confounds {
    tag "${subject_id}"

    cpus 3
    memory '7 GB'

    input:
    path bold_preprocess_path
    path nextflow_bin_path
    tuple(val(subject_id), val(bold_id), path(mc), path(mcdat), path(anat_wm_nii), path(anat_brainmask_nii), path(anat_brainmask_bin_nii), path(anat_ventricles_nii))

    output:
    tuple(val(subject_id), val(bold_id), path("${bold_preprocess_path}/${subject_id}/func/${bold_id}_desc-confounds_timeseries.txt")) // emit: bold_confounds_view

    script:
    script_py = "${nextflow_bin_path}/bold_cal_confounds.py"

    """
    python3 ${script_py} \
    --bold_preprocess_dir ${bold_preprocess_path} \
    --subject_id ${subject_id} \
    --bold_id ${bold_id} \
    --bold_file ${mc} \
    --mcdat ${mcdat} \
    --aseg_wm ${anat_wm_nii} \
    --aseg_brainmask ${anat_brainmask_nii} \
    --aseg_brainmask_bin ${anat_brainmask_bin_nii} \
    --aseg_ventricles ${anat_ventricles_nii}
    """
}


process qc_plot_norm_to_mni152 {
    tag "${subject_id}"

    cpus 1

    input:
    tuple(val(subject_id), path(norm_to_mni152_nii))
    path bold_preprocess_path
    path nextflow_bin_path
    path qc_result_path

    output:
    tuple(val(subject_id), path("${qc_plot_norm_to_mni152_fig_path}"))

    script:
    script_py = "${nextflow_bin_path}/qc_bold_norm_to_mni152.py"
    normtoMNI152_scene = "${nextflow_bin_path}/qc_tool/NormtoMNI152.scene"
    mni152_norm_png = "${nextflow_bin_path}/qc_tool/MNI152_norm.png"
    qc_plot_norm_to_mni152_fig_path = "${qc_result_path}/${subject_id}/figures/${subject_id}_desc-T1toMNI152_combine.svg"

    """
    python3 ${script_py} \
    --subject_id ${subject_id} \
    --bold_preprocess_path  ${bold_preprocess_path} \
    --qc_result_path  ${qc_result_path} \
    --norm_to_mni152 ${norm_to_mni152_nii} \
    --scene_file ${normtoMNI152_scene} \
    --mni152_norm_png ${mni152_norm_png} \
    --svg_outpath ${qc_plot_norm_to_mni152_fig_path}

    """
}


process qc_plot_bold_to_space {
    tag "${subject_id}"

    cpus 5
    memory '1.5 GB'

    input:
    tuple(val(subject_id), val(bold_id), path(bold_atlas_to_mni152))
    tuple(val(subject_id), val(bold_id), path(bold_to_T1w))
    val fs_native_space
    path subjects_dir
    path bold_preprocess_path
    path nextflow_bin_path
    path qc_result_path
    val freesurfer_home

    output:
    tuple(val(subject_id), val(bold_id), path("${qc_plot_norm_to_mni152_fig_path}"))

    script:
    qc_plot_norm_to_mni152_fig_path = "${qc_result_path}/${subject_id}/figures/${bold_id}_desc-reg2MNI152_bold.svg"
    script_py = "${nextflow_bin_path}/qc_bold_to_space.py"
    qc_tool_package = "${nextflow_bin_path}/qc_tool"

    """
    python3 ${script_py} \
    --subject_id ${subject_id} \
    --bold_id ${bold_id} \
    --fs_native_space ${fs_native_space} \
    --subjects_dir ${subjects_dir} \
    --bold_preprocess_path  ${bold_preprocess_path} \
    --qc_result_path  ${qc_result_path} \
    --space_mni152_bold_path  ${bold_atlas_to_mni152} \
    --space_t1w_bold_path  ${bold_to_T1w} \
    --qc_tool_package  ${qc_tool_package} \
    --svg_outpath ${qc_plot_norm_to_mni152_fig_path} \
    --freesurfer_home ${freesurfer_home}

    """
}


process qc_anat_create_report {
    tag "${subject_id}"

    cpus 1

    input:
    tuple(val(subject_id), path(aparc_aseg_svg))
    path nextflow_bin_path
    path bids_dir
    path subjects_dir
    path qc_result_path
    val deepprep_version
    output:
    tuple(val(subject_id), path("${qc_result_path}/${subject_id}/${subject_id}.html")) // emit: qc_report
    script:
    script_py = "${nextflow_bin_path}/qc_create_report.py"

    """
    python3 ${script_py} \
    --nextflow_bin_path ${nextflow_bin_path} \
    --subject_id ${subject_id} \
    --bids_dir ${bids_dir} \
    --subjects_dir ${subjects_dir} \
    --qc_result_path ${qc_result_path} \
    --deepprep_version ${deepprep_version} \
    --nextflow_log ${launchDir}/.nextflow.log
    """

}


process qc_bold_create_report {
    tag "${subject_id}"

    cpus 1

    input:
    tuple(val(subject_id), val(bold_id), val(bold_confounds))
    path nextflow_bin_path
    path bids_dir
    path subjects_dir
    path qc_result_path
    val bold_task_type
    val deepprep_version
    output:
    tuple(val(subject_id), path("${qc_result_path}/${subject_id}/${subject_id}.html")) // emit: qc_report
    script:
    script_py = "${nextflow_bin_path}/qc_create_report.py"

    """
    python3 ${script_py} \
    --nextflow_bin_path ${nextflow_bin_path} \
    --subject_id ${subject_id} \
    --bids_dir ${bids_dir} \
    --subjects_dir ${subjects_dir} \
    --qc_result_path ${qc_result_path} \
    --bold_task_type ${bold_task_type} \
    --deepprep_version ${deepprep_version} \
    --nextflow_log ${launchDir}/.nextflow.log
    """

}


workflow anat_wf {

    take:
    gpu_lock

    main:
    bids_dir = params.bids_dir
    subjects_dir = params.subjects_dir
    qc_result_path = params.qc_result_path
    bold_task_type = params.bold_task_type

    fsthreads = params.anat_fsthreads

    fastsurfer_home = params.fastsurfer_home

    freesurfer_home = params.freesurfer_home
    freesurfer_fsaverage_dir = params.freesurfer_fsaverage_dir

    fastcsr_home = params.fastcsr_home
    fastcsr_model_path = params.fastcsr_model_path

    surfreg_home = params.surfreg_home
    surfreg_model_path = params.surfreg_model_path

    nextflow_bin_path = params.nextflow_bin_path

    deepprep_version = params.deepprep_version

    // BIDS and SUBJECTS_DIR
    _directory = new File(subjects_dir)
    if (!_directory.exists()) {
        _directory.mkdirs()
        println "create dir: ..."
        println _directory
    }
    _directory = new File(qc_result_path)
    if (!_directory.exists()) {
        _directory.mkdirs()
        println "create dir: ..."
        println _directory
    }
    subject_t1wfile_txt = anat_get_t1w_file_in_bids(bids_dir, nextflow_bin_path)
    subjects_fsaverage_dir = cp_fsaverage(subjects_dir, freesurfer_fsaverage_dir)
    subject_id = anat_create_subject_orig_dir(subjects_dir, subject_t1wfile_txt, nextflow_bin_path)

    // freesurfer
    (orig_mgz, rawavg_mgz) = anat_motioncor(subjects_dir, subject_id)

    // fastsurfer
    (seg_deep_mgz, seg_orig_mgz) = anat_segment(subjects_dir, orig_mgz, fastsurfer_home, nextflow_bin_path, gpu_lock)
    (aseg_auto_noccseg_mgz, mask_mgz) = anat_reduce_to_aseg(subjects_dir, seg_deep_mgz, fastsurfer_home)

    anat_N4_bias_correct_input = orig_mgz.join(mask_mgz)
    orig_nu_mgz = anat_N4_bias_correct(subjects_dir, anat_N4_bias_correct_input, fastsurfer_home)

    anat_talairach_and_nu_input = orig_mgz.join(orig_nu_mgz)
    (nu_mgz, talairach_auto_xfm, talairach_xfm, talairach_xfm_lta, talairach_with_skull_lta, talairach_lta) = anat_talairach_and_nu(subjects_dir, anat_talairach_and_nu_input, freesurfer_home)

    t1_mgz = anat_T1(subjects_dir, nu_mgz)

    anat_brainmask_input = nu_mgz.join(mask_mgz)
    (norm_mgz, brainmask_mgz) = anat_brainmask(subjects_dir, anat_brainmask_input)

    anat_paint_cc_to_aseg_input = norm_mgz.join(seg_deep_mgz).join(aseg_auto_noccseg_mgz)
    (aseg_auto_mgz, seg_deep_withcc_mgz) = anat_paint_cc_to_aseg(subjects_dir, anat_paint_cc_to_aseg_input, fastsurfer_home)

    // freesurfer
    anat_paint_cc_to_aseg_input = aseg_auto_mgz.join(norm_mgz).join(brainmask_mgz).join(talairach_lta)
    (aseg_presurf_mgz, brain_mgz, brain_finalsurfs_mgz, wm_mgz, filled_mgz) = anat_fill(subjects_dir, anat_paint_cc_to_aseg_input, fsthreads)

    // ####################### Split two channel #######################
    hemis = Channel.of('lh', 'rh')
    subject_id_lh = subject_id_hemi_lh(aseg_presurf_mgz)
    subject_id_rh = subject_id_hemi_rh(aseg_presurf_mgz)
    hemis_orig_mgz = split_hemi_orig_mgz(hemis, orig_mgz)
    hemis_rawavg_mgz = split_hemi_rawavg_mgz(hemis, rawavg_mgz)
    hemis_brainmask_mgz = split_hemi_brainmask_mgz(hemis, brainmask_mgz)
    hemis_aseg_presurf_mgz = split_hemi_aseg_presurf_mgz(hemis, aseg_presurf_mgz)
    hemis_brain_finalsurfs_mgz = split_hemi_brain_finalsurfs_mgz(hemis, brain_finalsurfs_mgz)
    hemis_wm_mgz = split_hemi_wm_mgz(hemis, wm_mgz)
    hemis_filled_mgz = split_hemi_filled_mgz(hemis, filled_mgz)

    // fastcsr
    // hemi levelset_nii
    anat_fastcsr_levelset_input = orig_mgz.join(filled_mgz)
    levelset_nii = anat_fastcsr_levelset(subjects_dir, anat_fastcsr_levelset_input, fastcsr_home, fastcsr_model_path, nextflow_bin_path, hemis, gpu_lock)

    // hemi orig_surf, orig_premesh_surf
    anat_fastcsr_mksurface_input = levelset_nii.join(hemis_orig_mgz, by: [0, 1]).join(hemis_brainmask_mgz, by: [0, 1]).join(hemis_aseg_presurf_mgz, by: [0, 1])
    (orig_surf, orig_premesh_surf) = anat_fastcsr_mksurface(subjects_dir, anat_fastcsr_mksurface_input, fastcsr_home)

    // hemi autodet_gwstats
    anat_autodet_gwstats_input = orig_premesh_surf.join(hemis_brain_finalsurfs_mgz, by: [0, 1]).join(hemis_wm_mgz, by: [0, 1])
    autodet_gwstats = anat_autodet_gwstats(subjects_dir, anat_autodet_gwstats_input)

    // hemi white_preaparc_surf, white_surf, curv_surf, area_surf
    anat_white_surface_input = orig_surf.join(autodet_gwstats, by: [0, 1]).join(hemis_aseg_presurf_mgz, by: [0, 1]).join(hemis_brain_finalsurfs_mgz, by: [0, 1]).join(hemis_wm_mgz, by: [0, 1]).join(hemis_filled_mgz, by: [0, 1])
    (white_preaparc_surf, white_surf, curv_surf, area_surf) = anat_white_surface(subjects_dir, anat_white_surface_input)

    // hemi aseg_presurf_mgz, white_preaparc_surf
    anat_cortex_hipamyg_label_input = white_preaparc_surf.join(hemis_aseg_presurf_mgz, by: [0, 1])
    cortex_label = anat_cortex_label(subjects_dir, anat_cortex_hipamyg_label_input)
    cortex_hipamyg_label = anat_cortex_hipamyg_label(subjects_dir, anat_cortex_hipamyg_label_input)

    // anat_hyporelabel
    lh_anat_hyporelabel_input = white_surf.join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2] }
    rh_anat_hyporelabel_input = white_surf.join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2] }
    anat_hyporelabel_inputs = aseg_presurf_mgz.join(lh_anat_hyporelabel_input).join(rh_anat_hyporelabel_input)
    aseg_presurf_hypos_mgz = anat_hyporelabel(subjects_dir, anat_hyporelabel_inputs)

    // hemi white_preaparc_surf
    (smoothwm_surf, inflated_surf, sulc_surf) = anat_smooth_inflated(subjects_dir, white_preaparc_surf)

    // smoothwm_surf, inflated_surf, sulc_surf
    anat_curv_stats_input = smoothwm_surf.join(curv_surf, by: [0, 1]).join(sulc_surf, by: [0, 1])
    curv_stats = anat_curv_stats(subjects_dir, anat_curv_stats_input)
    sphere_surf = anat_sphere(subjects_dir, inflated_surf)

    // curv_surf, sulc_surf, sphere_surf
    anat_sphere_register_input = curv_surf.join(sulc_surf, by: [0, 1]).join(sphere_surf, by: [0, 1])
    sphere_reg_surf = anat_sphere_register(subjects_dir, anat_sphere_register_input, surfreg_home, surfreg_model_path, freesurfer_home, nextflow_bin_path, gpu_lock)

    // white_preaparc_surf, sphere_reg_surf
    anat_jacobian_input = white_preaparc_surf.join(sphere_reg_surf, by: [0, 1])
    jacobian_white = anat_jacobian(subjects_dir, anat_jacobian_input)

//     avg_curv = anat_avgcurv(subjects_dir, sphere_reg_surf, freesurfer_home)  // if for paper, comment out

    // cortex_label, sphere_reg_surf, smoothwm_surf
    anat_cortparc_aparc_input = cortex_label.join(sphere_reg_surf, by: [0, 1]).join(smoothwm_surf, by: [0, 1]).join(hemis_aseg_presurf_mgz, by: [0, 1])
    aparc_annot = anat_cortparc_aparc(subjects_dir, anat_cortparc_aparc_input, freesurfer_home)

//     aparc_a2009s_annot = anat_cortparc_aparc_a2009s(subjects_dir, anat_cortparc_aparc_input, freesurfer_home)  // if for paper, comment out

    anat_pial_surface_input = orig_surf.join(white_surf, by: [0, 1]).join(autodet_gwstats, by: [0, 1]).join(cortex_hipamyg_label, by: [0, 1]).join(cortex_label, by: [0, 1]).join(aparc_annot, by: [0, 1]).join(hemis_aseg_presurf_mgz, by: [0, 1]).join(hemis_wm_mgz, by: [0, 1]).join(hemis_brain_finalsurfs_mgz, by: [0, 1])
    (pial_surf, pial_t1_surf, curv_pial_surf, area_pial_surf, thickness_surf) = anat_pial_surface(subjects_dir, anat_pial_surface_input)

    qc_plot_volsurf_input_lh = white_surf.join(pial_surf, by: [0, 1]).join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3] }
    qc_plot_volsurf_input_rh = white_surf.join(pial_surf, by: [0, 1]).join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3] }
    qc_plot_volsurf_input = qc_plot_volsurf_input_lh.join(qc_plot_volsurf_input_rh).join(t1_mgz)
    vol_surface_svg = qc_plot_volsurf(subjects_dir, qc_plot_volsurf_input, nextflow_bin_path, qc_result_path, freesurfer_home)

    qc_plot_surfparc_input_lh = aparc_annot.join(white_surf, by: [0, 1]).join(pial_surf, by: [0, 1]).join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4] }
    qc_plot_surfparc_input_rh = aparc_annot.join(white_surf, by: [0, 1]).join(pial_surf, by: [0, 1]).join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4] }
    qc_plot_surfparc_input = qc_plot_surfparc_input_lh.join(qc_plot_surfparc_input_rh)
    surface_parc_svg = qc_plot_surfparc(subjects_dir, qc_plot_surfparc_input, nextflow_bin_path, qc_result_path, freesurfer_home)

    anat_pctsurfcon_input = cortex_label.join(white_surf, by: [0, 1]).join(thickness_surf, by: [0, 1]).join(hemis_rawavg_mgz, by: [0, 1]).join(hemis_orig_mgz, by: [0, 1])
    (w_g_pct_mgh, w_g_pct_stats) = anat_pctsurfcon(subjects_dir, anat_pctsurfcon_input)

    // need lh and rh
    lh_anat_parcstats_input = white_surf.join(cortex_label, by: [0, 1]).join(pial_surf, by: [0, 1]).join(aparc_annot, by: [0, 1]).join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4, 5] }
    rh_anat_parcstats_input = white_surf.join(cortex_label, by: [0, 1]).join(pial_surf, by: [0, 1]).join(aparc_annot, by: [0, 1]).join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4, 5] }
    anat_parcstats_input = lh_anat_parcstats_input.join(rh_anat_parcstats_input).join(wm_mgz)
    (aparc_pial_stats, aparc_stats) = anat_parcstats(subjects_dir, anat_parcstats_input)

//     lh_anat_parcstats2_input = white_surf.join(cortex_label, by: [0, 1]).join(pial_surf, by: [0, 1]).join(aparc_a2009s_annot, by: [0, 1]).join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4, 5] }
//     rh_anat_parcstats2_input = white_surf.join(cortex_label, by: [0, 1]).join(pial_surf, by: [0, 1]).join(aparc_a2009s_annot, by: [0, 1]).join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4, 5] }
//     anat_parcstats2_input = lh_anat_parcstats2_input.join(rh_anat_parcstats2_input).join(wm_mgz)
//     (aparc_pial_stats, aparc_stats) = anat_parcstats2(subjects_dir, anat_parcstats2_input)  // if for paper, comment out

    // aparc2aseg: create aparc+aseg and aseg
    lh_anat_ribbon_mgz_input = white_surf.join(pial_surf, by: [0, 1]).join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3] }
    rh_anat_ribbon_mgz_input = white_surf.join(pial_surf, by: [0, 1]).join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3] }
    anat_ribbon_mgz_inputs = aseg_presurf_mgz.join(lh_anat_ribbon_mgz_input).join(rh_anat_ribbon_mgz_input)
    (ribbon_mgz, lh_ribbon_mgz, rh_ribbon_mgz) = anat_ribbon(subjects_dir, anat_ribbon_mgz_inputs)

    lh_anat_apas2aseg_input = white_surf.join(pial_surf, by: [0, 1]).join(cortex_label, by: [0, 1]).join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4] }
    rh_anat_apas2aseg_input = white_surf.join(pial_surf, by: [0, 1]).join(cortex_label, by: [0, 1]).join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4] }
    anat_apas2aseg_inputs = aseg_presurf_hypos_mgz.join(ribbon_mgz).join(lh_anat_apas2aseg_input).join(rh_anat_apas2aseg_input)
    (aseg_mgz, brainvol_stats) = anat_apas2aseg(subjects_dir, anat_apas2aseg_inputs)

    lh_anat_aparc2aseg_input = white_surf.join(pial_surf, by: [0, 1]).join(cortex_label, by: [0, 1]).join(aparc_annot, by: [0, 1]).join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4, 5] }
    rh_anat_aparc2aseg_input = white_surf.join(pial_surf, by: [0, 1]).join(cortex_label, by: [0, 1]).join(aparc_annot, by: [0, 1]).join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4, 5] }
    anat_aparc2aseg_inputs = aseg_mgz.join(ribbon_mgz).join(lh_anat_aparc2aseg_input).join(rh_anat_aparc2aseg_input)
    aparc_aseg_mgz = anat_aparc2aseg(subjects_dir, anat_aparc2aseg_inputs)

    qc_plot_aparc_aseg_input = norm_mgz.join(aparc_aseg_mgz)
    aparc_aseg_svg = qc_plot_aparc_aseg(subjects_dir, qc_plot_aparc_aseg_input, nextflow_bin_path, qc_result_path, freesurfer_home)
    qc_report = qc_anat_create_report(aparc_aseg_svg, nextflow_bin_path, bids_dir, subjects_dir, qc_result_path, deepprep_version)

//     lh_anat_aparc_a2009s2aseg_input = white_surf.join(pial_surf, by: [0, 1]).join(cortex_label, by: [0, 1]).join(aparc_a2009s_annot, by: [0, 1]).join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4, 5] }
//     rh_anat_aparc_a2009s2aseg_input = white_surf.join(pial_surf, by: [0, 1]).join(cortex_label, by: [0, 1]).join(aparc_a2009s_annot, by: [0, 1]).join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4, 5] }
//     anat_aparc_a2009s2aseg_inputs = aseg_mgz.join(ribbon_mgz).join(lh_anat_aparc_a2009s2aseg_input).join(rh_anat_aparc_a2009s2aseg_input)
//     aparc_a2009s_aseg_mgz = anat_aparc_a2009s2aseg(subjects_dir, anat_aparc_a2009s2aseg_inputs)  // if for paper, comment out

    //  *exvivo* labels
//     balabels_lh = Channel.fromPath("${freesurfer_fsaverage_dir}/label/*lh*exvivo*.label")
//     balabels_rh = Channel.fromPath("${freesurfer_fsaverage_dir}/label/*rh*exvivo*.label")
//     anat_balabels_input_lh = sphere_reg_surf.join(white_surf, by: [0, 1]).join(subject_id_lh, by: [0, 1])
//     anat_balabels_input_rh = sphere_reg_surf.join(white_surf, by: [0, 1]).join(subject_id_rh, by: [0, 1])
//     balabel_lh = anat_balabels_lh(subjects_dir, anat_balabels_input_lh, balabels_lh, subjects_fsaverage_dir)  // if for paper, comment out
//     balabel_rh = anat_balabels_rh(subjects_dir, anat_balabels_input_rh, balabels_rh, subjects_fsaverage_dir)  // if for paper, comment out

//     lh_bold_bbregister_input = white_surf.join(thickness_surf, by: [0, 1]).join(cortex_label, by: [0, 1]).join(subject_id_lh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4] }
//     rh_bold_bbregister_input = white_surf.join(thickness_surf, by: [0, 1]).join(cortex_label, by: [0, 1]).join(subject_id_rh, by: [0, 1]).map { tuple -> return tuple[0, 2, 3, 4] }
//     bold_bbregister_input = seg_deep_withcc_mgz.join(lh_bold_bbregister_input).join(rh_bold_bbregister_input)
//     bold_bbregister_input = aparc_aseg_mgz.join(lh_bold_bbregister_input).join(rh_bold_bbregister_input)

    emit:
    t1_mgz
    norm_mgz
    aparc_aseg_mgz
    w_g_pct_mgh  // using for control the synth morph run schedule
}


workflow bold_wf {

    take:
    t1_mgz
    norm_mgz
    aparc_aseg_mgz
    w_g_pct_mgh
    gpu_lock

    main:
    // GPU
    gpuid = params.device  // not used

    // set dir path
    bids_dir = params.bids_dir
    subjects_dir = params.subjects_dir
    bold_preprocess_path = params.bold_preprocess_path
    qc_result_path = params.qc_result_path

    // set bold processing config
    bold_task_type = params.bold_task_type
    bold_skip_frame = params.bold_skip_frame
    bold_reorient = params.bold_reorient
    bold_susceptibility_distortion_correction = params.bold_susceptibility_distortion_correction

    bold_atlas_type = params.bold_atlas_type  // not used
    bold_fs_native_space = params.bold_fs_native_space

    // set software path
    nextflow_bin_path = params.nextflow_bin_path
    freesurfer_home = params.freesurfer_home

    bold_synthmorph_model_path = params.bold_synthmorph_model_path
    bold_synthmorph_template_path = params.bold_synthmorph_template_path

    deepprep_version = params.deepprep_version

    _directory = new File(bold_preprocess_path)
        if (!_directory.exists()) {
            _directory.mkdirs()
            println "create dir: ..."
            println _directory
        }

    _directory = new File(qc_result_path)
    if (!_directory.exists()) {
        _directory.mkdirs()
        println "create dir: ..."
        println _directory
    }

    subject_boldfile_txt = bold_get_bold_file_in_bids(bids_dir, nextflow_bin_path, bold_task_type)
    (subject_id, boldfile_id, subject_boldfile_txt) = subject_boldfile_txt.flatten().multiMap { it ->
                                                                                     a: it.name.split('_')[0]
                                                                                     c: it.name
                                                                                     b: it }
    subject_id_boldfile_id = subject_id.merge(boldfile_id)
    (subject_id_unique, boldfile_id_unique) = subject_id_boldfile_id.groupTuple(sort: true).multiMap { tuple ->
                                                                                                        a: tuple[0]
                                                                                                        b: tuple[1][0] }
    if (params.bold_only.toString().toUpperCase() == 'TRUE') {
        t1_mgz = subject_id_unique.join(t1_mgz)
        norm_mgz = subject_id_unique.join(norm_mgz)
        aparc_aseg_mgz = subject_id_unique.join(aparc_aseg_mgz)
        w_g_pct_mgh = subject_id_unique.join(w_g_pct_mgh)
    }


    bold_T1_to_2mm_input = t1_mgz.join(norm_mgz)
    (t1_native2mm, norm_native2mm) = bold_T1_to_2mm(subjects_dir, bold_preprocess_path, nextflow_bin_path, bold_T1_to_2mm_input)

    // add aparc+aseg to synthmorph process to make synthmorph and bbregister processes running at the same time
    t1_native2mm_aparc_aseg = t1_native2mm.join(aparc_aseg_mgz).join(w_g_pct_mgh)
    (affine_nii, affine_trans) = synthmorph_affine(subjects_dir, bold_preprocess_path, nextflow_bin_path, t1_native2mm_aparc_aseg, bold_synthmorph_model_path, bold_synthmorph_template_path, gpu_lock)
    synthmorph_norigid_input = t1_native2mm.join(norm_native2mm, by: [0]).join(affine_trans, by: [0])
    (t1_norigid_nii, norm_norigid_nii, transvoxel) = synthmorph_norigid(subjects_dir, bold_preprocess_path, nextflow_bin_path, synthmorph_norigid_input, bold_synthmorph_model_path, bold_synthmorph_template_path, gpu_lock)

    norm_to_mni152_svg = qc_plot_norm_to_mni152(norm_norigid_nii, bold_preprocess_path, nextflow_bin_path, qc_result_path)

    subject_boldref_file = bold_get_bold_ref_in_bids(bold_preprocess_path, bids_dir, nextflow_bin_path, subject_id_unique, boldfile_id_unique)  // subject_id, subject_boldref_file
    transvoxel_group = subject_id_boldfile_id.groupTuple(sort: true).join(transvoxel).transpose()
    aparc_aseg_mgz = subject_id_boldfile_id.groupTuple(sort: true).join(aparc_aseg_mgz).transpose()
    boldfile_id_group = subject_id_boldfile_id.groupTuple(sort: true).join(subject_boldref_file).map { tuple -> tuple[1] }
    boldfile_id_split = split_subject_boldref_file(boldfile_id_group.flatten())

    bold_info = bold_skip_reorient(bold_preprocess_path, nextflow_bin_path, qc_result_path, subject_boldfile_txt, bold_reorient, bold_skip_frame, bold_susceptibility_distortion_correction)
    bold_info = boldfile_id_split.join(bold_info, by: [0, 1])
    (mc_nii, mcdat, boldref) = bold_stc_mc(bold_preprocess_path, nextflow_bin_path, bold_info)

    bbregister_dat_input = aparc_aseg_mgz.join(mc_nii, by: [0, 1])
    bbregister_dat = bold_bbregister(subjects_dir, bold_preprocess_path, nextflow_bin_path, bbregister_dat_input)
    t1_native2mm_group = subject_id_boldfile_id.groupTuple(sort: true).join(t1_native2mm).transpose()
    bold_bbregister_to_native_input = boldref.join(mc_nii, by: [0, 1]).join(t1_native2mm_group, by: [0, 1]).join(bbregister_dat, by: [0, 1])

    (bbregister_native_2mm, bbregister_native_2mm_fframe) = bold_bbregister_to_native(subjects_dir, bold_preprocess_path, nextflow_bin_path, bold_bbregister_to_native_input, freesurfer_home)

    bold_aparaseg2mc_inputs = aparc_aseg_mgz.join(mc_nii, by: [0,1]).join(bbregister_dat, by: [0,1])
    (anat_wm_nii, anat_csf_nii, anat_aseg_nii, anat_ventricles_nii, anat_brainmask_nii, anat_brainmask_bin_nii) = bold_mkbrainmask(subjects_dir, bold_preprocess_path, nextflow_bin_path, bold_aparaseg2mc_inputs)
    bold_synthmorph_norigid_apply_input = t1_native2mm_group.join(mc_nii, by: [0,1]).join(bbregister_native_2mm, by: [0,1]).join(transvoxel_group, by: [0,1])
    (synthmorph_norigid_bold, synthmorph_norigid_bold_fframe) = bold_synthmorph_norigid_apply(subjects_dir, bold_preprocess_path, nextflow_bin_path, bold_synthmorph_norigid_apply_input, bold_synthmorph_template_path, gpu_lock)

    bold_confounds_inputs = mc_nii.join(mcdat, by: [0,1]).join(anat_wm_nii, by: [0,1]).join(anat_brainmask_nii, by: [0,1]).join(anat_brainmask_bin_nii, by: [0,1]).join(anat_ventricles_nii, by: [0,1])
    (bold_confounds_txt, bold_confounds_view_txt) = bold_confounds(bold_preprocess_path, nextflow_bin_path, bold_confounds_inputs)

    qc_plot_mctsnr_input = mc_nii.join(anat_brainmask_nii, by: [0,1])
    bold_mc_tsnr_svg = qc_plot_mctsnr(qc_plot_mctsnr_input, bold_preprocess_path, nextflow_bin_path, qc_result_path)
    qc_plot_carpet_inputs = mc_nii.join(mcdat, by: [0,1]).join(anat_aseg_nii, by: [0,1]).join(anat_brainmask_nii, by: [0,1]).join(anat_brainmask_bin_nii, by: [0,1]).join(anat_wm_nii, by: [0,1]).join(anat_csf_nii, by: [0,1])
    (bold_carpet_svg) = qc_plot_carpet(bold_preprocess_path, nextflow_bin_path, qc_result_path, qc_plot_carpet_inputs)
    bold_to_mni152_svg = qc_plot_bold_to_space(synthmorph_norigid_bold_fframe, bbregister_native_2mm_fframe, bold_fs_native_space, subjects_dir, bold_preprocess_path, nextflow_bin_path, qc_result_path, freesurfer_home)
    qc_bold_create_report_input = bold_to_mni152_svg.groupTuple(by: 0)
    qc_report = qc_bold_create_report(qc_bold_create_report_input, nextflow_bin_path, bids_dir, subjects_dir, qc_result_path, bold_task_type, deepprep_version)

}


workflow {

    nextflow_bin_path = params.nextflow_bin_path
    gpu_lock = gpu_schedule_lock(nextflow_bin_path)

    if (params.anat_only.toString().toUpperCase() == 'TRUE') {
        println "INFO: anat preprocess ONLY"
        (t1_mgz, norm_mgz, aparc_aseg_mgz, w_g_pct_mgh) = anat_wf(gpu_lock)
    } else if (params.bold_only.toString().toUpperCase() == 'TRUE') {
        println "INFO: Bold preprocess ONLY"
        subjects_dir = params.subjects_dir
        t1_mgz = Channel.fromPath("${subjects_dir}/sub-*/mri/T1.mgz")
        norm_mgz = Channel.fromPath("${subjects_dir}/sub-*/mri/norm.mgz")
        (t1_mgz, norm_mgz, aparc_aseg_mgz, w_g_pct_mgh) = t1_mgz.multiMap { it ->
                                                     c: [it.getParent().getParent().getName(), it]
                                                     b: [it.getParent().getParent().getName(), file("${it.getParent()}/norm.mgz")]
                                                     a: [it.getParent().getParent().getName(), file("${it.getParent()}/aparc+aseg.mgz")]
                                                     d: [it.getParent().getParent().getName(), "hemi", file("${it.getParent()}/aseg.mgz")]
                                                     }
        bold_wf(t1_mgz, norm_mgz, aparc_aseg_mgz, w_g_pct_mgh, gpu_lock)
    } else {
        println "INFO: anat && Bold preprocess"
        (t1_mgz, norm_mgz, aparc_aseg_mgz, w_g_pct_mgh) = anat_wf(gpu_lock)
        bold_wf(t1_mgz, norm_mgz, aparc_aseg_mgz, w_g_pct_mgh, gpu_lock)
    }
}
