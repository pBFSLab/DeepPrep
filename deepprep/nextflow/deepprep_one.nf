params.subjects_dir = '/mnt/ngshare/temp/MSC_nf'
params.subject_id = 'sub-MSC01'
params.files = '/mnt/ngshare/temp/MSC/sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz'
params.fsthreads = '1'

params.fastsurfer_home = '/home/anning/workspace/DeepPrep/deepprep/FastSurfer'
params.freesurfer_home = '/usr/local/freesurfer720'

params.fastcsr_home = '/home/anning/workspace/DeepPrep/deepprep/FastCSR'
params.fastcsr_model_path = '/home/anning/workspace/DeepPrep/deepprep/model/FastCSR'

params.surfreg_home = '/home/anning/workspace/DeepPrep/deepprep/SageReg'
params.surfreg_model_path = '/home/anning/workspace/DeepPrep/deepprep/model/SageReg/model_files'

process mkdir_exist {
    input:
    val dir_path

    shell:
    """
    #!/usr/bin/env python3

    from pathlib import Path
    sd = Path('${dir_path}')
    sd.mkdir(parents=True, exist_ok=True)
    """
}

process anat_motioncor {
    input:  // https://www.nextflow.io/docs/latest/process.html#inputs
    path subjects_dir
    val subject_id
    path files


//     publishDir "${subjects_dir}", mode: 'copy'

    output:
    path("${subjects_dir}/${subject_id}/mri/orig.mgz"), emit: orig_mgz
    path("${subjects_dir}/${subject_id}/mri/rawavg.mgz"), emit: rawavg_mgz

    script:
    threads = 1

    """
    recon-all -sd ${subjects_dir} -subject ${subject_id} -i ${files} -motioncor -threads ${threads} -itkthreads ${threads}
    """
}

process anat_segment {
    input:
    path subjects_dir
    val subject_id
    path orig_mgz

    path fastsurfer_home

    output:
    path("${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.deep.mgz"), emit: seg_deep_mgz
    path("${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.orig.mgz"), emit: seg_orig_mgz

    script:
    script_py = "${fastsurfer_home}/FastSurferCNN/eval.py"

    seg_deep_mgz = "${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.deep.mgz"
    seg_orig_mgz = "${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.orig.mgz"

    network_sagittal_path = "${fastsurfer_home}/checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    network_coronal_path = "${fastsurfer_home}/checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    network_axial_path = "${fastsurfer_home}/checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl"
    """
    python3 ${script_py} \
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
    input:
    path subjects_dir
    val subject_id
    path seg_deep_mgz

    path fastsurfer_home

    output:
    path("${subjects_dir}/${subject_id}/mri/aseg.auto_noCCseg.mgz"), emit: aseg_auto_noccseg_mgz
    path("${subjects_dir}/${subject_id}/mri/mask.mgz"), emit: mask_mgz

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
    input:
    path subjects_dir
    val subject_id
    path orig_mgz
    path mask_mgz

    path fastsurfer_home

    output:
    path("${subjects_dir}/${subject_id}/mri/orig_nu.mgz"), emit: orig_nu_mgz

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
    input:
    path subjects_dir
    val subject_id
    path orig_mgz
    path orig_nu_mgz

    path freesurfer_home

    output:
    path("${subjects_dir}/${subject_id}/mri/nu.mgz"), emit: nu_mgz
    path("${subjects_dir}/${subject_id}/mri/transforms/talairach.auto.xfm"), emit: talairach_auto_xfm
    path("${subjects_dir}/${subject_id}/mri/transforms/talairach.xfm"), emit: talairach_xfm
    path("${subjects_dir}/${subject_id}/mri/transforms/talairach.xfm.lta"), emit: talairach_xfm_lta
    path("${subjects_dir}/${subject_id}/mri/transforms/talairach_with_skull.lta"), emit: talairach_with_skull_lta
    path("${subjects_dir}/${subject_id}/mri/transforms/talairach.lta"), emit: talairach_lta

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
    input:
    path subjects_dir
    val subject_id
    path nu_mgz

    output:
    path("${subjects_dir}/${subject_id}/mri/T1.mgz"), emit: t1_mgz

    script:
    threads = 1

    t1_mgz = "${subjects_dir}/${subject_id}/mri/T1.mgz"

    """
    mri_normalize -seed 1234 -g 1 -mprage ${nu_mgz} ${t1_mgz}
    """
}


process anat_brainmask {
    input:
    path subjects_dir
    val subject_id
    path nu_mgz
    path mask_mgz

    output:
    path("${subjects_dir}/${subject_id}/mri/norm.mgz"), emit: norm_mgz
    path("${subjects_dir}/${subject_id}/mri/brainmask.mgz"), emit: brainmask_mgz

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
    input:
    path subjects_dir
    val subject_id
    path norm_mgz
    path seg_deep_mgz
    path aseg_auto_noccseg_mgz

    path fastsurfer_home

    output:
    path("${subjects_dir}/${subject_id}/mri/aseg.auto.mgz"), emit: aseg_auto_mgz
    path("${subjects_dir}/${subject_id}/mri/aparc.DKTatlas+aseg.deep.withCC.mgz"), emit: seg_deep_withcc_mgz

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
    input:
    path subjects_dir
    val subject_id
    path aseg_auto_mgz
    path norm_mgz
    path brainmask_mgz
    path talairach_lta

    val fsthreads

    output:
    path("${subjects_dir}/${subject_id}/mri/aseg.presurf.mgz"), emit: aseg_presurf_mgz
    path("${subjects_dir}/${subject_id}/mri/brain.mgz"), emit: brain_mgz
    path("${subjects_dir}/${subject_id}/mri/brain.finalsurfs.mgz"), emit: brain_finalsurfs_mgz
    path("${subjects_dir}/${subject_id}/mri/wm.mgz"), emit: wm_mgz
    path("${subjects_dir}/${subject_id}/mri/filled.mgz"), emit: filled_mgz

    script:
    threads = 1
    """
    recon-all -sd ${subjects_dir} -subject ${subject_id} \
    -asegmerge -normalization2 -maskbfs -segmentation -fill \
    -threads ${threads} -itkthreads ${threads}
    """
}


process anat_fastcsr_levelset {
    input:
    path subjects_dir
    val subject_id
    path orig_mgz
    path filled_mgz

    val fastcsr_home
    path fastcsr_model_path

    each hemi

    output:
    path("${subjects_dir}/${subject_id}/mri/${hemi}_levelset.nii.gz"), emit: levelset_nii

    script:

    script_py = "${fastcsr_home}/fastcsr_model_infer.py"
    threads = 1

    """
    python3 ${script_py} \
    --fastcsr_subjects_dir ${subjects_dir} \
    --subj ${subject_id} \
    --hemi ${hemi} \
    --model-path ${fastcsr_model_path}
    """
}


process anat_fastcsr_mksurface {
    input:
    path subjects_dir
    val subject_id
    path orig_mgz
    path brainmask_mgz
    path aseg_presurf_mgz
    path levelset_nii

    val fastcsr_home

    output:
    path("${subjects_dir}/${subject_id}/surf/${hemi}.orig"), emit: orig_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.orig.premesh"), emit: orig_premesh_surf

    script:
    String path_name = levelset_nii.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in levelset path name"

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
    input:
    path subjects_dir
    val subject_id
    path brain_finalsurfs_mgz
    path wm_mgz
    path orig_premesh_surf

    output:
    path("${subjects_dir}/${subject_id}/surf/autodet.gw.stats.${hemi}.dat"), emit: autodet_gwstats

    script:
    String path_name = orig_premesh_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in orig_premesh_surf path name"
    //
    """
    mris_autodet_gwstats --o ${subjects_dir}/${subject_id}/surf/autodet.gw.stats.${hemi}.dat \
     --i ${brain_finalsurfs_mgz} --wm ${wm_mgz} --surf ${orig_premesh_surf}
    """
}


process anat_white_surface {
    input:
    path subjects_dir
    val subject_id
    path aseg_presurf_mgz
    path brain_finalsurfs_mgz
    path wm_mgz
    path filled_mgz
    path orig_surf
    path autodet_gwstats

    output:
    path("${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc"), emit: white_preaparc_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.white"), emit: white_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.curv"), emit: curv_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.area"), emit: area_surf

//     errorStrategy 'ignore'

    script:
    String path_name = orig_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in orig_surf path name"
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
    input:
    path subjects_dir
    val subject_id
    path aseg_presurf_mgz
    path white_preaparc_surf

    output:
    path("${subjects_dir}/${subject_id}/label/${hemi}.cortex.label"), emit: cortex_label

    script:
    String path_name = white_preaparc_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in white_preaparc_surf path name"
    threads = 1

    """
    mri_label2label --label-cortex ${white_preaparc_surf} ${aseg_presurf_mgz} 0 ${subjects_dir}/${subject_id}/label/${hemi}.cortex.label
    """
}


process anat_cortex_hipamyg_label {
    input:
    path subjects_dir
    val subject_id
    path aseg_presurf_mgz
    path white_preaparc_surf

    output:
    path("${subjects_dir}/${subject_id}/label/${hemi}.cortex+hipamyg.label"), emit: cortex_hipamyg_label

    script:
    String path_name = white_preaparc_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in white_preaparc_surf path name"
    threads = 1

    """
    mri_label2label --label-cortex ${white_preaparc_surf} ${aseg_presurf_mgz} 1 ${subjects_dir}/${subject_id}/label/${hemi}.cortex+hipamyg.label
    """
}


process anat_pctsurfcon {
    input:
    path subjects_dir
    val subject_id
    path rawavg_mgz
    path orig_mgz
    tuple path(cortex_label), path(white_surf), path(thickness_surf)

    output:
    path("${subjects_dir}/${subject_id}/surf/${hemi}.w-g.pct.mgh"), emit: w_g_pct_mgh
    path("${subjects_dir}/${subject_id}/stats/${hemi}.w-g.pct.stats"), emit: w_g_pct_stats

    script:
    String path_name = cortex_label.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in orig_surf path name"
    threads = 1
//     """
//     recon-all -sd ${subjects_dir} -subject ${subject_id} -pctsurfcon -threads ${threads} -itkthreads ${threads}
//     """
    """
    SUBJECTS_DIR=${subjects_dir} pctsurfcon --s ${subject_id} --${hemi}-only
    """
}


process anat_hyporelabel {
    input:
    path subjects_dir
    val subject_id
    path aseg_presurf_mgz
    tuple path(lh_white_surf), path(rh_white_surf)

    output:
    path("${subjects_dir}/${subject_id}/mri/aseg.presurf.hypos.mgz"), emit: aseg_presurf_hypos_mgz

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
    input:
    path subjects_dir
    val subject_id
    path white_preaparc_surf

    output:
    path("${subjects_dir}/${subject_id}/surf/${hemi}.smoothwm"), emit: smoothwm_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.inflated"), emit: inflated_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.sulc"), emit: sulc_surf

    script:
    String path_name = white_preaparc_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in white_preaparc_surf path name"
    threads = 1

    """
    mris_smooth -n 3 -nw  ${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc ${subjects_dir}/${subject_id}/surf/${hemi}.smoothwm
    mris_inflate ${subjects_dir}/${subject_id}/surf/${hemi}.smoothwm ${subjects_dir}/${subject_id}/surf/${hemi}.inflated
    """
}


process anat_curv_stats {
    input:
    path subjects_dir
    val subject_id
    tuple path(smoothwm_surf), path(curv_surf), path(sulc_surf)

    output:
    path("${subjects_dir}/${subject_id}/stats/${hemi}.curv.stats"), emit: curv_stats

    script:
    String path_name = smoothwm_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in smoothwm_surf path name"
    threads = 1
    """
    SUBJECTS_DIR=${subjects_dir} mris_curvature_stats -m --writeCurvatureFiles -G -o "${subjects_dir}/${subject_id}/stats/${hemi}.curv.stats" -F smoothwm ${subject_id} ${hemi} curv sulc
    """
//     """
//     recon-all -sd ${subjects_dir} -subject ${subject_id} -curvstats
//     """
}


process anat_sphere {
    input:
    path subjects_dir
    val subject_id
    path inflated_surf

    output:
    path("${subjects_dir}/${subject_id}/surf/${hemi}.sphere"), emit: sphere_surf

    script:
    String path_name = inflated_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in inflated_surf path name"
    threads = 1

    """
    mris_sphere -seed 1234 ${subjects_dir}/${subject_id}/surf/${hemi}.inflated ${subjects_dir}/${subject_id}/surf/${hemi}.sphere
    """
}


process anat_sphere_register {
    input:
    path subjects_dir
    val subject_id
    tuple path(curv_surf), path(sulc_surf), path(sphere_surf)

    path surfreg_home
    path surfreg_model_path
    path freesurfer_home

    output:
    path("${subjects_dir}/${subject_id}/surf/${hemi}.sphere.reg"), emit: sphere_reg_surf

    script:
    String path_name = sphere_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in sphere_surf path name"

    script_py = "${surfreg_home}/predict.py"
    threads = 1
    device = "cuda"

    """
    python3 ${script_py} --sd ${subjects_dir} --sid ${subject_id} --fsd ${freesurfer_home} \
    --hemi ${hemi} --model_path ${surfreg_model_path} --device ${device}
    """
}


process anat_jacobian {
    input:
    path subjects_dir
    val subject_id
    tuple path(white_preaparc_surf), path(sphere_reg_surf)

    output:
    path("${subjects_dir}/${subject_id}/surf/${hemi}.jacobian_white"), emit: jacobian_white

    script:
    String path_name = sphere_reg_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in sphere_reg_surf path name"

    device = "cuda"
    """
    mris_jacobian ${subjects_dir}/${subject_id}/surf/${hemi}.white.preaparc \
    ${subjects_dir}/${subject_id}/surf/${hemi}.sphere.reg \
    ${subjects_dir}/${subject_id}/surf/${hemi}.jacobian_white
    """
}


process anat_avgcurv {
    input:
    path subjects_dir
    val subject_id
    path sphere_reg_surf

    path freesurfer_home

    output:
    path("${subjects_dir}/${subject_id}/surf/${hemi}.avg_curv"), emit: avg_curv

    script:
    String path_name = sphere_reg_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in sphere_reg_surf path name"

    device = "cuda"
    """
    mrisp_paint -a 5 \
    ${freesurfer_home}/average/${hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 \
    ${subjects_dir}/${subject_id}/surf/${hemi}.sphere.reg ${subjects_dir}/${subject_id}/surf/${hemi}.avg_curv
    """
}


process anat_cortparc_aparc {
    input:
    path subjects_dir
    val subject_id
    path aseg_presurf_mgz
    tuple path(cortex_label), path(sphere_reg_surf), path(smoothwm_surf)
    // smoothwm_surf is hidden needed

    path freesurfer_home

    output:
    path("${subjects_dir}/${subject_id}/label/${hemi}.aparc.annot"), emit: aparc_annot

    script:
    String path_name = sphere_reg_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in sphere_reg_surf path name"

    device = "cuda"
    """
    mris_ca_label -SDIR ${subjects_dir} -l ${cortex_label} -aseg ${aseg_presurf_mgz} -seed 1234 ${subject_id} ${hemi} ${sphere_reg_surf} \
    ${freesurfer_home}/average/${hemi}.curvature.buckner40.filled.desikan_killiany.2010-03-25.gcs \
    ${subjects_dir}/${subject_id}/label/${hemi}.aparc.annot
    """
}


process anat_cortparc_aparc_a2009s {
    input:
    path subjects_dir
    val subject_id
    path aseg_presurf_mgz
    tuple path(cortex_label), path(sphere_reg_surf), path(smoothwm_surf)
    // smoothwm_surf is hidden needed

    path freesurfer_home

    output:
    path("${subjects_dir}/${subject_id}/label/${hemi}.aparc.a2009s.annot"), emit: aparc_a2009s_annot

    script:
    String path_name = sphere_reg_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in sphere_reg_surf path name"

    device = "cuda"
    """
    mris_ca_label -SDIR ${subjects_dir} -l ${cortex_label} -aseg ${aseg_presurf_mgz} -seed 1234 ${subject_id} ${hemi} ${sphere_reg_surf} \
    ${freesurfer_home}/average/${hemi}.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs \
    ${subjects_dir}/${subject_id}/label/${hemi}.aparc.a2009s.annot
    """
}


process anat_pial_surface {
    input:
    path subjects_dir
    val subject_id
    path aseg_presurf_mgz
    path wm_mgz
    path brain_finalsurfs_mgz
    tuple path(orig_surf), path(white_surf), path(autodet_gwstats), path(cortex_hipamyg_label), path(cortex_label), path(aparc_annot)

    output:
    path("${subjects_dir}/${subject_id}/surf/${hemi}.pial"), emit: pial_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.pial.T1"), emit: pial_t1_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.curv.pial"), emit: curv_pial_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.area.pial"), emit: area_pial_surf
    path("${subjects_dir}/${subject_id}/surf/${hemi}.thickness"), emit: thickness_surf

    script:
    String path_name = white_surf.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in orig_surf path name"
    threads = 1

    """
    mris_place_surface --adgws-in ${autodet_gwstats} --seg ${aseg_presurf_mgz} --threads ${threads} --wm ${wm_mgz} \
    --invol ${brain_finalsurfs_mgz} --${hemi} --i ${white_surf} --o ${subjects_dir}/${subject_id}/surf/${hemi}.pial.T1 \
    --pial --nsmooth 0 --rip-label ${subjects_dir}/${subject_id}/label/lh.cortex+hipamyg.label --pin-medial-wall ${cortex_label} --aparc ${aparc_annot} --repulse-surf ${white_surf} --white-surf ${white_surf}
    cp ${subjects_dir}/${subject_id}/surf/${hemi}.pial.T1 ${subjects_dir}/${subject_id}/surf/${hemi}.pial

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


process anat_ribbon {
    input:
    path subjects_dir
    val subject_id
    path aseg_presurf_mgz
    tuple path(lh_white_surf), path(lh_white_surf)
    tuple path(lh_pial_surf), path(rh_pial_surf)

    output:
    path("${subjects_dir}/${subject_id}/mri/ribbon.mgz"), emit: ribbon_mgz
    path("${subjects_dir}/${subject_id}/mri/lh.ribbon.mgz"), emit: lh_ribbon_mgz
    path("${subjects_dir}/${subject_id}/mri/rh.ribbon.mgz"), emit: rh_ribbon_mgz

    script:
    threads = 1
    """
    mris_volmask --sd ${subjects_dir} --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon ${subject_id}
    """
}


process anat_parcstats {
    input:
    path subjects_dir
    val subject_id
    path wm_mgz
    tuple path(white_surf), path(cortex_label), path(pial_surf), path(aparc_annot)

    output:
    path("${subjects_dir}/${subject_id}/stats/${hemi}.aparc.pial.stats"), emit: aparc_pial_stats
    path("${subjects_dir}/${subject_id}/stats/${hemi}.aparc.stats"), emit: aparc_stats

    script:
    String path_name = aparc_annot.name
    if ( path_name.matches("lh(.*)") )
        hemi = "lh"
    else if ( path_name.matches("rh(.*)") )
        hemi = "rh"
    else
        error "Can not find hemi in aparc_annot path name"
    threads = 1
//     """
//     recon-all -sd ${subjects_dir} -subject ${subject_id} -parcstats -threads ${threads} -itkthreads ${threads}
//     """
    """
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${cortex_label} -f ${subjects_dir}/${subject_id}/stats/${hemi}.aparc.pial.stats -b -a ${aparc_annot} -c ${subjects_dir}/${subject_id}/stats/aparc.annot.ctab ${subject_id} ${hemi} pial
    SUBJECTS_DIR=${subjects_dir} mris_anatomical_stats -th3 -mgz -cortex ${cortex_label} -f ${subjects_dir}/${subject_id}/stats/${hemi}.aparc.stats -b -a ${aparc_annot} -c ${subjects_dir}/${subject_id}/stats/aparc.annot.ctab ${subject_id} ${hemi} white
    """
}


process anat_apas2aseg {
    input:
    path subjects_dir
    val subject_id
    path aseg_presurf_hypos_mgz
    path ribbon_mgz
    tuple path(lh_white_surf), path(rh_white_surf)
    tuple path(lh_pial_surf), path(rh_pial_surf)
    tuple path(lh_cortex_label), path(rh_cortex_label)

    output:
    path("${subjects_dir}/${subject_id}/mri/aseg.mgz"), emit: parcstats
    path("${subjects_dir}/${subject_id}/stats/brainvol.stats"), emit: brainvol_stats

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
    input:
    path subjects_dir
    val subject_id
    path aseg_mgz
    path ribbon_mgz
    tuple path(lh_white_surf), path(rh_white_surf)
    tuple path(lh_pial_surf), path(rh_pial_surf)
    tuple path(lh_cortex_label), path(rh_cortex_label)
    tuple path(lh_aparc_annot), path(rh_aparc_annot)

    output:
    path("${subjects_dir}/${subject_id}/mri/aparc+aseg.mgz"), emit: parcstats

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


process anat_aparc_a2009s2aseg {
    input:
    path subjects_dir
    val subject_id
    path aseg_mgz
    path ribbon_mgz
    tuple path(lh_white_surf), path(rh_white_surf)
    tuple path(lh_pial_surf), path(rh_pial_surf)
    tuple path(lh_cortex_label), path(rh_cortex_label)
    tuple path(lh_aparc_annot), path(rh_aparc_annot)

    output:
    path("${subjects_dir}/${subject_id}/mri/aparc.a2009s+aseg.mgz"), emit: parcstats

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
    input:
    path subjects_subject_dir
    val subject_id
    path freesurfer_fsaverage
    tuple(path(sphere_reg_surf), path(white_surf))
    each path(label)

    output:
    path("${subjects_subject_dir}/label/${label}"), emit: balabel

    script:
    """
    SUBJECTS_DIR=\${PWD} mri_label2label --srcsubject fsaverage --srclabel ${freesurfer_fsaverage}/label/${label} --trgsubject ${subject_id} --trglabel ${subjects_subject_dir}/label/${label} --hemi lh --regmethod surface
    """
}


process anat_balabels_rh {
    input:
    path subjects_subject_dir
    val subject_id
    path freesurfer_fsaverage
    tuple(path(sphere_reg_surf), path(white_surf))
    each path(label)

    output:
    path("${subjects_subject_dir}/label/${label}"), emit: balabel

    script:
    """
    SUBJECTS_DIR=\${PWD} mri_label2label --srcsubject fsaverage --srclabel ${freesurfer_fsaverage}/label/${label} --trgsubject ${subject_id} --trglabel ${subjects_subject_dir}/label/${label} --hemi rh --regmethod surface
    """
}


process anat_test {
    input:
    path subjects_dir
    val subject_id

//     output:
//     path("${subjects_dir}/${subject_id}/mri/aseg.presurf.hypos.mgz"), emit: parcstats

    script:
    threads = 1
    """
    recon-all -sd ${subjects_dir} -subject ${subject_id} -parcstats -threads ${threads} -itkthreads ${threads}
    """
}


workflow {
    subjects_dir = params.subjects_dir
    subject_id = params.subject_id
    files = params.files

    fsthreads = params.fsthreads

    fastsurfer_home = params.fastsurfer_home

    freesurfer_home = params.freesurfer_home

    fastcsr_home = params.fastcsr_home
    fastcsr_model_path = params.fastcsr_model_path

    surfreg_home = params.surfreg_home
    surfreg_model_path = params.surfreg_model_path

//     rawavg_mgz = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/mri/rawavg.mgz'
//     orig_mgz = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/mri/orig.mgz'
//     filled_mgz = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/mri/filled.mgz'
//     brainmask_mgz = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/mri/brainmask.mgz'
//     aseg_presurf_mgz = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/mri/aseg.presurf.mgz'
//     brain_finalsurfs_mgz = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/mri/brain.finalsurfs.mgz'
//     wm_mgz = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/mri/wm.mgz'
//     filled_mgz = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/mri/filled.mgz'
//
//     orig_surf = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/lh.orig'
//     orig_premesh_surf = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/lh.orig.premesh'
//     white_preaparc_surf = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/lh.white.preaparc'
//     white_surf = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/lh.white'
//     smoothwm_surf = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/lh.smoothwm'
//     curv_surf = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/lh.curv'
//     sulc_surf = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/lh.sulc'
//     sphere_surf = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/lh.sphere'
//     sphere_reg_surf = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/lh.sphere.reg'
//     autodet_gwstats = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/surf/autodet.gw.stats.lh.dat'
//
//     cortex_label = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/label/lh.cortex.label'
//     cortex_hipamyg_label = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/label/lh.cortex+hipamyg.label'
//     aparc_annot = '/mnt/ngshare/temp/MSC_nf/sub-MSC01/label/lh.aparc.annot'

    // freesurfer
    (orig_mgz, rawavg_mgz) = anat_motioncor(subjects_dir, subject_id, files)

    // fastsurfer
    (seg_deep_mgz, seg_orig_mgz) = anat_segment(subjects_dir, subject_id, orig_mgz, fastsurfer_home)
    (aseg_auto_noccseg_mgz, mask_mgz) = anat_reduce_to_aseg(subjects_dir, subject_id, seg_deep_mgz, fastsurfer_home)
    orig_nu_mgz = anat_N4_bias_correct(subjects_dir, subject_id, orig_mgz, mask_mgz, fastsurfer_home)
    (nu_mgz, talairach_auto_xfm, talairach_xfm, talairach_xfm_lta, talairach_with_skull_lta, talairach_lta) = anat_talairach_and_nu(subjects_dir, subject_id, orig_mgz, orig_nu_mgz, freesurfer_home)
    t1_mgz = anat_T1(subjects_dir, subject_id, nu_mgz)
    (norm_mgz, brainmask_mgz) = anat_brainmask(subjects_dir, subject_id, nu_mgz, mask_mgz)
    (aseg_auto_mgz, cc_up_lta, seg_deep_withcc_mgz) = anat_paint_cc_to_aseg(subjects_dir, subject_id, norm_mgz, seg_deep_mgz, aseg_auto_noccseg_mgz, fastsurfer_home)

    // freesurfer
    (aseg_presurf_mgz, brain_mgz, brain_finalsurfs_mgz, wm_mgz, filled_mgz) = anat_fill(subjects_dir, subject_id, aseg_auto_mgz, norm_mgz, brainmask_mgz, talairach_lta, fsthreads)

    // ####################### Split two channel #######################

    // fastcsr
    hemis = channel.of('lh', 'rh')
    levelset_nii = anat_fastcsr_levelset(subjects_dir, subject_id, orig_mgz, filled_mgz, fastcsr_home, fastcsr_model_path, hemis)
    (orig_surf, orig_premesh_surf) = anat_fastcsr_mksurface(subjects_dir, subject_id, orig_mgz, brainmask_mgz, aseg_presurf_mgz, levelset_nii, fastcsr_home)

    autodet_gwstats = anat_autodet_gwstats(subjects_dir, subject_id, brain_finalsurfs_mgz, wm_mgz, orig_premesh_surf)
    (white_preaparc_surf, white_surf, curv_surf, area_surf) = anat_white_surface(subjects_dir, subject_id, aseg_presurf_mgz, brain_finalsurfs_mgz, wm_mgz, filled_mgz, orig_surf, autodet_gwstats)
    cortex_label = anat_cortex_label(subjects_dir, subject_id, aseg_presurf_mgz, white_preaparc_surf)

    cortex_hipamyg_label = anat_cortex_hipamyg_label(subjects_dir, subject_id, aseg_presurf_mgz, white_preaparc_surf)

    (smoothwm_surf, inflated_surf, sulc_surf) = anat_smooth_inflated(subjects_dir, subject_id, white_preaparc_surf)

    // 如果使用filter，那么需要等相关的channel run success，才能开始。
    anat_curv_stats_input_lh = smoothwm_surf.filter(~/.*lh\..*/).concat(curv_surf.filter(~/.*lh\..*/), sulc_surf.filter(~/.*lh\..*/)).collect()
    anat_curv_stats_input_rh = smoothwm_surf.filter(~/.*rh\..*/).concat(curv_surf.filter(~/.*rh\..*/), sulc_surf.filter(~/.*rh\..*/)).collect()
    anat_curv_stats_input = anat_curv_stats_input_lh.concat(anat_curv_stats_input_rh)
    curv_stats = anat_curv_stats(subjects_dir, subject_id, anat_curv_stats_input)

    sphere_surf = anat_sphere(subjects_dir, subject_id, inflated_surf)

    anat_sphere_register_input_lh = curv_surf.filter(~/.*lh\..*/).concat(sulc_surf.filter(~/.*lh\..*/), sphere_surf.filter(~/.*lh\..*/)).collect()
    anat_sphere_register_input_rh = curv_surf.filter(~/.*rh\..*/).concat(sulc_surf.filter(~/.*rh\..*/), sphere_surf.filter(~/.*rh\..*/)).collect()
    anat_sphere_register_input = anat_sphere_register_input_lh.concat(anat_sphere_register_input_rh)
    sphere_reg_surf = anat_sphere_register(subjects_dir, subject_id, anat_sphere_register_input, surfreg_home, surfreg_model_path, freesurfer_home)

    anat_jacobian_input_lh = white_preaparc_surf.filter(~/.*lh\..*/).concat(sphere_reg_surf.filter(~/.*lh\..*/)).collect()
    anat_jacobian_input_rh = white_preaparc_surf.filter(~/.*rh\..*/).concat(sphere_reg_surf.filter(~/.*rh\..*/)).collect()
    anat_jacobian_input = anat_jacobian_input_lh.concat(anat_jacobian_input_rh)
    jacobian_white = anat_jacobian(subjects_dir, subject_id, anat_jacobian_input)

    avg_curv = anat_avgcurv(subjects_dir, subject_id, sphere_reg_surf, freesurfer_home)


    anat_cortparc_aparc_input_lh = cortex_label.filter(~/.*lh\..*/).concat(sphere_reg_surf.filter(~/.*lh\..*/), smoothwm_surf.filter(~/.*lh\..*/)).collect()
    anat_cortparc_aparc_input_rh = cortex_label.filter(~/.*rh\..*/).concat(sphere_reg_surf.filter(~/.*rh\..*/), smoothwm_surf.filter(~/.*rh\..*/)).collect()
    anat_cortparc_aparc_input = anat_cortparc_aparc_input_lh.concat(anat_cortparc_aparc_input_rh)
    aparc_annot = anat_cortparc_aparc(subjects_dir, subject_id, aseg_presurf_mgz, anat_cortparc_aparc_input, freesurfer_home)

    anat_pial_surface_input_lh = orig_surf.filter(~/.*lh\..*/).concat(white_surf.filter(~/.*lh\..*/), autodet_gwstats.filter(~/.*lh\..*/), cortex_hipamyg_label.filter(~/.*lh\..*/), cortex_label.filter(~/.*lh\..*/), aparc_annot.filter(~/.*lh\..*/)).collect()
    anat_pial_surface_input_rh = orig_surf.filter(~/.*rh\..*/).concat(white_surf.filter(~/.*rh\..*/), autodet_gwstats.filter(~/.*rh\..*/), cortex_hipamyg_label.filter(~/.*rh\..*/), cortex_label.filter(~/.*rh\..*/), aparc_annot.filter(~/.*rh\..*/)).collect()
    anat_pial_surface_input = anat_pial_surface_input_lh.concat(anat_pial_surface_input_rh)
    (pial_surf, pial_t1_surf, curv_pial_surf, area_pial_surf, thickness_surf) = anat_pial_surface(subjects_dir, subject_id, aseg_presurf_mgz, wm_mgz, brain_finalsurfs_mgz, anat_pial_surface_input)

    anat_pctsurfcon_input_lh = cortex_label.filter(~/.*lh\..*/).concat(white_surf.filter(~/.*lh\..*/), thickness_surf.filter(~/.*lh\..*/)).collect()
    anat_pctsurfcon_input_rh = cortex_label.filter(~/.*rh\..*/).concat(white_surf.filter(~/.*rh\..*/), thickness_surf.filter(~/.*rh\..*/)).collect()
    anat_pctsurfcon_input = anat_pctsurfcon_input_lh.concat(anat_pctsurfcon_input_rh)
    (w_g_pct_mgh, w_g_pct_stats) = anat_pctsurfcon(subjects_dir, subject_id, rawavg_mgz, orig_mgz, anat_pctsurfcon_input)

    white_surfs = white_surf.collect()
    pial_surfs = pial_surf.collect()
    aseg_presurf_hypos_mgz = anat_hyporelabel(subjects_dir, subject_id, aseg_presurf_mgz, white_surfs)
    (ribbon_mgz, lh_ribbon_mgz, rh_ribbon_mgz) = anat_ribbon(subjects_dir, subject_id, aseg_presurf_mgz, white_surfs, pial_surfs)

    cortex_labels = cortex_label.collect()
    (aseg_mgz, brainvol_stats) = anat_apas2aseg(subjects_dir, subject_id, aseg_presurf_hypos_mgz, ribbon_mgz, white_surfs, pial_surfs, cortex_labels)

    aparc_annots = aparc_annot.collect()
    aparc_aseg = anat_aparc2aseg(subjects_dir, subject_id, aseg_mgz, ribbon_mgz, white_surfs, pial_surfs, cortex_labels, aparc_annots)

    anat_parcstats_input_lh = white_surf.filter(~/.*lh\..*/).concat(cortex_label.filter(~/.*lh\..*/), pial_surf.filter(~/.*lh\..*/), aparc_annot.filter(~/.*lh\..*/)).collect()
    anat_parcstats_input_rh = white_surf.filter(~/.*rh\..*/).concat(cortex_label.filter(~/.*rh\..*/), pial_surf.filter(~/.*rh\..*/), aparc_annot.filter(~/.*lh\..*/)).collect()
    anat_parcstats_input = anat_parcstats_input_lh.concat(anat_parcstats_input_rh)
    (aparc_pial_stats, aparc_stats) = anat_parcstats(subjects_dir, subject_id, wm_mgz, anat_parcstats_input)

    freesurfer_fsaverage_dir = "/home/anning/workspace/DeepPrep/deepprep/nextflow/fsaverage"
    subjects_subject_dir = Channel.of("${subjects_dir}/${subject_id}")
    balabels_lh = Channel.fromPath("${freesurfer_fsaverage_dir}/label/*lh*exvivo*.label")
    balabels_rh = Channel.fromPath("${freesurfer_fsaverage_dir}/label/*rh*exvivo*.label")
    anat_balabels_input_lh = sphere_reg_surf.filter(~/.*lh\..*/).concat(white_surf.filter(~/.*lh\..*/)).collect()
    anat_balabels_input_rh = sphere_reg_surf.filter(~/.*rh\..*/).concat(white_surf.filter(~/.*rh\..*/)).collect()
    balabel_lh = anat_balabels_lh(subjects_subject_dir, subject_id, freesurfer_fsaverage_dir, anat_balabels_input_lh, balabels_lh)
    balabel_rh = anat_balabels_rh(subjects_subject_dir, subject_id, freesurfer_fsaverage_dir, anat_balabels_input_rh, balabels_rh)
}
