import os
import argparse
from pathlib import Path
import json
import time
import bids
from app.utils.utils import timing_func


def set_envrion(threads: int = 1):
    # FreeSurfer recon-all env
    freesurfer_home = '/usr/local/freesurfer720'
    os.environ['FREESURFER_HOME'] = f'{freesurfer_home}'
    os.environ['FREESURFER'] = f'{freesurfer_home}'
    os.environ['SUBJECTS_DIR'] = f'{freesurfer_home}/subjects'
    os.environ['PATH'] = f'{freesurfer_home}/bin:{freesurfer_home}/mni/bin:{freesurfer_home}/tktools:' + \
                         f'{freesurfer_home}/fsfast/bin:' + os.environ['PATH']
    os.environ['MINC_BIN_DIR'] = f'{freesurfer_home}/mni/bin'
    os.environ['MINC_LIB_DIR'] = f'{freesurfer_home}/mni/lib'
    os.environ['PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
    os.environ['MNI_PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
    # FreeSurfer fsfast env
    os.environ['FSF_OUTPUT_FORMAT'] = 'nii.gz'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

    # FSL
    os.environ['PATH'] = '/usr/local/fsl/bin:' + os.environ['PATH']

    # set threads
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads)


def run_with_timing(cmd):
    print('*' * 50)
    print(cmd)
    print('*' * 50)
    start = time.time()
    os.system(cmd)
    print('=' * 50)
    print(cmd)
    print('=' * 50, 'runtime:', ' ' * 3, time.time() - start)


@timing_func
def fastsurfer_seg(t1_input: str, fs_home: Path, sub_mri_dir: Path):
    orig_o = sub_mri_dir / 'orig.mgz'
    aseg_o = sub_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'

    fastsurfer_eval = fs_home / 'FastSurferCNN' / 'eval.py'
    weight_dir = fs_home / 'checkpoints'

    cmd = f'{python} {fastsurfer_eval} ' \
          f'--in_name {t1_input} ' \
          f'--out_name {aseg_o} ' \
          f'--conformed_name {orig_o} ' \
          '--order 1 ' \
          f'--network_sagittal_path {weight_dir}/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl ' \
          f'--network_axial_path {weight_dir}/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl ' \
          f'--network_coronal_path {weight_dir}/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl ' \
          '--batch_size 8 --simple_run --run_viewagg_on check'
    run_with_timing(cmd)


@timing_func
def creat_orig_and_rawavg(sub_mri_dir: Path):
    # create orig.mgz and aparc.DKTatlas+aseg.orig.mgz (copy of segmentation)
    t1 = sub_mri_dir / 'orig.mgz'
    cmd = f'mri_convert {t1} {t1}'
    run_with_timing(cmd)

    seg_deep = sub_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'
    seg_orig = sub_mri_dir / 'aparc.DKTatlas+aseg.orig.mgz'
    cmd = f'mri_convert {seg_deep} {seg_orig}'
    run_with_timing(cmd)

    # link to rawavg (needed by pctsurfcon)
    rawavg = sub_mri_dir / 'rawavg.mgz'
    if not os.path.exists(rawavg):
        cmd = f'ln -sf {t1} {rawavg}'
        run_with_timing(cmd)


@timing_func
def creat_aseg_noccseg(fs_bin: Path, sub_mri_dir: Path):
    # reduce labels to aseg, then create mask (dilate 5, erode 4, largest component), also mask aseg to remove outliers
    # output will be uchar (else mri_cc will fail below)
    py = fs_bin / 'reduce_to_aseg.py'
    mask = sub_mri_dir / 'mask.mgz'
    cmd = f'{python} {py} ' \
          f'-i {sub_mri_dir}/aparc.DKTatlas+aseg.orig.mgz ' \
          f'-o {sub_mri_dir}/aseg.auto_noCCseg.mgz --outmask {mask} --fixwm'
    run_with_timing(cmd)


@timing_func
def creat_talairach_and_nu(fs_bin: Path, sub_mri_dir: Path, threads: int):
    # orig_nu nu correct
    py = fs_bin / 'N4_bias_correct.py'
    cmd = f"{python} {py} --in {sub_mri_dir}/orig.mgz --out {sub_mri_dir}/orig_nu.mgz " \
          f"--mask {sub_mri_dir}/mask.mgz  --threads {threads}"
    run_with_timing(cmd)

    # talairach.xfm: compute talairach full head (25sec)
    cmd = f'cd {sub_mri_dir} && ' \
          f'talairach_avi --i {sub_mri_dir}/orig_nu.mgz --xfm {sub_mri_dir}/transforms/talairach.auto.xfm'
    run_with_timing(cmd)
    cmd = f'cp {sub_mri_dir}/transforms/talairach.auto.xfm {sub_mri_dir}/transforms/talairach.xfm'
    run_with_timing(cmd)

    # talairach.lta:  convert to lta
    freesufer_home = os.environ['FREESURFER_HOME']
    cmd = f"lta_convert --src {sub_mri_dir}/orig.mgz --trg {freesufer_home}/average/mni305.cor.mgz " \
          f"--inxfm {sub_mri_dir}/transforms/talairach.xfm --outlta {sub_mri_dir}/transforms/talairach.xfm.lta " \
          f"--subject fsaverage --ltavox2vox"
    run_with_timing(cmd)

    # Since we do not run mri_em_register we sym-link other talairach transform files here
    cmd = f"ln -sf {sub_mri_dir}/transforms/talairach.xfm.lta {sub_mri_dir}/transforms/talairach_with_skull.lta"
    run_with_timing(cmd)
    cmd = f"ln -sf {sub_mri_dir}/transforms/talairach.xfm.lta {sub_mri_dir}/transforms/talairach.lta"
    run_with_timing(cmd)

    # Add xfm to nu
    cmd = f'mri_add_xform_to_header -c {sub_mri_dir}/transforms/talairach.xfm {sub_mri_dir}/orig_nu.mgz {sub_mri_dir}/nu.mgz'
    run_with_timing(cmd)


@timing_func
def creat_brainmask(sub_mri_dir: Path, need_t1=True):
    # create norm by masking nu 0.7s
    cmd = f'mri_mask {sub_mri_dir}/nu.mgz {sub_mri_dir}/mask.mgz {sub_mri_dir}/norm.mgz'
    run_with_timing(cmd)

    if need_t1:  # T1.mgz 相比 orig.mgz 更平滑，对比度更高
        # create T1.mgz from nu 96.9s
        cmd = f'mri_normalize -g 1 -seed 1234 -mprage {sub_mri_dir}/nu.mgz {sub_mri_dir}/T1.mgz'
        run_with_timing(cmd)

        # create brainmask by masking T1
        cmd = f'mri_mask {sub_mri_dir}/T1.mgz {sub_mri_dir}/mask.mgz {sub_mri_dir}/brainmask.mgz'
        run_with_timing(cmd)
    else:
        cmd = f'ln -sf {sub_mri_dir}/norm.mgz {sub_mri_dir}/brainmask.mgz'
        run_with_timing(cmd)


@timing_func
def update_aseg(fs_bin: Path, sub_mri_dir: Path,
                subject='recon'):
    # create aseg.auto including cc segmentation and add cc into aparc.DKTatlas+aseg.deep;
    # 46 sec: (not sure if this is needed), requires norm.mgz
    cmd = f'mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz ' \
          f'-lta {sub_mri_dir}/transforms/cc_up.lta {subject}'
    run_with_timing(cmd)

    # 0.8s
    seg = sub_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'
    cmd = f'{python} {fs_bin}/paint_cc_into_pred.py -in_cc {sub_mri_dir}/aseg.auto.mgz -in_pred {seg} ' \
          f'-out {sub_mri_dir}/aparc.DKTatlas+aseg.deep.withCC.mgz'
    run_with_timing(cmd)


@timing_func
def create_segstats(sub_mri_dir: Path, sub_stats_dir: Path,
                    subject='recon'):
    # if ['vol_segstats' == '1']
    # Calculate volume-based segstats for deep learning prediction (with CC, requires norm.mgz as invol)
    freesufer_home = os.environ['FREESURFER_HOME']
    cmd = f"mri_segstats --seed 1234 --seg {sub_mri_dir}/aparc.DKTatlas+aseg.deep.withCC.mgz " \
          f"--sum {sub_stats_dir}/aparc.DKTatlas+aseg.deep.volume.stats --pv {sub_mri_dir}/norm.mgz " \
          f"--empty --brainmask {sub_mri_dir}/brainmask.mgz --brain-vol-from-seg --excludeid 0 --subcortgray " \
          f"--in {sub_mri_dir}/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv " \
          "--id 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, " \
          "52, 53, 54, 58, 60, 63, 77, 251, 252, 253, 254, 255, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, " \
          "1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, " \
          "1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, " \
          "2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, " \
          f"2031, 2034, 2035 --ctab {freesufer_home}/FreeSurferColorLUT.txt --subject {subject}"
    run_with_timing(cmd)


@timing_func
def create_filled_from_brain(fsthreads: str,
                             subject='recon'):
    cmd = f'recon-all -subject {subject} -asegmerge {fsthreads}'
    run_with_timing(cmd)
    cmd = f'recon-all -subject {subject} -normalization2 {fsthreads}'
    run_with_timing(cmd)
    cmd = f'recon-all -subject {subject} -maskbfs {fsthreads}'
    run_with_timing(cmd)
    cmd = f'recon-all -subject {subject} -segmentation {fsthreads}'
    run_with_timing(cmd)
    cmd = f'recon-all -subject {subject} -fill {fsthreads}'
    run_with_timing(cmd)


@timing_func
def create_orig_nofixed(fs_bin: Path, sub_mri_dir: Path, sub_surf_dir: Path,
                        hemi: str, fsthreads: str,
                        fstess: bool,
                        subject='recon'):
    if hemi == 'lh':
        hemivalue = 255
    else:
        hemivalue = 127

    if fstess:
        # cmd = f"recon-all -subject {subject} -hemi {hemi} -tessellate -smooth1 -no-isrunning {fsthreads}"
        # run_with_timing(cmd)
        cmd = f"recon-all -subject {subject} -hemi {hemi} -tessellate -no-isrunning {fsthreads}"
        run_with_timing(cmd)
        cmd = f"recon-all -subject {subject} -hemi {hemi} -smooth1 -no-isrunning {fsthreads}"
        run_with_timing(cmd)
    else:
        # extract initial surface "?h.orig.nofix"
        cmd = f"mri_pretess {sub_mri_dir}/filled.mgz {hemivalue} {sub_mri_dir}/brain.mgz " \
              f"{sub_mri_dir}/filled-pretess{hemivalue}.mgz"
        run_with_timing(cmd)

        # Marching cube does not return filename and wrong volume info!
        cmd = f"mri_mc {sub_mri_dir}/filled-pretess{hemivalue}.mgz {hemivalue} {sub_surf_dir}/{hemi}.orig.nofix"
        run_with_timing(cmd)

        # Rewrite surface orig.nofix to fix vertex locs bug (scannerRAS instead of surfaceRAS set with mc)
        cmd = f"{python} {fs_bin}/rewrite_mc_surface.py --input {sub_surf_dir}/{hemi}.orig.nofix " \
              f"--output {sub_surf_dir}/{hemi}.orig.nofix " \
              f"--filename_pretess {sub_mri_dir}/filled-pretess{hemivalue}.mgz"
        run_with_timing(cmd)

        # Check if the surfaceRAS was correctly set and exit otherwise
        # (sanity check in case nibabel changes their default header behaviour)
        cmd = f"mris_info {sub_surf_dir}/{hemi}.orig.nofix | grep -q 'vertex locs : surfaceRAS'"
        run_with_timing(cmd)

        # Reduce to largest component (usually there should only be one)
        cmd = f"mris_extract_main_component {sub_surf_dir}/{hemi}.orig.nofix {sub_surf_dir}/{hemi}.orig.nofix"
        run_with_timing(cmd)

        # -smooth1 (explicitly state 10 iteration (default) but may change in future)
        cmd = f"mris_smooth -n 10 -nw -seed 1234 {sub_surf_dir}/{hemi}.orig.nofix {sub_surf_dir}/{hemi}.smoothwm.nofix"
        run_with_timing(cmd)


@timing_func
def create_qsphere(fs_bin: Path, sub_surf_dir: Path,
                   hemi: str, threads: int, fsthreads: str, fsqsphere: bool,
                   subject='recon'):
    # surface inflation (54sec both hemis) (needed for qsphere and for topo-fixer)
    cmd = f"recon-all -subject {subject} -hemi {hemi} -inflate1 -no-isrunning {fsthreads}"
    run_with_timing(cmd)

    if fsqsphere:
        # quick spherical mapping (2min48sec)
        cmd = f"recon-all -subject {subject} -hemi {hemi} -qsphere -no-isrunning {fsthreads}"
        run_with_timing(cmd)
    else:
        # instead of mris_sphere, directly project to sphere with spectral approach
        # equivalent to -qsphere
        # (23sec)
        cmd = f"{python} {fs_bin}/spherically_project_wrapper.py --hemi {hemi} --sdir {sub_surf_dir} " \
              f"--subject {subject} --threads={threads} --py {python} --binpath {fs_bin}/"
        run_with_timing(cmd)


@timing_func
def create_orig_fix(fs_bin: Path, fc_bin: Path, sub_mri_dir: Path, sub_surf_dir: Path,
                    threads: int, fsthreads: str,
                    fsfixed: bool, fstess: bool, fsqsphere: bool,
                    subject='recon'):
    # use FreeSurfer and FastSurfer
    if fsfixed:
        for hemi in ['lh', 'rh']:
            create_orig_nofixed(fs_bin, sub_mri_dir, sub_surf_dir, hemi, fsthreads, fstess)
            create_qsphere(fs_bin, sub_surf_dir, hemi, threads, fsthreads, fsqsphere)

            # -fix
            cmd = f'recon-all -subject {subject} -hemi {hemi} -fix -no-isrunning {fsthreads}'
            run_with_timing(cmd)

    # use FastCSR
    else:
        # sd = sub_tmp_dir
        # sub_id = 'FastCSR'
        sd = sub_mri_dir.parent.parent
        t1 = sub_mri_dir / 'orig' / '001.mgz'
        py = fc_bin / 'pipeline.py'
        cmd = f'{python} {py} --sd {sd} --sid {subject}  --t1 {t1} --optimizing_surface off --parallel_scheduling off'
        run_with_timing(cmd)
        for hemi in ['lh', 'rh']:
            orig_premesh = sub_surf_dir / f'{hemi}.orig.premesh'
            cmd = f'ln -sf {sub_surf_dir}/{hemi}.orig {orig_premesh}'
            run_with_timing(cmd)


@timing_func
def create_white_preaparc(hemi: str, threads: int, fsthreads: str,
                          fswhitepreaparc: bool,
                          subject='recon'):
    if fswhitepreaparc:
        # cmd = f'recon-all -subject {subject} -hemi {hemi} -white -no-isrunning {fsthreads}'
        cmd = f'mris_make_surfaces -aseg aseg.presurf -white white.preaparc -whiteonly -noaparc -mgz ' \
              f'-T1 brain.finalsurfs {subject} {hemi} threads {threads}'
        run_with_timing(cmd)
    else:
        cmd = f'recon-all -subject {subject} -hemi {hemi} -autodetgwstats -white-preaparc -cortex-label ' \
              f'-no-isrunning {fsthreads}'
        run_with_timing(cmd)
        # cmd = f'recon-all -subject {subject} -hemi {hemi} -no-isrunning {fsthreads}'
        # run_with_timing(cmd)
        # cmd = f'recon-all -subject {subject} -hemi {hemi} -no-isrunning {fsthreads}'
        # run_with_timing(cmd)


@timing_func
def create_inflated_sphere(hemi: str, fsthreads: str,
                           subject='recon'):
    # create nicer inflated surface from topo fixed (not needed, just later for visualization)
    cmd = f"recon-all -subject {subject} -hemi {hemi} -smooth2 -no-isrunning {fsthreads}"
    run_with_timing(cmd)

    cmd = f"recon-all -subject {subject} -hemi {hemi} -inflate2 -no-isrunning {fsthreads}"
    run_with_timing(cmd)

    cmd = f"recon-all -subject {subject} -hemi {hemi} -sphere -no-isrunning {fsthreads}"
    run_with_timing(cmd)


@timing_func
def create_surfreg(fs_bin: Path, fr_bin: Path, sub_mri_dir: Path, sub_surf_dir: Path, sub_label_dir: Path,
                   hemi: str, fssurfreg: bool, subject: str = 'recon'):
    freesufer_home = os.environ['FREESURFER_HOME']
    if fssurfreg:
        # (mr) FIX: sometimes FreeSurfer Sphere Reg. fails and moves pre and post central
        # one gyrus too far posterior, FastSurferCNN's image-based segmentation does not
        # seem to do this, so we initialize the spherical registration with the better
        # cortical segmentation from FastSurferCNN, this replaces recon-al -surfreg
        # 1. get alpha, beta, gamma for global alignment (rotation) based on aseg centers
        # (note the former fix, initializing with pre-central label, is not working in FS7.2
        # as they broke the label initializiation in mris_register)
        cmd = f"{python} {fs_bin}/rotate_sphere.py \
               --srcsphere {sub_surf_dir}/{hemi}.sphere \
               --srcaparc {sub_label_dir}/{hemi}.aparc.DKTatlas.mapped.annot \
               --trgsphere {freesufer_home}/subjects/fsaverage/surf/{hemi}.sphere \
               --trgaparc {freesufer_home}/subjects/fsaverage/label/{hemi}.aparc.annot \
               --out {sub_surf_dir}/{hemi}.angles.txt"
        run_with_timing(cmd)
        # 2. use global rotation as initialization to non-linear registration:
        with open(f'{sub_surf_dir}/{hemi}.angles.txt', 'r') as f:
            rotate = f.readline().strip()
        cmd = f"mris_register -curv -norot -rotate {rotate} \
               {sub_surf_dir}/{hemi}.sphere \
               {freesufer_home}/average/{hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \
               {sub_surf_dir}/{hemi}.sphere.reg"
        run_with_timing(cmd)
        # command to generate new aparc to check if registration was OK
        # run only for debugging
        # cmd=f"mris_ca_label -l {sub_label_dir}/{hemi}.cortex.label \
        #     -aseg {sub_mri_dir}/aseg.presurf.mgz \
        #     -seed 1234 {subject} {hemi} {sub_surf_dir}/{hemi}.sphere.reg \
        #     {sub_label_dir}/{hemi}.aparc.DKTatlas-guided.annot"
    else:
        print('Run FeatReg')
        curv = sub_surf_dir / f"{hemi}.curv"
        if not curv.exists():
            delete_curv = True
            cddir = f'cd {sub_mri_dir} &&'
            cmd = f"{cddir} mris_place_surface --curv-map {sub_surf_dir}/{hemi}.white.preaparc " \
                  f"2 10 {curv}"
            run_with_timing(cmd)
        else:
            delete_curv = False

        py = fr_bin / 'predict.py'
        sid = subject
        sd = sub_mri_dir.parent.parent
        fsd = freesufer_home
        cmd = f'{python} {py} --sid {sid} --sd {sd} --fsd {fsd} --hemi {hemi}'
        run_with_timing(cmd)

        if delete_curv:
            cmd = f'rm -f {curv}'
            run_with_timing(cmd)


@timing_func
def create_jacobian_avgcurv_cortparc(hemi: str, fsthreads: str,
                                     subject='recon'):
    cmd = f"recon-all -subject {subject} -hemi {hemi} -jacobian_white -avgcurv -cortparc " \
          f"-no-isrunning {fsthreads}"
    run_with_timing(cmd)


@timing_func
def map_seg_to_surf(fs_bin: Path, sub_surf_dir: Path, sub_label_dir: Path,
                    hemi: str,
                    subject='recon'):
    # sample input segmentation (aparc.DKTatlas+aseg orig) onto wm surface:
    # map input aparc to surface (requires thickness (and thus pail) to compute projfrac 0.5),
    # here we do projmm which allows us to compute based only on white
    # this is dangerous, as some cortices could be < 0.6 mm, but then there is no volume label probably anyway.
    # Also note that currently we cannot mask non-cortex regions here, should be done in mris_anatomical stats later
    # the smoothing helps
    freesufer_home = os.environ['FREESURFER_HOME']
    cmd = f"mris_sample_parc -ct {freesufer_home}/average/colortable_desikan_killiany.txt " \
          f"-file {fs_bin}/{hemi}.DKTatlaslookup.txt -projmm 0.6 -f 5  " \
          f"-surf white.preaparc {subject} {hemi} aparc.DKTatlas+aseg.orig.mgz aparc.DKTatlas.mapped.prefix.annot"
    run_with_timing(cmd)

    cmd = f"{python} {fs_bin}/smooth_aparc.py --insurf {sub_surf_dir}/{hemi}.white.preaparc " \
          f"--inaparc {sub_label_dir}/{hemi}.aparc.DKTatlas.mapped.prefix.annot " \
          f"--incort {sub_label_dir}/{hemi}.cortex.label --outaparc {sub_label_dir}/{hemi}.aparc.DKTatlas.mapped.annot"
    run_with_timing(cmd)


@timing_func
def create_white_pial_thickness(sub_mri_dir: Path, sub_surf_dir: Path,
                                hemi: str, fsthreads: str,
                                fswhitepial: bool,
                                subject='recon'):
    if fswhitepial:
        # must run surfreg first
        # 20-25 min for traditional surface segmentation (each hemi)
        # this creates aparc and creates pial using aparc, also computes jacobian

        cmd = f"recon-all -subject {subject} -hemi {hemi} -white " \
              f"-no-isrunning {fsthreads}"
        run_with_timing(cmd)
        cmd = f"recon-all -subject {subject} -hemi {hemi} -pial " \
              f"-no-isrunning {fsthreads}"
        run_with_timing(cmd)
    else:
        # 4 min compute white :
        cddir = f'cd {sub_mri_dir} &&'
        cmd = f"{cddir} mris_place_surface --adgws-in ../surf/autodet.gw.stats.{hemi}.dat " \
              f"--seg aseg.presurf.mgz --wm wm.mgz --invol brain.finalsurfs.mgz --{hemi} " \
              f"--i ../surf/{hemi}.white.preaparc --o ../surf/{hemi}.white --white --nsmooth 0 " \
              f"--rip-label ../label/{hemi}.cortex.label --rip-bg --rip-surf ../surf/{hemi}.white.preaparc " \
              f"--aparc ../label/{hemi}.aparc.DKTatlas.mapped.annot"
        run_with_timing(cmd)
        # 4 min compute pial :
        cmd = f"{cddir} mris_place_surface --adgws-in ../surf/autodet.gw.stats.{hemi}.dat --seg aseg.presurf.mgz " \
              f"--wm wm.mgz --invol brain.finalsurfs.mgz --{hemi} --i ../surf/{hemi}.white " \
              f"--o ../surf/{hemi}.pial.T1 --pial --nsmooth 0 --rip-label ../label/{hemi}.cortex+hipamyg.label " \
              f"--pin-medial-wall ../label/{hemi}.cortex.label --aparc ../label/{hemi}.aparc.DKTatlas.mapped.annot " \
              f"--repulse-surf ../surf/{hemi}.white --white-surf ../surf/{hemi}.white"
        run_with_timing(cmd)

        # Here insert DoT2Pial  later --> if T2pial is not run, need to softlink pial.T1 to pial!

        cddir = f'cd {sub_surf_dir} &&'
        cmd = f"{cddir} cp {hemi}.pial.T1 {hemi}.pial"
        run_with_timing(cmd)

        cddir = f'cd {sub_mri_dir} &&'
        # these are run automatically in fs7* recon-all and
        # cannot be called directly without -pial flag (or other t2 flags)
        cmd = f"{cddir} mris_place_surface --curv-map {sub_surf_dir}/{hemi}.white 2 10 {sub_surf_dir}/{hemi}.curv"
        run_with_timing(cmd)
        cmd = f"{cddir} mris_place_surface --area-map {sub_surf_dir}/{hemi}.white {sub_surf_dir}/{hemi}.area"
        run_with_timing(cmd)
        cmd = f"{cddir} mris_place_surface --curv-map {sub_surf_dir}/{hemi}.pial 2 10 {sub_surf_dir}/{hemi}.curv.pial"
        run_with_timing(cmd)
        cmd = f"{cddir} mris_place_surface --area-map {sub_surf_dir}/{hemi}.pial {sub_surf_dir}/{hemi}.area.pial"
        run_with_timing(cmd)
        cmd = f"{cddir} mris_place_surface --thickness {sub_surf_dir}/{hemi}.white {sub_surf_dir}/{hemi}.pial " \
              f"20 5 {sub_surf_dir}/{hemi}.thickness"
        run_with_timing(cmd)


@timing_func
def create_curvstats(hemi: str, fsthreads: str,
                     subject='recon'):
    # in FS7 curvstats moves here
    cmd = f"recon-all -subject {subject} -hemi {hemi} -curvstats -no-isrunning {fsthreads}"
    run_with_timing(cmd)


@timing_func
def create_ribbon(fsthreads: str,
                  subject='recon'):
    # -cortribbon 4 minutes, ribbon is used in mris_anatomical stats
    # to remove voxels from surface based volumes that should not be cortex
    # anatomical stats can run without ribon, but will omit some surface based measures then
    # wmparc needs ribbon, probably other stuff (aparc to aseg etc).
    # could be stripped but lets run it to have these measures below
    cmd = f"recon-all -subject {subject} -cortribbon {fsthreads}"
    run_with_timing(cmd)


@timing_func
def create_fs_aseg_stats(sub_mri_dir: Path, sub_surf_dir: Path,  sub_label_dir: Path,
                         threads: int, fsthreads: str,
                         fsstats: bool,
                         subject='recon'):
    if fsstats:
        # cmd = f"recon-all -subject {subject} -parcstats -cortparc2 -parcstats2 -cortparc3 -parcstats3 " \
        #       f"-pctsurfcon -hyporelabel -aparc2aseg -apas2aseg {fsthreads}"
        # run_with_timing(cmd)

        cmd = f"recon-all -subject {subject} -parcstats " \
              f"-pctsurfcon -hyporelabel -apas2aseg {fsthreads}"
        run_with_timing(cmd)

        cmd = f'cd {sub_mri_dir} && mri_surf2volseg --o aparc+aseg.mgz --label-cortex --i aseg.mgz ' \
              f'--threads {threads} ' \
              f'--lh-annot {sub_label_dir}/lh.aparc.annot 1000 ' \
              f'--lh-cortex-mask {sub_label_dir}/lh.cortex.label --lh-white {sub_surf_dir}/lh.white ' \
              f'--lh-pial {sub_surf_dir}/lh.pial --rh-annot {sub_label_dir}/rh.aparc.annot 2000 ' \
              f'--rh-cortex-mask {sub_label_dir}/rh.cortex.label --rh-white {sub_surf_dir}/rh.white ' \
              f'--rh-pial {sub_surf_dir}/rh.pial '
        run_with_timing(cmd)

        cmd = f"recon-all -subject {subject} -segstats  {fsthreads}"
        run_with_timing(cmd)
    else:
        freesufer_home = os.environ['FREESURFER_HOME']
        cmd = f"mri_aparc2aseg --s {subject} --volmask --aseg aseg.presurf.hypos " \
              f"--relabel mri/norm.mgz mri/transforms/talairach.m3z " \
              f"{freesufer_home}/average/RB_all_2016-05-10.vc700.gca mri/aseg.auto_noCCseg.label_intensities.txt"
        run_with_timing(cmd)


@timing_func
def map_stats(sub_label_dir: Path, sub_stats_dir: Path,
              subject='recon'):
    # 2x18sec create stats from mapped aparc
    for hemi in ['lh', 'rh']:
        cmd = f"mris_anatomical_stats -th3 -mgz -cortex {sub_label_dir}/{hemi}.cortex.label " \
              f"-f {sub_stats_dir}/{hemi}.aparc.DKTatlas.mapped.stats -b " \
              f"-a {sub_label_dir}/{hemi}.aparc.DKTatlas.mapped.annot " \
              f"-c {sub_label_dir}/aparc.annot.mapped.ctab {subject} {hemi} white"
        run_with_timing(cmd)


@timing_func
def create_pctsurcon_hypo_segstats(sub_mri_dir: Path, sub_surf_dir: Path, sub_label_dir: Path,
                                   threads: int, fsthreads: str,
                                   fsstats: bool, fssurfreg: bool,
                                   subject='recon'):
    # ====== Creating surfaces - pctsurfcon, hypo, segstats ======
    if not fsstats:
        # pctsurfcon (has no way to specify which annot to use, so we need to link ours as aparc is not available)
        # cddir = f'cd {sub_label_dir} &&'
        # cmd = f"{cddir} ln -sf lh.aparc.DKTatlas.mapped.annot lh.aparc.annot"
        # run_with_timing(cmd)
        # cmd = f"{cddir} ln -sf rh.aparc.DKTatlas.mapped.annot rh.aparc.annot"
        # run_with_timing(cmd)
        # for hemi in ['lh', 'rh']:
        #     cmd = f"pctsurfcon --s {subject} --{hemi}-only"
        #     run_with_timing(cmd)

        # cddir = f'cd {sub_label_dir} &&'
        # cmd = f"{cddir} rm *h.aparc.annot"
        # run_with_timing(cmd)

        cmd = f"recon-all -subject {subject} -cortparc -pctsurfcon  {fsthreads}"
        run_with_timing(cmd)

        # 25 sec hyporelabel run whatever else can be done without sphere, cortical ribbon and segmentations
        # -hyporelabel creates aseg.presurf.hypos.mgz from aseg.presurf.mgz
        # -apas2aseg creates aseg.mgz by editing aseg.presurf.hypos.mgz with surfaces
        cmd = f"recon-all -subject {subject} -hyporelabel -apas2aseg {fsthreads}"
        run_with_timing(cmd)

    # creating aparc.DKTatlas+aseg.mapped.mgz by mapping aparc.DKTatlas.mapped from surface to aseg.mgz
    # (should be a nicer aparc+aseg compared to orig CNN segmentation, due to surface updates)???
    cmd = f"mri_surf2volseg --o {sub_mri_dir}/aparc.DKTatlas+aseg.mapped.mgz --label-cortex " \
          f"--i {sub_mri_dir}/aseg.mgz --threads {threads} " \
          f"--lh-annot {sub_label_dir}/lh.aparc.DKTatlas.mapped.annot 1000 " \
          f"--lh-cortex-mask {sub_label_dir}/lh.cortex.label --lh-white {sub_surf_dir}/lh.white " \
          f"--lh-pial {sub_surf_dir}/lh.pial --rh-annot {sub_label_dir}/rh.aparc.DKTatlas.mapped.annot 2000 " \
          f"--rh-cortex-mask {sub_label_dir}/rh.cortex.label --rh-white {sub_surf_dir}/rh.white " \
          f"--rh-pial {sub_surf_dir}/rh.pial"
    run_with_timing(cmd)


@timing_func
def create_wmparc_from_mapped(sub_mri_dir: Path, sub_surf_dir: Path, sub_label_dir: Path, sub_stats_dir: Path,
                              threads: int, fsaparc: bool,
                              subject='recon'):
    freesufer_home = os.environ['FREESURFER_HOME']

    # 1m 11sec also create stats for aseg.presurf.hypos (which is basically the aseg
    # derived from the input with CC and hypos)
    # difference between this and the surface improved one above are probably tiny,
    # so the surface improvement above can probably be skipped to save time
    cmd = f"mri_segstats --seed 1234 --seg {sub_mri_dir}/aseg.presurf.hypos.mgz " \
          f"--sum {sub_stats_dir}/aseg.presurf.hypos.stats --pv {sub_mri_dir}/norm.mgz --empty " \
          f"--brainmask {sub_mri_dir}/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent " \
          f"--subcortgray --in {sub_mri_dir}/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv " \
          f"--surf-wm-vol --surf-ctx-vol --totalgray --euler " \
          f"--ctab {freesufer_home}/ASegStatsLUT.txt --subject {subject}"
    run_with_timing(cmd)

    # -wmparc based on mapped aparc labels (from input seg) (1min40sec) needs
    # ribbon and we need to point it to aparc.mapped:
    cmd = f"mri_surf2volseg --o {sub_mri_dir}/wmparc.DKTatlas.mapped.mgz --label-wm " \
          f"--i {sub_mri_dir}/aparc.DKTatlas+aseg.mapped.mgz  --threads {threads} " \
          f"--lh-annot {sub_label_dir}/lh.aparc.DKTatlas.mapped.annot 3000 " \
          f"--lh-cortex-mask {sub_label_dir}/lh.cortex.label --lh-white {sub_surf_dir}/lh.white " \
          f"--lh-pial {sub_surf_dir}/lh.pial --rh-annot {sub_label_dir}/rh.aparc.DKTatlas.mapped.annot 4000 " \
          f"--rh-cortex-mask {sub_label_dir}/rh.cortex.label " \
          f"--rh-white {sub_surf_dir}/rh.white --rh-pial {sub_surf_dir}/rh.pial"
    run_with_timing(cmd)

    # takes a few mins
    cmd = f"mri_segstats --seed 1234 --seg {sub_mri_dir}/wmparc.DKTatlas.mapped.mgz " \
          f"--sum {sub_stats_dir}/wmparc.DKTatlas.mapped.stats --pv {sub_mri_dir}/norm.mgz --excludeid 0 " \
          f"--brainmask {sub_mri_dir}/brainmask.mgz --in {sub_mri_dir}/norm.mgz --in-intensity-name norm " \
          f"--in-intensity-units MR --subject {subject} --surf-wm-vol --ctab {freesufer_home}/WMParcStatsLUT.txt"
    run_with_timing(cmd)

    # Create symlinks for downstream analysis (sub-segmentations, TRACULA, etc.)
    if not fsaparc:
        # Symlink of aparc.DKTatlas+aseg.mapped.mgz
        cddir = f'cd {sub_mri_dir} &&'
        cmd = f"{cddir} ln -sf aparc.DKTatlas+aseg.mapped.mgz aparc.DKTatlas+aseg.mgz"
        run_with_timing(cmd)
        cmd = f"{cddir} ln -sf aparc.DKTatlas+aseg.mapped.mgz aparc+aseg.mgz"
        run_with_timing(cmd)

        # Symlink of wmparc.mapped
        cddir = f'cd {sub_mri_dir} &&'
        cmd = f"{cddir} ln -sf wmparc.DKTatlas.mapped.mgz wmparc.mgz"
        run_with_timing(cmd)

        # Symbolic link for mapped surface parcellations
        cddir = f'cd {sub_label_dir} &&'
        cmd = f"{cddir} ln -sf lh.aparc.DKTatlas.mapped.annot lh.aparc.DKTatlas.annot"
        run_with_timing(cmd)
        cmd = f"{cddir} ln -sf rh.aparc.DKTatlas.mapped.annot rh.aparc.DKTatlas.annot"
        run_with_timing(cmd)


@timing_func
def create_balabels(fsthreads: str,
                    subject='recon'):
    # balabels need sphere.reg
    # can be produced if surf registration exists
    cmd = f"recon-all -subject {subject} -balabels {fsthreads}"
    run_with_timing(cmd)


@timing_func
def create_rawavg(files: str,
                  fsthreads: str,
                  subject='recon'):
    cmd = f"recon-all -subject {subject} -i {files} -motioncor {fsthreads}"
    run_with_timing(cmd)


@timing_func
def preprocess(subject_id, sub_t1: str, sub_recon_path: Path, rewrite=True):
    if (not rewrite) and os.path.exists(sub_recon_path):
        print(f'path is already exists : {sub_recon_path}')
        return

    # FastSurfer seg
    fs_home = Path.cwd() / 'FastSurfer'
    fs_recon_bin = fs_home / 'recon_surf'

    # FastCSR recon
    fc_home = Path.cwd() / 'FastCSR'

    # FeatReg recon
    fr_home = Path.cwd() / 'FeatReg' / 'featreg'

    # subject dir
    subj_mri = sub_recon_path / 'mri'
    subj_surf = sub_recon_path / 'surf'
    subj_label = sub_recon_path / 'label'
    subj_stats = sub_recon_path / 'stats'

    # =============== Volume =============== #

    # Create seg
    fastsurfer_seg(sub_t1, fs_home, subj_mri)

    # # Creating orig and rawavg from input
    creat_orig_and_rawavg(subj_mri)

    # # Create noccseg
    creat_aseg_noccseg(fs_recon_bin, subj_mri)

    # # Computing Talairach Transform and NU (bias corrected)
    creat_talairach_and_nu(fs_recon_bin, subj_mri, threads)

    # Creating brainmask from aseg and norm, and update aseg
    creat_brainmask(subj_mri, need_t1=need_t1)

    # update aseg
    update_aseg(fs_recon_bin, subj_mri, subject=subject_id)

    # Create segstats, stats/aparc.DKTatlas+aseg.deep.volume.stats
    if need_vol_segstats:
        create_segstats(subj_mri, subj_stats, subject=subject_id)

    # Creating filled from brain
    create_filled_from_brain(fsthreads=fsthreads, subject=subject_id)

    # # =============== SURFACES =============== #
    create_orig_fix(fs_recon_bin, fc_home, subj_mri, subj_surf,
                    threads, fsthreads,
                    fsfixed, fstess, fsqsphere,
                    subject=subject_id)

    for hemi in ['lh', 'rh']:
        create_white_preaparc(hemi, threads, fsthreads,
                              fswhitepreaparc,
                              subject=subject_id)
        create_inflated_sphere(hemi, fsthreads,
                               subject=subject_id)
        create_surfreg(fs_recon_bin, fr_home, subj_mri, subj_surf, subj_label,
                       hemi, fssurfreg, subject=subject_id)

        create_jacobian_avgcurv_cortparc(hemi, fsthreads,
                                         subject=subject_id)

        # 2*22s map DKatlas to aseg
        map_seg_to_surf(fs_recon_bin, subj_surf, subj_label,
                        hemi,
                        subject=subject_id)
        create_white_pial_thickness(subj_mri, subj_surf,
                                    hemi, fsthreads, fswhitepial,
                                    subject=subject_id)

    # =============== STATS =============== #
    for hemi in ['lh', 'rh']:
        # for hemi in ['lh']:
        # 2*2s ?h.curv.stats
        create_curvstats(hemi, fsthreads,
                         subject=subject_id)

    create_ribbon(fsthreads, subject=subject_id)

    create_fs_aseg_stats(subj_mri, subj_surf, subj_label, threads, fsthreads, fsstats, subject=subject_id)

    # =============== OPTIONAL STATS =============== #
    # stats DKatlas
    # map_stats(subj_label, subj_stats, subject=subject_id)

    # create_pctsurcon_hypo_segstats(subj_mri, subj_surf, subj_label, threads, fsthreads,
    #                                fsstats, fssurfreg, subject=subject_id)
    # create_wmparc_from_mapped(subj_mri, subj_surf, subj_label, subj_stats, threads, fsaparc,
    #                           subject=subject_id)
    create_balabels(fsthreads, subject=subject_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bd', required=True, help='directory of bids type')
    parser.add_argument('--fsd', default=os.environ.get('FREESURFER_HOME'),
                        help='Output directory $FREESURFER_HOME (pass via environment or here)')
    parser.add_argument('--respective', default='off',
                        help='if on, while processing T1w file respectively')
    parser.add_argument('--rewrite', default='on',
                        help='set off, while not preprocess if subject recon path exist')
    parser.add_argument('--python', default='python3',
                        help='which python version to use')

    args = parser.parse_args()
    args_dict = vars(args)

    if args.fsd is None:
        args_dict['fsd'] = '/usr/local/freesurfer'
    args_dict['respective'] = True if args.respective == 'on' else False
    args_dict['rewrite'] = True if args.rewrite == 'on' else False

    return argparse.Namespace(**args_dict)


if __name__ == '__main__':
    start_time = time.time()

    args = parse_args()

    data_path = Path(args.bd)

    python = args.python

    threads = 1
    need_t1 = True
    need_vol_segstats = False
    fstess = False
    fsqsphere = False
    fsfixed = False
    fswhitepreaparc = False
    fssurfreg = False
    fsstats = True
    fswhitepial = False

    # DeepPrep dataset_description
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep'
    derivative_deepprep_path.mkdir(exist_ok=True)

    dataset_description_file = derivative_deepprep_path / 'dataset_description.json'
    if not os.path.exists(dataset_description_file):
        dataset_description = dict()
        dataset_description['Name'] = 'DeepPrep Outputs'
        dataset_description['BIDSVersion'] = '1.4.0'
        dataset_description['DatasetType'] = 'derivative'
        dataset_description['GeneratedBy'] = [{'Name': 'deepprep', 'Version': '0.0.1'}]

        with open(dataset_description_file, 'w') as jf:
            json.dump(dataset_description, jf, indent=4)

    set_envrion(threads=threads)

    layout = bids.BIDSLayout(str(data_path), derivatives=True)
    subjs = layout.get_subjects()

    # FreeSurfer Subject Path
    deepprep_path = Path(layout.derivatives['deepprep'].root)
    deepprep_subj_path = deepprep_path / 'Recon'
    deepprep_subj_path.mkdir(exist_ok=True)
    os.environ['SUBJECTS_DIR'] = str(deepprep_subj_path)

    if threads > 1:
        fsthreads = f'-threads {threads} -itkthreads {threads}'
    else:
        fsthreads = ''

    subjs.sort()
    for subj in subjs:

        bids_t1s = layout.get(subject=subj, datatype='anat', suffix='T1w', extension='.nii.gz')
        if len(bids_t1s) <= 0:
            continue
        elif len(bids_t1s) == 1:
            subject_id = f'sub-{subj}'
            subj_recon_path = deepprep_subj_path / subject_id
            sub_t1 = bids_t1s[0].path
            print(f'{subject_id}  :  {subj_recon_path}  :  {str(sub_t1)}')
            preprocess(subject_id, sub_t1, subj_recon_path, rewrite=args.rewrite)

        else:
            if args.respective:
                for bids_t1 in bids_t1s:
                    subject = f'sub-{subj}'
                    session = f'ses-{bids_t1.entities["session"]}' if ('session' in bids_t1.entities) else None
                    run = f'run-{bids_t1.entities["run"]}' if ('run' in bids_t1.entities) else None
                    filters = [i for i in [subject, session, run] if i]
                    subject_id = f'_'.join(filters)
                    subj_recon_path = deepprep_subj_path / subject_id
                    sub_t1 = bids_t1.path
                    print(f'{subject_id}  :  {subj_recon_path}  :  {str(sub_t1)}')
                    preprocess(subject_id, sub_t1, subj_recon_path, rewrite=args.rewrite)
            # combine T1
            else:
                subject_id = f'sub-{subj}'
                subj_recon_path = deepprep_subj_path / subject_id
                if (not args.rewrite) and os.path.exists(subj_recon_path):
                    print(f'path is already exists : {subj_recon_path}')
                    continue
                t1_list = [i.path for i in bids_t1s]
                t1_list_str = ' -i '.join(t1_list)
                create_rawavg(t1_list_str, fsthreads=fsthreads, subject=subject_id)
                sub_t1 = subj_recon_path / 'mri' / 'rawavg.mgz'
                print(f'{subject_id}  :  {subj_recon_path}  :  {str(sub_t1)}')
                preprocess(subject_id, str(sub_t1), subj_recon_path)

    print('time: ', time.time() - start_time)
