import os
import sys
import json
import zipfile
from pathlib import Path
import shutil
import sh
import nibabel as nib
import numpy as np
from filters.filters import gauss_nifti, bandpass_nifti
from regressors.regressors import compile_regressors, regression
from surface_projection import surface_projection as sp
from app_pipeline.volume_projection import volume_projection as vp
from utils.utils import timing_func


def set_envrion():
    # FreeSurfer recon-all env
    os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
    os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
    os.environ['PATH'] = '/usr/local/freesurfer/bin:/usr/local/freesurfer/mni/bin:/usr/local/freesurfer/tktools:' + \
                         '/usr/local/freesurfer/fsfast/bin:' + os.environ['PATH']
    os.environ['MINC_BIN_DIR'] = '/usr/local/freesurfer/mni/bin'
    os.environ['MINC_LIB_DIR'] = '/usr/local/freesurfer/mni/lib'
    os.environ['PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'
    os.environ['MNI_PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'
    # FreeSurfer fsfast env
    os.environ['FSF_OUTPUT_FORMAT'] = 'nii.gz'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

    # # APP env
    # os.environ['NGIQ_SUBJECTS_DIR'] = '/home/weiwei/workdata/App'
    # os.environ['CODE_DIR'] = '/home/weiwei/workdata/App/indilab/0.9.9/code'
    # os.environ['TEMPL_DIR'] = '/home/weiwei/workdata/App/parameters'


def recon_step():
    pass

@timing_func
def discover_upload_step(data_path, subj):
    # decompression
    upload_file = data_path / subj / 'upload' / f'{subj}.zip'
    tmp_dir = data_path / subj / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(upload_file) as zf:
        zf.extractall(tmp_dir)

    # recon
    recon_dir = data_path / subj / 'recon' / subj
    recon_dir.mkdir(parents=True, exist_ok=True)
    reconall_file = tmp_dir / subj / f'{subj}_reconall.zip'
    with zipfile.ZipFile(reconall_file) as zf:
        zf.extractall(recon_dir)

    # nifti
    nifti_dir = data_path / subj / 'nifti'
    recon_dir.mkdir(parents=True, exist_ok=True)

    anat_src_dir = tmp_dir / subj / 'anat'
    anat_dst_dir = nifti_dir / subj / 'anat'
    shutil.copytree(anat_src_dir, anat_dst_dir)
    bold_src_dir = tmp_dir / subj / 'bold'
    bold_dst_dir = nifti_dir / subj / 'bold'
    shutil.copytree(bold_src_dir, bold_dst_dir)


def dimstr2dimno(dimstr):
    if 'x' in dimstr:
        return 0

    if 'y' in dimstr:
        return 1

    if 'z' in dimstr:
        return 2


def swapdim(infile, a, b, c, outfile):
    '''
    infile  - str. Path to file to read and swap dimensions of.
    a       - str. New x dimension.
    b       - str. New y dimension.
    c       - str. New z dimension.
    outfile - str. Path to file to create.

    Returns None.
    '''

    # Read original file.
    img = nib.load(infile)

    # Build orientation matrix.
    ornt = np.zeros((3, 2))
    order_strs = [a, b, c]
    dim_order = list(map(dimstr2dimno, order_strs))
    i_dim = np.argsort(dim_order)
    for i, dim in enumerate(i_dim):
        ornt[i, 1] = -1 if '-' in order_strs[dim] else 1

    ornt[:, 0] = i_dim

    # Transform and save.
    newimg = img.as_reoriented(ornt)
    nib.save(newimg, outfile)

@timing_func
def bold_skip_reorient(preprocess_dir, subj):
    runs = os.listdir(preprocess_dir / subj / 'bold')
    for run in runs:
        bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest.nii.gz'
        skip_bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest_skip.nii.gz'
        # skip 0 frame
        sh.mri_convert('-i', bold_file, '-o', skip_bold_file, _out=sys.stdout)
        # reorient
        reorient_bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest_reorient_skip.nii.gz'
        swapdim(str(skip_bold_file), 'x', '-y', 'z', str(reorient_bold_file))

@timing_func
def preprocess_common(data_path, subj):
    preprocess_dir = (data_path / subj / 'preprocess')
    if preprocess_dir.exists():
        shutil.rmtree(preprocess_dir)
    preprocess_dir.mkdir(exist_ok=True)

    nifti_dir = data_path / subj / 'nifti'
    bold_src_dir = nifti_dir / subj / 'bold'
    bold_dst_dir = preprocess_dir / subj / 'bold'
    shutil.copytree(bold_src_dir, bold_dst_dir)
    os.environ['SUBJECTS_DIR'] = str(data_path / subj / 'recon')

    bold_skip_reorient(preprocess_dir, subj)

    runs = os.listdir(preprocess_dir / subj / 'bold')
    for run in runs:
        src_bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest_reorient_skip.nii.gz'
        dst_bold_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip.nii.gz'
        os.rename(src_bold_file, dst_bold_file)

    # stc
    input_fname = f'{subj}_bld_rest_reorient_skip'
    output_fname = f'{subj}_bld_rest_reorient_skip_faln'
    shargs = [
        '-s', subj,
        '-d', preprocess_dir,
        '-fsd', 'bold',
        '-so', 'odd',
        '-ngroups', 1,
        '-i', input_fname,
        '-o', output_fname,
        '-nolog']
    sh.stc_sess(*shargs, _out=sys.stdout)

    # mk_template
    shargs = [
        '-s', subj,
        '-d', preprocess_dir,
        '-fsd', 'bold',
        '-funcstem', f'{subj}_bld_rest_reorient_skip_faln',
        '-nolog']
    sh.mktemplate_sess(*shargs, _out=sys.stdout)

    # mc
    shargs = [
        '-s', subj,
        '-d', preprocess_dir,
        '-per-session',
        '-fsd', 'bold',
        '-fstem', f'{subj}_bld_rest_reorient_skip_faln',
        '-fmcstem', f'{subj}_bld_rest_reorient_skip_faln_mc',
        '-nolog']
    sh.mc_sess(*shargs, _out=sys.stdout)

    # register
    for run in runs:
        mov_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'
        reg_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.register.dat'
        shargs = [
            '--bold',
            '--s', subj,
            '--mov', mov_file,
            '--reg', reg_file]
        sh.bbregister(*shargs, _out=sys.stdout)

    # mk_brainmask
    recon_dir = Path(data_path / subj / 'recon')
    for run in runs:
        func_path = preprocess_dir / subj / 'bold' / run / f'{subj}.func.aseg.nii'
        mov_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'
        reg_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.register.dat'
        shargs = [
            '--seg', recon_dir / subj / 'mri/aparc+aseg.mgz',
            '--temp', mov_file,
            '--reg', reg_file,
            '--o', func_path]
        sh.mri_label2vol(*shargs, _out=sys.stdout)

        wm_path = preprocess_dir / subj / 'bold' / run / f'{subj}.func.wm.nii.gz'
        shargs = [
            '--i', func_path,
            '--wm',
            '--erode', 1,
            '--o', wm_path]
        sh.mri_binarize(*shargs, _out=sys.stdout)

        vent_path = preprocess_dir / subj / 'bold' / run / f'{subj}.func.ventricles.nii.gz'
        shargs = [
            '--i', func_path,
            '--ventricles',
            '--o', vent_path]
        sh.mri_binarize(*shargs, _out=sys.stdout)

        mask_path = preprocess_dir / subj / 'bold' / run / f'{subj}.brainmask.nii.gz'
        shargs = [
            '--reg', reg_file,
            '--targ', recon_dir / subj / 'mri/brainmask.mgz',
            '--mov', mov_file,
            '--inv',
            '--o', mask_path]
        sh.mri_vol2vol(*shargs, _out=sys.stdout)

        binmask_path = preprocess_dir / subj / 'bold' / run / f'{subj}.brainmask.bin.nii.gz'
        shargs = [
            '--i', mask_path,
            '--o', binmask_path,
            '--min', 0.0001]
        sh.mri_binarize(*shargs, _out=sys.stdout)


def smooth_downsampling(preprocess_dir, bold_path, bldrun, subject):
    os.environ['SUBJECTS_DIR'] = str(data_path / subj / 'recon')
    fsaverage6_dir = data_path / subj / 'recon' / 'fsaverage6'
    if not fsaverage6_dir.exists():
        src_fsaverage6_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage6'
        os.symlink(src_fsaverage6_dir, fsaverage6_dir)

    fsaverage5_dir = data_path / subj / 'recon' / 'fsaverage5'
    if not fsaverage5_dir.exists():
        src_fsaverage5_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage5'
        os.symlink(src_fsaverage5_dir, fsaverage5_dir)

    fsaverage4_dir = data_path / subj / 'recon' / 'fsaverage4'
    if not fsaverage4_dir.exists():
        src_fsaverage4_dir = Path(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage4'
        os.symlink(src_fsaverage4_dir, fsaverage4_dir)

    logs_path = preprocess_dir / subject / 'logs'
    logs_path.mkdir(exist_ok=True)

    surf_path = preprocess_dir / subject / 'surf'
    surf_path.mkdir(exist_ok=True)

    bldrun_path = bold_path / bldrun
    reg_name = '%s_bld_rest_reorient_skip_faln_mc.register.dat' % (subject)
    reg_path = bldrun_path / reg_name

    resid_name = '%s_bld_rest_reorient_skip_faln' % (subject)
    resid_name += '_mc_g1000000000_bpss_resid.nii.gz'
    resid_path = bldrun_path / resid_name

    for hemi in ['lh', 'rh']:
        fs6_path = sp.indi_to_fs6(surf_path, subject, resid_path, reg_path, hemi)
        sm6_path = sp.smooth_fs6(fs6_path, hemi)
        sp.downsample_fs6_to_fs4(sm6_path, hemi)

@timing_func
def preprocess_rest(data_path, subj):
    preprocess_dir = (data_path / subj / 'preprocess')
    runs = os.listdir(preprocess_dir / subj / 'bold')
    runs = [run for run in runs if os.path.isdir(preprocess_dir / subj / 'bold' / run)]
    # TR
    task_info_file = data_path / subj / 'task_info' / 'base_data.json'
    with open(task_info_file) as jf:
        TR = json.load(jf)['TR']

    fcmri_dir = preprocess_dir / subj / 'fcmri'
    fcmri_dir.mkdir(exist_ok=True)
    bold_dir = preprocess_dir / subj / 'bold'
    for run in runs:
        # spatial_smooth
        mc_path = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.nii.gz'
        gauss_path = gauss_nifti(str(mc_path), 1000000000)

        # bandpass_filtering
        bpss_path = bandpass_nifti(gauss_path, TR)

        # compile_regressors, regression
        all_regressors_path = compile_regressors(preprocess_dir, bold_dir, run, subj, fcmri_dir, bpss_path)
        regression(bpss_path, all_regressors_path)

        # smooth_downsampling
        smooth_downsampling(preprocess_dir, bold_dir, run, subj)


def preprocess(data_path, subj):
    preprocess_common(data_path, subj)
    preprocess_rest(data_path, subj)

@timing_func
def res_proj(data_path, subj):
    os.environ['SUBJECTS_DIR'] = str(data_path / subj / 'recon')

    preprocess_dir = (data_path / subj / 'preprocess')
    runs = os.listdir(preprocess_dir / subj / 'bold')
    runs = [run for run in runs if os.path.isdir(preprocess_dir / subj / 'bold' / run)]
    vol_path = data_path / subj / 'preprocess' / subj / 'vol'
    vol_path.mkdir(exist_ok=True)
    recon_dir = data_path / subj / 'recon'
    mni_path = recon_dir / 'FSL_MNI152_FS'
    if not mni_path.exists():
        src_dir = Path(__file__).parent / 'resources' / 'FSL_MNI152_FS4.5.0'
        os.symlink(src_dir, mni_path)

    vp.T1_to_templates(data_path, subj, vol_path)
    for run in runs:
        src_resid_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
        reg_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld_rest_reorient_skip_faln_mc.register.dat'
        resid_path = preprocess_dir / subj / 'residuals'
        if not resid_path.exists():
            resid_path.mkdir(exist_ok=True)
        resid_file = resid_path / f'{subj}_bld{run}_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
        shutil.copy(src_resid_file, resid_file)
        print("Split indi native residuals")
        print("=====================================")
        indi_path, part_num = vp.sub_nii(resid_file, resid_path, run)

        brain_mask_fs2mm_binary, brain_mask_mni2mm_binary = vp.get_brain_mask_func_2mm(data_path, subj, resid_file, run)

        fpath_fs2 = resid_path / f'{subj}_bld{run}_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_FS1mm_FS2mm.nii.gz'
        fpath_mni2 = resid_path / f'{subj}_bld{run}_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_FS1mm_MNI1mm_MNI2mm.nii.gz'

        print("Project BOLD residuals to FS2mm")
        print("=====================================")
        vp.indi_to_FS2mm(subj, resid_file, reg_file, run, indi_path)
        print("Project FS2mm residuals to MNI2mm")
        print("=====================================")
        vp.indi_FS2mm_to_MNI2mm(fpath_mni2, part_num, run, brain_mask_mni2mm_binary)
        print("FS2mm residuals smooth")
        print("=====================================")
        FS2mm_dirpath = resid_path / (run + '_FS2mm')
        smooth_path = Path(str(fpath_fs2).replace('_FS2mm', '_FS2mm_sm6'))
        vp.group_nii(FS2mm_dirpath, smooth_path, part_num, brain_mask_fs2mm_binary)


if __name__ == '__main__':
    data_path = Path('data').absolute()
    subj = 'NC_15'

    set_envrion()
    discover_upload_step(data_path, subj)
    # recon_step()
    preprocess(data_path, subj)
    res_proj(data_path, subj)
