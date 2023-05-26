import os
import sys
import json
import zipfile
from path import Path as path_A
from pathlib import Path as path_D
import shutil
import sh
import nibabel as nib
import numpy as np
from filters.filters import gauss_nifti, bandpass_nifti
from regressors.regressors import compile_regressors, regression
from surface_projection import surface_projection as sp
from app_pipeline.volume_projection import volume_projection as vp
from utils.utils import timing_func
import csv
import scipy
import math
from scipy import stats, interpolate, signal
import re
import pandas as pd
from multiprocessing import Pool

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
    os.environ['NGIQ_SUBJECTS_DIR'] = '/home/zhenyu/workdata/deepprep_task'
    os.environ['CODE_DIR'] = '/home/zhenyu/workdata/App/indilab/0.9.9/code'
    os.environ['TEMPL_DIR'] = '/home/zhenyu/workdata/App/parameters'


def move_EVS(data_dir, tmp_path):
    EVS_path = path_A(data_dir / 'EVS')
    EVS_path.rmtree_p()
    for roots, dirs, files in os.walk(tmp_path):
        for dir in dirs:
            if dir.lower() == 'evs':
                origin_path = path_A(roots) / dir
                shutil.copytree(origin_path, EVS_path)
                return


@timing_func
def discover_upload_step(data_path, subj):
    # decompression
    upload_file = data_path / subj / 'upload' / f'{subj}.zip'
    tmp_dir = data_path / subj / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(upload_file) as zf:
        zf.extractall(tmp_dir)

    # move_EVS(data_path / subj, tmp_dir)

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
    detect_niftis(subj, path_A(nifti_dir))
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
    recon_dir = path_D(data_path / subj / 'recon')
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


def smooth_downsampling(preprocess_dir, bold_path, bldrun, subject, data_path):
    os.environ['SUBJECTS_DIR'] = str(data_path / subj / 'recon')
    fsaverage6_dir = data_path / subj / 'recon' / 'fsaverage6'
    if not fsaverage6_dir.exists():
        src_fsaverage6_dir = path_D(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage6'
        os.symlink(src_fsaverage6_dir, fsaverage6_dir)

    fsaverage5_dir = data_path / subj / 'recon' / 'fsaverage5'
    if not fsaverage5_dir.exists():
        src_fsaverage5_dir = path_D(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage5'
        os.symlink(src_fsaverage5_dir, fsaverage5_dir)

    fsaverage4_dir = data_path / subj / 'recon' / 'fsaverage4'
    if not fsaverage4_dir.exists():
        src_fsaverage4_dir = path_D(os.environ['FREESURFER_HOME']) / 'subjects' / 'fsaverage4'
        os.symlink(src_fsaverage4_dir, fsaverage4_dir)

    logs_path = preprocess_dir / subject / 'logs'
    logs_path.mkdir(exist_ok=True)

    surf_path = preprocess_dir / subject / 'surf'
    surf_path.mkdir(exist_ok=True)

    bldrun_path = bold_path / bldrun
    reg_name = '%s_bld_rest_reorient_skip_faln_mc.register.dat' % (subject)
    reg_path = bldrun_path / reg_name
    reg_run_path = bldrun_path / reg_name.replace('bld', f'bld{bldrun}')
    reg_path.replace(reg_run_path)

    resid_name = '%s_bld_rest_reorient_skip_faln' % (subject)
    resid_name += '_mc_g1000000000_bpss_resid.nii.gz'
    resid_path = bldrun_path / resid_name
    resid_run_path = bldrun_path / resid_name.replace('bld', f'bld{bldrun}')
    resid_path.replace(resid_run_path)

    for hemi in ['lh', 'rh']:
        fs6_path = sp.indi_to_fs6(surf_path, subject, resid_run_path, reg_run_path, hemi)
        sm6_path = sp.smooth_fs6(fs6_path, hemi)
        sp.downsample_fs6_to_fs4(sm6_path, hemi)


@timing_func
def preprocess_rest(data_path, subj):
    print(subj)
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
        smooth_downsampling(preprocess_dir, bold_dir, run, subj, data_path)


def preprocess(data_path, subj):
    # preprocess_common(data_path, subj)
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
        src_dir = path_D(__file__).parent / 'resources' / 'FSL_MNI152_FS4.5.0'
        os.symlink(src_dir, mni_path)

    vp.T1_to_templates(data_path, subj, vol_path)
    for run in runs:
        src_resid_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
        reg_file = preprocess_dir / subj / 'bold' / run / f'{subj}_bld{run}_rest_reorient_skip_faln_mc.register.dat'
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
        smooth_path = path_D(str(fpath_fs2).replace('_FS2mm', '_FS2mm_sm6'))
        vp.group_nii(FS2mm_dirpath, smooth_path, part_num, brain_mask_fs2mm_binary)


def snm_task(data_dir, subj):
    data_path = data_dir / subj
    mni_path = data_path / 'recon' / 'FSL_MNI152_FS'
    code_dir = data_dir / 'indilab/0.9.9/code'
    if not mni_path.exists():
        mni_source_path = code_dir / 'templates/volume/FSL_MNI152_FS4.5.0'
        mni_source_path.symlink(mni_path)

    run_path_obj, common_obj = check_inputs(data_path, subj)
    common_obj = prepare_env(common_obj, data_path)

    snm_task_process_main(subj, run_path_obj, common_obj)
    # task_good_run_list, conditions_list = snm_task_process_main(subj, run_path_obj, common_obj)

    # prepare_output(subj, common_obj)

    # write_cluster_metrics(subj, common_obj, conditions_list)

    pass


def check_inputs(data_path, subj):
    recon_path = data_path / 'recon' / subj
    preproc_path = data_path / 'preprocess' / subj
    brainmask_path = recon_path / 'mri' / 'brainmask.mgz'
    aparc_path = recon_path / 'mri' / 'aparc+aseg.mgz'
    aseg_path = recon_path / 'mri' / 'aseg.mgz'

    # Check the task design information files.
    evs_path = data_path / 'EVS'
    task_info_path = data_path / 'task_info'
    csv2tsv(evs_path, task_info_path)
    json_path = task_info_path / 'base_data.json'
    with open(json_path, 'r') as js_obj:
        uni_lists = json.load(js_obj)
    task_run_list = uni_lists['task']
    TR = uni_lists.get('TR')
    skipframes = uni_lists.get('skipframes')
    if skipframes is None:
        skipframes = 0
    common_obj = {
        "brainmask": brainmask_path,
        "aparc": aparc_path,
        "aseg": aseg_path,
        "TR": TR,
        "skipframes": skipframes,
        "recon_path": recon_path}

    # Check the match of base_data.json and tsv files.
    # Output many paths.
    run_path_obj = {}
    for task_run in task_run_list:
        task_path_obj = {}
        tsv_path = task_info_path / (task_run + '.tsv')
        task_path_obj['tsv_path'] = tsv_path

        bold_path = preproc_path / 'bold' / task_run
        # nii_path = bold_path / (subj + '_bld' + task_run + '_rest_reorient_skip_faln_mc.nii.gz')
        nii_path = bold_path / (subj + '_bld_rest_reorient_skip_faln_mc.nii.gz')
        task_path_obj['nii_path'] = nii_path

        # reg_path = preproc_path / 'registration_data' / 'bold' / task_run / (subj + '_bld' + task_run + \
        #            '_rest_reorient_skip_faln_mc.register.dat')
        # reg_path = bold_path / (subj + '_bld' + task_run + \
        #            '_rest_reorient_skip_faln_mc.register.dat')
        reg_path = bold_path / (subj + '_bld_rest_reorient_skip_faln_mc.register.dat')
        task_path_obj['reg_path'] = reg_path

        run_path_obj[task_run] = task_path_obj

    return run_path_obj, common_obj

def csv2tsv(evs_path, task_info_path):
    for csv in os.listdir(evs_path):
        tev_name = csv.split('.')[0] + '.tsv'
        csv_file = os.path.join(evs_path, csv)
        data_tmp = pd.read_csv(csv_file)
        df = pd.DataFrame(data_tmp)
        condition_list = list(df['Condition'].loc[7:])
        start_list = list(df['StartTime'].loc[7:])
        duration_list = list(df['Duration'].loc[7:])
        df_new = pd.DataFrame(
            {'condition': condition_list, 'start': start_list, 'duration': duration_list, 'value': 1})
        df_new.to_csv(task_info_path / tev_name, sep='\t', index=False)
        print()





def prepare_env(common_obj, data_dir):
    recon_path = data_dir / 'recon'
    os.environ['SUBJECTS_DIR'] = recon_path
    task_path = data_dir / 'snm_task'
    task_path.makedirs_p()

    viz_path = data_dir / 'viz' / 'snm_task'
    viz_path.makedirs_p()
    common_obj['task_path'] = task_path
    common_obj['viz_path'] = viz_path

    mni_path = recon_path / 'FSL_MNI152_FS'
    common_obj['mni_path'] = mni_path

    fs_path = path_A(os.environ['FREESURFER_HOME'])
    for fs_avg in ['fsaverage4', 'fsaverage6']:
        fs_avg_link = recon_path / fs_avg
        if not fs_avg_link.exists():
            fs_avg_path = fs_path / 'subjects' / fs_avg
            if not fs_avg_path.exists():
                raise Exception("No such directory: " + fs_avg_path)
            fs_avg_path.symlink(fs_avg_link)
    common_obj['fs6_path'] = recon_path / 'fsaverage6'
    return common_obj


def snm_task_process_main(subject, run_path_obj, common_obj):
    '''
    Task process main function.
    Generate many zvalues of contrast condition in native and FS1mm space.
        subject       -str. Subject id
        run_path_obj  -dict. Object of task runs tsv files path, nii files path, register files path.
        common_obj - dict. Object of task runs masks path
    '''
    task_run_list = sorted(run_path_obj.keys())
    task_path = common_obj['task_path']
    viz_path = common_obj['viz_path']
    conditions_list = []

    # Generate zvalues in each run.
    task_good_run_list = []
    for task_run in task_run_list:
        onerun_paths = run_path_obj[task_run]
        conditions_str = generate_zvalues(subject, task_run, onerun_paths, common_obj)
        if conditions_str:
            task_good_run_list.append(task_run)
            conditions_list += conditions_str

    # Remove duplicate conditions.
    conditions_list = sorted(list(set(conditions_list)))

    # Merge many 3D zvalues into one 3D result.
    if len(task_good_run_list) != 0:
        aggregate_zvalues(subject, task_path, viz_path, task_good_run_list, conditions_list)

        return task_good_run_list, conditions_list
    else:
        return None


def generate_zvalues(subject, task_run, onerun_paths, common_obj):
    '''
    Generate zvaluse main function.
    Includes get masks of brain and nuisance, get brain signals and nuisance signals,
             read task design and conv it with hrf curve,
             generate the contrast map,
             use SNM model to get the sparse coefficient map,
             use t-test,z-test to get the zvalues of contrast map,
        subject       -str. Subject id
        task_run      -str. Task run number
        onerun_paths  -dict. Paths of nii.gz file, tsv file, register file in current task run.
        common_obj -dict. Paths of brainmask, aseg files.
    '''
    # Prepare the paths.
    nii_path = onerun_paths['nii_path']
    tsv_path = onerun_paths['tsv_path']
    reg_path = onerun_paths['reg_path']
    mni_path = common_obj['mni_path']
    brainmask_path = common_obj['brainmask']
    aparc_path = common_obj['aparc']
    TR_json = common_obj['TR']
    skipframes = common_obj['skipframes']
    onerun_paths['task_path'] = common_obj['task_path']
    basic_info = {
        "subject": subject,
        "task_run": task_run,
        "mni_path": mni_path}

    # Smooth of nii files.
    sm6_nii_path = smooth_nii(nii_path)
    onerun_paths['sm6_nii_path'] = sm6_nii_path

    # Get masks of brain and nuisance voxels part.
    brain_mask_path, wm_nii, ventricles_nii = get_masks(sm6_nii_path, brainmask_path, aparc_path, reg_path)

    # Get signals of brain and nuisance voxels part.
    nuisance_signals, brain_signals, basic_info, mask_brain = get_signals(
        sm6_nii_path, brain_mask_path, wm_nii, ventricles_nii, basic_info)

    # Get TR.
    TR_nii = get_TR(sm6_nii_path.parent.parent, subject, task_run)
    TR = check_TR_nowarn(TR_json, TR_nii)
    basic_info['TR'] = TR
    basic_info['interp'] = 100

    # Read task design from tsv files.
    tasks_design = read_task_design(basic_info, tsv_path, TR, skipframes)

    # Generate conv EVs.
    tasks_convEVs = conv_hrf(basic_info, tasks_design, TR)

    # Make contrast mat.
    contrast_mat = make_contrast_mat(tasks_convEVs)

    # SNM main function, generate zvalues.
    snm_main(nuisance_signals, brain_signals, tasks_convEVs, contrast_mat, basic_info, mask_brain, onerun_paths)

    # Return condition str list.
    conditions_str = sorted(tasks_convEVs.keys())
    return conditions_str


def smooth_nii(nii_path):
    sm6_nii_path = path_A(nii_path.replace('.nii.gz', '_sm6.nii.gz'))
    shargs = ['--fwhm', '6']
    mri_fwhm_ng(nii_path, sm6_nii_path, shargs)
    return sm6_nii_path


def mri_fwhm_ng(input_path, output_path, other_shargs=None):
    shargs = [
        '--i', input_path,
        '--o', output_path]
    if other_shargs is not None:
        shargs += other_shargs
    sh.mri_fwhm(*shargs)


def get_masks(nii_path, brain_mask_path, aparc_mask_path, reg_path):
    '''
    Project brainmask and aparc files from fs6 into native space.
        nii_path           -path. Path of task run nii.gz after preprocess.
        brain_mask_path:   -path. Path of brainmask.mgz
        aparc_mask_path      -path. Path of aparc+aseg.mgz.
        reg_path:          -path. Path of register file of current task run.
    returns
        brain_mask_nii -path. Path of %s.brainmask.nii.gz
        aparc_mask_nii -path. Path of %s.aparc+aseg.nii.gz.
    '''
    bld_path = nii_path.parent
    brain_mask_nii = bld_path / 'brainmask.func.nii.gz'
    aseg_mask_nii = bld_path / 'aparc+aseg.func.nii'
    wm_nii = bld_path / 'wm.func.nii'
    ventricles_nii = bld_path / 'ventricles.func.nii'

    mri_label2vol_ng(input_path=aparc_mask_path,
                     output_path=aseg_mask_nii,
                     template_path=nii_path,
                     reg_path=reg_path
                     )
    shargs_wm = [
        '--wm',
        '--erode', '1']
    mri_binarize_ng(input_path=aseg_mask_nii,
                    output_path=wm_nii,
                    other_shargs=shargs_wm)
    shargs_ventricles = [
        '--ventricles',
        '--erode', '1']
    mri_binarize_ng(input_path=aseg_mask_nii,
                    output_path=ventricles_nii,
                    other_shargs=shargs_ventricles)
    aseg_mask_nii.remove_p()
    other_shargs = ['--inv']
    mri_vol2vol_ng(input_path=nii_path,
                   output_path=brain_mask_nii,
                   target_path=brain_mask_path,
                   reg_path=reg_path,
                   other_shargs=other_shargs)
    return brain_mask_nii, wm_nii, ventricles_nii


def get_signals(fpath, brain_mask_path, wm_nii, ventricles_nii, basic_info):
    '''
    Get nuisance signals and brain signals in 2D format
    Format (1D-data, nframes) , like (2882,104)
        fpath   -path. Path of task run nii.gz after preprocess.
        brain_mask_nii -path. Path of %s.brainmask.nii.gz
        aseg_mask_nii -path. Path of %s.aseg.nii.gz.
    returns
        nuisance_signals  -ndarray.2D nuisance signals.
        brain_signals     -ndarray.2D brain signals.
        nframes           -int. Number of frames in fpath file.
        aff               -ndarray. Affine of fpath image.
        hdr               -NiftiHeader. Header of fpath image.
    '''
    # Get task run nii.gz after preprocess.
    f_img = nib.load(fpath)
    f_data = f_img.get_data()
    aff = f_img.affine
    hdr = f_img.header

    # Get brain signals.
    brain_img = nib.load(brain_mask_path)
    data_brain = brain_img.get_data()
    nframes = f_data.shape[3]

    # Get nuisance signals.
    wm_img = nib.load(wm_nii)
    wm_data = wm_img.get_data()
    ventricles_img = nib.load(ventricles_nii)
    ventricles_data = ventricles_img.get_data()

    wm_mask = wm_data != 0
    ventricles_mask = ventricles_data != 0

    mask_nuisance = wm_mask + ventricles_mask
    mask_brain = data_brain != 0

    # make the data coordination the same to matlab
    mask_nuisance_T = np.swapaxes(mask_nuisance, 0, 1)
    mask_brain_T = np.swapaxes(mask_brain, 0, 1)
    f_data = np.swapaxes(f_data, 0, 1)

    # Use mask and reshape signals into 1D.
    mask_nuisance_1D = mask_nuisance_T.flatten(order='F')
    mask_brain_1D = mask_brain_T.flatten(order='F')
    f_data = f_data.reshape((f_data.shape[0] * f_data.shape[1] * f_data.shape[2], nframes), order='F')
    nuisance_signals = np.mat(f_data[mask_nuisance_1D])
    brain_signals = np.mat(f_data[mask_brain_1D])

    if nuisance_signals.shape[0] > 1000:
        nuisance_signals_downsample = np.mat(np.zeros((1000, nframes)))
        for idx_noise_dict in range(1000):
            nuisance_signals_downsample[idx_noise_dict] = \
                nuisance_signals[
                int(round(nuisance_signals.shape[0] * 1.0 * idx_noise_dict / 1000)), :]
        nuisance_signals = nuisance_signals_downsample
    # Save basic info of nii img.
    basic_info["nframes"] = nframes
    basic_info["aff"] = aff
    basic_info["hdr"] = hdr
    return nuisance_signals.T, brain_signals.T, basic_info, mask_brain_T


def mri_convert_ng(input_path, output_path, other_shargs=None):
    shargs = [
        input_path,
        output_path]
    if other_shargs is not None:
        shargs += other_shargs
    sh.mri_convert(*shargs)


def mri_label2vol_ng(input_path, output_path, template_path, reg_path, other_shargs=None):
    shargs = [
        '--seg', input_path,
        '--o', output_path,
        '--temp', template_path,
        '--reg', reg_path]
    if other_shargs is not None:
        shargs += other_shargs
    sh.mri_label2vol(*shargs)


def mri_vol2vol_ng(input_path, output_path, target_path, reg_path, other_shargs=None):
    shargs = [
        '--mov', input_path,
        '--o', output_path,
        '--targ', target_path,
        '--reg', reg_path]
    if other_shargs is not None:
        shargs += other_shargs
    sh.mri_vol2vol(*shargs)


def mri_binarize_ng(input_path, output_path, other_shargs=None):
    shargs = [
        '--i', input_path,
        '--o', output_path]
    if other_shargs is not None:
        shargs += other_shargs
    sh.mri_binarize(*shargs)


def mri_fwhm_ng(input_path, output_path, other_shargs=None):
    shargs = [
        '--i', input_path,
        '--o', output_path]
    if other_shargs is not None:
        shargs += other_shargs
    sh.mri_fwhm(*shargs)


def mri_vol2surf_ng(input_path, output_path, reg_path, hemi, other_shargs):
    shargs = [
        '--mov', input_path,
        '--o', output_path,
        '--reg', reg_path,
        '--hemi', hemi]
    if other_shargs is not None:
        shargs += other_shargs
    sh.mri_vol2surf(*shargs)


def get_nii_TR(fpath):
    '''
    Get TR, in seconds, from the MRI file located at Path fpath. Returns float.
    '''
    return float(sh.mri_info('--TR', fpath)) / 1000


def get_TR(bold_path, subject, bldrun):
    '''
    Get the time interval of frames in the task run.
        bold_path  -path.
        bldrun     -str.
        subject    -str.
    returns
        tr         -float. Time interval of frames in the task run.
    '''
    # Extract TR and scale it appropriately.
    bld_path = bold_path / bldrun
    for file in bld_path.files():
        if '%s_bld%s_rest_reorient_skip_faln_mc.nii.gz' % (subject, bldrun) in file:
            return get_nii_TR(file)
    return


def check_TR_nowarn(TR_json, TR_nii):
    '''
    Check TR.
    if only one is legal (0.2<TR<10),use that one.
    if TR_json and TR_nii match, return;
    if not match,use TR_json.
    No warn.
    '''
    TR_json = float(TR_json if TR_json is not None else 0)
    if not check_TR(TR_json):
        if not check_TR(TR_nii):
            raise Exception()
        return TR_nii
    return TR_json


def check_TR(TR):
    return (TR >= 0.2) and (TR <= 10.0)


def read_task_design(basic_info, tsv_path, tr, skipframes):
    '''
    Read task design from tsv files.
        nframes     -int. Number of frames.
        tsv_path    -path. Path of tsv files.
        tr          -float. Time interval between frames.
    returns
        tasks_design  -dict. Object of task block design.
    '''
    nframes = basic_info['nframes']
    interp = basic_info['interp']
    # Minus the skip part from prerecon.
    tasks_design = {}
    csv.register_dialect('tsv', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(tsv_path) as tsv_file:
        tsv_task = csv.DictReader(tsv_file, dialect='tsv')
        for row in tsv_task:
            condition = row['condition']
            if condition not in tasks_design.keys():
                tasks_design[condition] = np.zeros((nframes * interp,))
            design_one = tasks_design[condition]
            start = max(int(round((float(row['start']) / tr - 1 - skipframes) * interp)), 0)
            dur = int(round(float(row['duration']) / tr * interp))
            if (start + dur) > nframes * interp:
                return None
            design_one[start: start + dur + 1] = row['value']
            tasks_design[condition] = design_one

    return tasks_design


def conv_hrf(basic_info, tasks_design, tr):
    '''
    Conv the block task design and hrf curve.
        tasks_design  -dict. Object of task block design.
        tr            -float. Time interval of frames in the task run.
    returns
        tasks_design  -dict. Object of task convEVs.
    '''
    interp = basic_info['interp']
    nframes = basic_info['nframes']
    hrf_signal = _hrf(tr / interp)
    convEVs_tmp = None
    for task_run in tasks_design.keys():
        convEVs_tmp = scipy.signal.fftconvolve(tasks_design[task_run], hrf_signal)
        tasks_design[task_run] = convEVs_tmp[:nframes * interp][::interp]

    return tasks_design


def make_contrast_mat(tasks_convEVs):
    '''
    Makes contrast matrix.
        tasks_convEVs - dict. Object of task run convEVs.
    returns
        contrast_mat
    '''
    num_conditions = len(list(tasks_convEVs.keys()))
    contrast_mat = np.zeros((pow(num_conditions, 2), num_conditions))
    contrast_style = np.zeros((num_conditions,))
    idx = 0
    for num in range(num_conditions):
        contrast_one = contrast_style.copy()
        contrast_one[num] = 1
        contrast_mat[idx] = contrast_one
        idx += 1
    for left in range(num_conditions - 1):
        for right in range(left + 1, num_conditions):
            contrast_one = contrast_style.copy()
            contrast_one[left] = 1
            contrast_one[right] = -1
            contrast_mat[idx] = contrast_one
            idx += 1
    for num in range(num_conditions, int(num_conditions * (num_conditions + 1) / 2.0)):
        contrast_one = -contrast_mat[num]
        contrast_mat[idx] = contrast_one
        idx += 1
    return contrast_mat


def _hrf(tr):
    '''
    Generate the hrf curve.
         tr   -float. Time interval of frames in the task run.
    returns
        hrf   -ndarray. 1D array hrf curve.
    '''

    # based on https://github.com/neurodebian/spm12/blob/master/spm_hrf.m
    p = [6, 16, 1, 1, 6, 0, 32]
    fMRI_T = 16
    first_par = p[0]
    second_par = p[1]
    dt = tr * 1.0 / fMRI_T
    u = [i for i in range(int(round(p[6] / dt)))]
    for i in range(len(u)):
        u[i] = _gpdf(i, first_par, dt) - _gpdf(i, second_par, dt) / first_par
    hrf = u[0::fMRI_T]
    hrf = np.transpose(hrf / np.sum(hrf))
    return hrf


def _gpdf(idx, h, dt):
    # Also based on https://github.com/neurodebian/spm12/blob/master/spm_hrf.m
    return pow(dt, h) * pow(idx, h - 1) * np.exp(-idx * dt) / math.gamma(h)


def snm_main(nuisance_signals, brain_signals, tasks_convEVs, contrast_mat, basic_info, mask_brain, onerun_paths):
    '''
    SNM main function, generate zvalues.
        nuisance_signals:    -ndarray. (nframes, num_nuisance_signals)
        brain_signals:       -ndarray. (nframes, num_brain_signals)
        tasks_convEVs:       -dict. ['condition name': convEV]
        contrast_mat:        -ndarray. (num_conditions^2, num_conditions)
        basic_info:          -dict. Basic info.
        mask_brain           -ndarray. Mask of brain 3D.
        task_path            -path. $DATA_DIR/'snm_task'
        reg_path             -path. Register files of each bold run.
    '''

    nframes = basic_info['nframes']
    num_conditions = len(list(tasks_convEVs.keys()))
    num_nuisance_signals = nuisance_signals.shape[1]
    num_brain_signals = brain_signals.shape[1]
    num_combine_signal = num_brain_signals + num_nuisance_signals
    num_task_contrast = pow(num_conditions, 2)
    num_noise_dictionary = min(NUM_NOISE_DICTIONARY_, num_nuisance_signals)

    # Save info into basic_info
    basic_info['num_conditions'] = num_conditions
    basic_info['num_nuisance_signals'] = num_nuisance_signals
    basic_info['num_brain_signals'] = num_brain_signals
    basic_info['num_combine_signal'] = num_combine_signal
    basic_info['num_task_contrast'] = num_task_contrast
    basic_info['num_noise_dictionary'] = num_noise_dictionary

    # Get expected response list.
    idx = 0
    expected_response = np.zeros((nframes, num_conditions))
    for condition in sorted(tasks_convEVs.keys()):
        expected_response[:, idx] = tasks_convEVs[condition]
        idx += 1

    # Normalize signals and remove the zero signals.
    nuisance_signals = normalize(nuisance_signals)
    brain_signals = normalize(brain_signals)
    nuisance_signals, nuisance_signals_nonzeromask = non_zeros(nuisance_signals)
    brain_signals, brain_signals_nonzeromask = non_zeros(brain_signals)
    basic_info['num_nuisance_signal_nonzero'] = nuisance_signals.shape[1]
    combine_signal_nonzero_mask = np.concatenate((nuisance_signals_nonzeromask, brain_signals_nonzeromask), axis=0)

    # Make the noise dictionary.
    noise_dictionary = make_noise_dictionary(nuisance_signals, basic_info)
    noise_dictionary = normalize(noise_dictionary)
    expected_response = normalize(expected_response)

    # Downsample signals into 1000.
    combine_signal = np.concatenate((nuisance_signals, brain_signals), axis=1)
    num_combine_signal_nonzero = combine_signal.shape[1]
    basic_info['num_combine_signal_nonzero'] = num_combine_signal_nonzero
    combine_signal_downsample = np.mat(np.zeros((nframes, NUM_COMBINE_DOWNSAMPLE_)))

    for idx_downsample in range(NUM_COMBINE_DOWNSAMPLE_):
        combine_signal_downsample[:, idx_downsample] = \
            combine_signal[:, int(round(num_combine_signal_nonzero * 1.0 * idx_downsample / NUM_COMBINE_DOWNSAMPLE_))]

    # Calculate the sparse coefficinent mat.
    noise_response_dictionarty = np.concatenate((expected_response, noise_dictionary), axis=1)
    sparse_coefficient = cal_sparse_coefficient(combine_signal_downsample, noise_response_dictionarty, basic_info,
                                                num_combine_downsample=NUM_COMBINE_DOWNSAMPLE_,
                                                num_brain_noise=NUM_BRAIN_NOISE_)

    # Assign combine signals to downsample signals.
    corrmat_upsample = AlmightyCorrcoefEinsumOptimized(
        np.array(combine_signal, np.float16), np.array(combine_signal_downsample, np.float16))
    idx_upsample = np.argsort(corrmat_upsample.T, axis=1)[:, -NUM_REPEAT_DENOISE_:]
    for line in range(idx_upsample.shape[0]):
        idx_upsample[line] = idx_upsample[line][::-1]

    # GLM with AR, loop each downsampled signal, get zvalues.
    wls_mat, cov_mat, icov_mat = make_wls_mat(basic_info)
    zvalue_combine_nonzero, count_upsample = GLM(expected_response, noise_dictionary, combine_signal,
                                                 sparse_coefficient, idx_upsample, wls_mat, cov_mat, icov_mat,
                                                 contrast_mat, basic_info)

    # Upsampling and save
    zvalue_brain, zvalue_nuisance = control_upsample(zvalue_combine_nonzero, count_upsample,
                                                     combine_signal_nonzero_mask, basic_info)

    # FPR control
    # zvalue_corr_brain = FPR_control(zvalue_brain, zvalue_nuisance, num_task_contrast)

    # Save zvalues in native and FS1mm space.
    condition_strs = sorted(tasks_convEVs.keys())
    save_zvalues(zvalue_brain, condition_strs, contrast_mat, basic_info, mask_brain, onerun_paths)


def normalize(signals):
    '''
    Normalize noise dictionary.
    '''
    for idx_signal in range(signals.shape[1]):
        signal_temp = signals[:, idx_signal] - np.mean(signals[:, idx_signal])
        signal_temp = signal_temp / (np.std(signals[:, idx_signal], ddof=1))
        signals[:, idx_signal] = signal_temp
    return signals


@timing_func
def make_noise_dictionary(nuisance_signals, basic_info):
    '''
    Make noise dictionary.
    '''
    # Get basic info.
    num_noise_dictionary = basic_info['num_noise_dictionary']
    num_nuisance_signals = basic_info['num_nuisance_signals']
    nframes = basic_info['nframes']
    num_nuisance_signal_nonzero = basic_info['num_nuisance_signal_nonzero']

    # Use uniform sampling to get initial noise dictionary.
    noise_dictionary = np.mat(np.zeros((nframes, num_noise_dictionary)))
    for idx_noise_dict in range(num_noise_dictionary):
        noise_dictionary[:, idx_noise_dict] = \
            nuisance_signals[:, int(round(num_nuisance_signal_nonzero * 1.0 * idx_noise_dict / num_noise_dictionary))]

    # Use GLM to optimize the noise dictionary.
    for idx_iter in range(NUM_MAX_ITER_):
        sparse_coefficient = np.zeros((num_noise_dictionary, num_nuisance_signals))
        sparse_coefficient = np.mat(sparse_coefficient)
        for idx_nuisance_signal in range(num_nuisance_signals):
            sparse_coefficient_tmp = np.zeros((num_noise_dictionary, 1))
            sparse_coefficient_tmp = np.mat(sparse_coefficient_tmp)
            sparse_idx = []
            for idx_noise in range(NUM_TRAIN_NOISE_):
                residual_tmp = nuisance_signals[:, idx_nuisance_signal] - noise_dictionary * sparse_coefficient_tmp
                residual_tmp = np.mat(residual_tmp)
                if np.sum(np.square(residual_tmp)) < 1e-3:
                    break
                sparse_idx_new = np.argmax(abs(residual_tmp.T * noise_dictionary))
                sparse_idx.append(sparse_idx_new)
                sparse_idx = list(set(sparse_idx))
                sparse_coefficient_tmp[sparse_idx] = np.linalg.pinv(
                    noise_dictionary[:, sparse_idx]) * nuisance_signals[:, idx_nuisance_signal]
            sparse_coefficient[:, idx_nuisance_signal] = sparse_coefficient_tmp

        for idx_noise in range(num_noise_dictionary):
            idx_noise_mask = sparse_coefficient[idx_noise] != 0
            idx_noise_mask = np.array(idx_noise_mask)[0]
            if not idx_noise_mask.any():
                continue
            noise_dictionary_tmp = np.delete(noise_dictionary, idx_noise, 1)
            sparse_coefficient_tmp = np.delete(sparse_coefficient, idx_noise, 0)
            sparse_coefficient_tmp = sparse_coefficient_tmp[:, idx_noise_mask]
            residual_tmp = nuisance_signals[:, idx_noise_mask] - noise_dictionary_tmp * sparse_coefficient_tmp
            eigen_vector_left, eigen_value, eigen_vector_right = np.linalg.svd(residual_tmp)
            eigen_vector_left = np.mat(eigen_vector_left)
            eigen_vector_right = np.mat(eigen_vector_right.T)
            sparse_coefficient[idx_noise, idx_noise_mask] = (eigen_value[0] * eigen_vector_right[:, 0] * np.sign(
                eigen_vector_left[:, 0].T * noise_dictionary[:, idx_noise])).T
            noise_dictionary[:, idx_noise] = eigen_vector_left[:, 0] * np.sign(
                eigen_vector_left[:, 0].T * noise_dictionary[:, idx_noise])

    return noise_dictionary


@timing_func
def make_wls_mat(basic_info):
    # Get basic info.
    TR = basic_info['TR']
    nframes = basic_info['nframes']

    autocorr_mat = np.zeros((nframes, nframes))
    for idx_frame in range(nframes):
        for row in range(nframes):
            autocorr_mat[idx_frame][row] = pow(pow(np.exp(-1), TR), abs(row - idx_frame))
    identity_mat = np.eye(nframes)
    cov_mat = []
    icov_mat = []
    wls_mat = []
    # Because np.linalg.eig is't same with eig in matlab, but np.linalg.eigh is.
    # Link https://blog.csdn.net/qq_26004387/article/details/88166611
    for idx_lambda in range(101):
        cov_mat_one = np.mat((idx_lambda * autocorr_mat + identity_mat) / (idx_lambda + 1))
        cov_mat.append(cov_mat_one)
        icov_mat.append(np.linalg.inv(cov_mat_one))
        eigen_value_one, eigen_vector_one = np.linalg.eigh(cov_mat_one)
        wls_mat_one = np.diag(1 / np.sqrt(eigen_value_one)) * eigen_vector_one.T
        wls_mat.append(wls_mat_one)
    return wls_mat, cov_mat, icov_mat


def non_zeros(signals):
    idx_bool = []
    for idx_signal in range(signals.shape[1]):
        signal_temp = signals[:, idx_signal]
        idx_bool.append(pow(signal_temp.max(), 2) > 1e-3)
    signals_nonzero = signals[:, idx_bool]
    return signals_nonzero, idx_bool


def cal_sparse_coefficient(combine_signal_downsample, noise_response_dictionary, basic_info,
                           num_combine_downsample, num_brain_noise):
    # Get basic info.
    num_conditions = basic_info['num_conditions']
    num_noise_dictionary = basic_info['num_noise_dictionary']
    noise_response_dictionary = np.mat(noise_response_dictionary)
    combine_signal_downsample = np.mat(combine_signal_downsample)
    sparse_coefficient = np.mat(np.zeros((num_noise_dictionary + num_conditions, num_combine_downsample)))
    for idx_downsample in range(num_combine_downsample):
        sparse_idx = [i for i in range(num_conditions)]
        sparse_coefficient[sparse_idx, idx_downsample] = \
            (np.linalg.pinv(noise_response_dictionary[:, sparse_idx]) * combine_signal_downsample[:, idx_downsample]).T
        for id_noise in range(num_brain_noise):
            residual_tmp = combine_signal_downsample[:, idx_downsample] - \
                           noise_response_dictionary * sparse_coefficient[:, idx_downsample]
            if np.sum(np.square(residual_tmp)) < 1e-3:
                break
            sparse_idx_new = np.argmax(abs(np.dot(residual_tmp.T, noise_response_dictionary)))
            sparse_idx.append(sparse_idx_new)
            sparse_idx = list(set(sparse_idx))
            sparse_coefficient[sparse_idx, idx_downsample] = \
                (np.linalg.pinv(noise_response_dictionary[:, sparse_idx]) *
                 combine_signal_downsample[:, idx_downsample]).T
    return np.array(sparse_coefficient)


def GLM(expected_response, noise_dictionary, combine_signal, sparse_coefficient, idx_upsample, wls_mat, cov_mat,
        icov_mat, contrast_mat, basic_info):
    # Get basic info.
    nframes = basic_info['nframes']
    num_conditions = basic_info['num_conditions']
    num_task_contrast = basic_info['num_task_contrast']
    num_combine_signal_nonzero = basic_info['num_combine_signal_nonzero']

    count_upsample = np.zeros((num_combine_signal_nonzero,))
    zvalue_combine_nonzero = np.zeros((num_task_contrast, num_combine_signal_nonzero))

    nframes_ones = np.ones((nframes, 1))
    idx_lambda_downsample = []

    combine_signal = np.mat(combine_signal)
    for idx_downsample in range(NUM_COMBINE_DOWNSAMPLE_):
        noise_dictionary_tmp = sparse_coefficient[num_conditions:, idx_downsample]
        index_noise = noise_dictionary_tmp != 0
        independent_variable = np.concatenate((expected_response, noise_dictionary[:, index_noise]), axis=1)
        independent_variable = np.concatenate((independent_variable, nframes_ones), axis=1)
        log_likelihood = -np.inf
        upsample_idx_mask = np.sum(idx_upsample == idx_downsample, 1) > 0.5
        for idx_lambda in range(101):
            log_likelihood_old = log_likelihood

            # REML
            independent_variable = np.mat(independent_variable)
            try:
                coefficient_tmp = np.linalg.inv(
                    independent_variable.T * icov_mat[idx_lambda] * independent_variable) * \
                                  independent_variable.T * icov_mat[idx_lambda] * combine_signal[:, upsample_idx_mask]
            except:
                coefficient_tmp = np.linalg.pinv(
                    independent_variable.T * icov_mat[idx_lambda] * independent_variable) * \
                                  independent_variable.T * icov_mat[idx_lambda] * combine_signal[:, upsample_idx_mask]

            residual_tmp = combine_signal[:, upsample_idx_mask] - independent_variable * coefficient_tmp
            noise_variance_tmp = np.sum(np.multiply(icov_mat[idx_lambda].T * residual_tmp, residual_tmp), axis=0) / (
                    nframes - independent_variable.shape[1])

            log_likelihood = -((nframes - independent_variable.shape[1]) * np.sum(np.log(noise_variance_tmp)) + np.sum(
                upsample_idx_mask) * np.log(np.linalg.det(cov_mat[idx_lambda])) + np.sum(upsample_idx_mask) * np.log(
                np.linalg.det(independent_variable.T * icov_mat[idx_lambda] * independent_variable)) + np.sum(
                upsample_idx_mask) * (nframes - independent_variable.shape[1])) / 2.0
            if log_likelihood == np.inf or log_likelihood == np.nan:
                log_likelihood = -np.inf
            if len(idx_lambda_downsample) <= idx_downsample:
                idx_lambda_downsample.append(max(idx_lambda - 1, 1))
            else:
                idx_lambda_downsample[idx_downsample] = max(idx_lambda - 1, 1)
            if log_likelihood <= log_likelihood_old:
                break

        try:
            coefficient = np.linalg.inv(independent_variable.T * icov_mat[
                idx_lambda_downsample[idx_downsample]] * independent_variable) * independent_variable.T * icov_mat[
                              idx_lambda_downsample[idx_downsample]] * combine_signal[:, upsample_idx_mask]
        except:
            coefficient = np.linalg.pinv(independent_variable.T * icov_mat[
                idx_lambda_downsample[idx_downsample]] * independent_variable) * independent_variable.T * icov_mat[
                              idx_lambda_downsample[idx_downsample]] * combine_signal[:, upsample_idx_mask]

        residual = wls_mat[idx_lambda_downsample[idx_downsample]] * combine_signal[:, upsample_idx_mask] - wls_mat[
            idx_lambda_downsample[idx_downsample]] * independent_variable * coefficient
        noise_variance = np.sum(np.square(residual), axis=0) / (nframes - independent_variable.shape[1])
        zvalue_tmp = np.zeros((num_task_contrast, sum(upsample_idx_mask)))
        for idx_task_contrast in range(num_task_contrast):
            task_contrast_extension = np.concatenate(
                (contrast_mat[idx_task_contrast, :], np.zeros((independent_variable.shape[1] - num_conditions), )),
                axis=0)
            task_contrast_extension = np.mat(task_contrast_extension)
            tvalue_tmp = np.divide((task_contrast_extension * coefficient), np.sqrt(
                task_contrast_extension * np.linalg.inv(independent_variable.T * icov_mat[idx_lambda_downsample[
                    idx_downsample]] * independent_variable) * task_contrast_extension.T * noise_variance))
            zvalue_tmp[idx_task_contrast, :] = np.multiply(
                -stats.norm.ppf(stats.t.cdf(-abs(tvalue_tmp), nframes - independent_variable.shape[1])),
                np.sign(tvalue_tmp))

        zvalue_combine_nonzero[:, upsample_idx_mask] += zvalue_tmp
        count_upsample[upsample_idx_mask] = count_upsample[upsample_idx_mask] + 1

    return zvalue_combine_nonzero, count_upsample


@timing_func
def control_upsample(zvalue_combine_nonzero, count_upsample, combine_signal_nonzero_mask, basic_info):
    '''
    Control upsampling and get brain zvalues.
        zvalue_combine_nonzero        -ndarray. Zvalues of nonzero combine signals.
        count_upsample                -
        combine_signal_nonzero_mask   -ndarray. Mask of nonzero signal of combine signal.

    returns
        zvalue_brain
    '''
    # Get basic info.
    num_task_contrast = basic_info['num_task_contrast']
    num_combine_signal = basic_info['num_combine_signal']
    num_nuisance_signals = basic_info['num_nuisance_signals']

    count_upsample = count_upsample.reshape(count_upsample.shape[0], 1)
    zvalue_combine_nonzero = np.divide(zvalue_combine_nonzero,
                                       np.repeat(count_upsample.T,
                                                 num_task_contrast, 0))
    zvalue_combine_nonzero[np.isnan(zvalue_combine_nonzero)] = 0
    zvalue_combine = np.zeros((num_task_contrast, num_combine_signal))
    zvalue_combine[:, combine_signal_nonzero_mask] = zvalue_combine_nonzero
    zvalue_brain = zvalue_combine[:, num_nuisance_signals:]
    zvalue_brain[np.isnan(zvalue_brain)] = 0
    zvalue_nuisance = zvalue_combine[:, :num_nuisance_signals]
    zvalue_nuisance[np.isnan(zvalue_nuisance)] = 0
    return zvalue_brain, zvalue_nuisance


@timing_func
def save_zvalues(zvalue_brain, condition_strs, contrast_mat, basic_info, mask_brain, onerun_paths):
    '''
    Save zvalues of each run in native space and FS1mm space.
        zvalue_brain      -ndarray. Brain zvalues.
        contrast_mat      -ndarray. (num_task_contrast, num_conditions)
        basic_info        -dict. Basic info.
        mask_brain        -ndarray. Mask of brain 3D.
        task_path         -path. $DATA_DIR/'snm_task'
        reg_path          -path. Register files of each bold run.
    '''

    # Get basic info.
    subject = basic_info['subject']
    task_run = basic_info['task_run']
    aff = basic_info['aff']
    hdr = basic_info['hdr']

    reg_path = onerun_paths['reg_path']
    task_path = onerun_paths['task_path']
    mni_path = basic_info['mni_path']
    num_task_contrast = basic_info['num_task_contrast']

    # Save zvalues of each contrast.
    mask_brain_1D = mask_brain.flatten(order='F')
    for idx_task_contrast in range(num_task_contrast):
        contrast_mat_one = contrast_mat[idx_task_contrast]
        condition_one = condition_strs[list(contrast_mat_one).index(1)]
        if -1 in contrast_mat[idx_task_contrast]:
            condition_minus_one = '_' + condition_strs[list(contrast_mat_one).index(-1)]
        else:
            condition_minus_one = ''
        img_brain = np.zeros(mask_brain_1D.shape, np.float16)
        img_brain[mask_brain_1D] = zvalue_brain[idx_task_contrast]
        img_brain_3D = img_brain.reshape(mask_brain.shape, order='F')
        img_brain_3D = img_brain_3D.swapaxes(0, 1)
        img = nib.Nifti1Image(img_brain_3D, aff, hdr)
        dir_path = task_path / task_run
        dir_path.makedirs_p()
        # zvalues_bld_path = dir_path / subject + '_bld' + task_run + '_contrast_' + condition_one + \
        #                    condition_minus_one + '_native_bld.nii.gz'
        zvalues_bld_path = dir_path / subject + '_bld_contrast_' + condition_one + \
                           condition_minus_one + '_native_bld.nii.gz'
        nib.save(img, zvalues_bld_path)

        # Project zvalues to native space surface.
        for hemi in ['lh', 'rh']:
            surf_outpath_tmp = zvalues_bld_path.replace('_native_bld.nii.gz', '_native_surf.mgh')
            native_surf_path = surf_outpath_tmp.replace('%s_' % subject, '%s_%s_' % (hemi, subject))
            shargs = [
                '--trgsubject', subject,
                '--projfrac', 0.5,
                '--reshape',
                '--interp', 'trilinear',
                '--surf-fwhm', 2
            ]
            mri_vol2surf_ng(zvalues_bld_path, native_surf_path, reg_path, hemi, shargs)

            # Project zvalues to fs6 surface.
            fs6_file_path = native_surf_path.replace('_native_surf.mgh', '_fs6_surf.mgh')

            sh.mri_surf2surf(
                '--hemi', hemi,
                '--srcsubject', subject,
                '--sval', native_surf_path,
                '--label-deepprep', hemi + '.cortex.label',
                '--nsmooth-in', 1,
                '--trgsubject', 'fsaverage6',
                '--tval', fs6_file_path,
                '--reshape',
                _out=sys.stdout
            )

            # Project zvalues to fs4 surface.
            fs4_file_path = native_surf_path.replace('_native_surf.mgh', '_fs4_surf.mgh')
            sh.mri_surf2surf(
                '--hemi', hemi,
                '--srcsubject', 'fsaverage6',
                '--sval', fs6_file_path,
                '--label-deepprep', hemi + '.cortex.label',
                '--nsmooth-in', 1,
                '--trgsubject', 'fsaverage4',
                '--tval', fs4_file_path,
                '--reshape',
                _out=sys.stdout
            )

        # Project zvalues to T1 space.
        zvalues_anat_path = zvalues_bld_path.replace('_native_bld.nii.gz', '_native.nii.gz')
        T1_targ = task_path.parent / 'recon' / subject / 'mri/brain.mgz'
        shargs = [
            '--no-save-reg']
        mri_vol2vol_ng(zvalues_bld_path, zvalues_anat_path, T1_targ, reg_path, shargs)

        # Project zvalues to FS1mm space.
        freesurfer_path = path_D(os.environ['FREESURFER_HOME'])
        targ_path = freesurfer_path / 'average/mni305.cor.mgz'
        FS1mm_outpath = zvalues_anat_path.replace('_native.nii.gz', '_FS1mm.nii.gz')

        shargs = [
            '--m3z', 'talairach.m3z',
            '--no-save-reg',
            '--interp', 'trilin']
        mri_vol2vol_ng(zvalues_bld_path, FS1mm_outpath, targ_path, reg_path, shargs)

        # Project zvalues from FS1mm to MNI1mm space.
        input_path = mni_path / 'mri/norm.mgz'
        MNI1mm_outpath = FS1mm_outpath.replace('_FS1mm.nii.gz', '_MNI1mm.nii.gz')

        shargs = [
            '--mov', input_path,
            '--s', 'FSL_MNI152_FS',
            '--targ', FS1mm_outpath,
            '--m3z', 'talairach.m3z',
            '--o', MNI1mm_outpath,
            '--no-save-reg',
            '--inv-morph',
            '--interp', 'trilin']

        sh.mri_vol2vol(*shargs)


def FPR_control(zvalue_uncorr, zvalue_nuisance, num_task_contrast):
    zvalue_corr = zvalue_uncorr.copy()
    for idx_task_contrast in range(num_task_contrast):
        zvalue_nuisance_ascend = sorted(abs(zvalue_nuisance[idx_task_contrast]))
        numel = len(zvalue_nuisance[idx_task_contrast])
        fpr_nuisance = [1 - i / (numel + 1) for i in range(1, numel + 1)]
        zvalue_nuisance_ascend.extend([0, 1e5])
        fpr_nuisance.extend([1, 0])
        _, idx_zvalue_nuisance_ascend_unique = np.unique(zvalue_nuisance_ascend, return_index=True)
        zvalue_nuisance_ascend = np.array(zvalue_nuisance_ascend)
        idx_zvalue_nuisance_ascend_unique = np.array(idx_zvalue_nuisance_ascend_unique)
        fpr_nuisance = np.array(fpr_nuisance)
        f = interpolate.interp1d(zvalue_nuisance_ascend[idx_zvalue_nuisance_ascend_unique],
                                 fpr_nuisance[idx_zvalue_nuisance_ascend_unique])
        fpr_brain = f(abs(zvalue_uncorr[idx_task_contrast]))
        zvalue_corr[idx_task_contrast] = -stats.norm.ppf(fpr_brain / 2) * np.sign(zvalue_uncorr[idx_task_contrast])

    return zvalue_corr


def AlmightyCorrcoefEinsumOptimized(O, P):
    # Fast correlation calculate.
    # Link:https://github.com/ikizhvatov/efficient-columnwise-correlation
    (n, t) = O.shape  # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates
    # compute O - mean(O), compute P - mean(P).
    DO = O - (np.einsum("nt->t", O, optimize='optimal') / np.double(n))
    DP = P - (np.einsum("nm->m", P, optimize='optimal') / np.double(n))

    cov = np.einsum("nm,nt->mt", DP, DO, optimize='optimal')

    varP = np.einsum("nm,nm->m", DP, DP, optimize='optimal')
    varO = np.einsum("nt,nt->t", DO, DO, optimize='optimal')
    tmp = np.einsum("m,t->mt", varP, varO, optimize='optimal')
    tmp[tmp == 0] = np.inf
    return cov / np.sqrt(tmp)


def aggregate_zvalues(subject, task_path, viz_path, task_run_list, conditions_list):
    files_list = []
    for task_run in task_run_list:
        run_path = task_path / task_run
        files_list += run_path.files()
    files_list = sorted(files_list)

    for condition in conditions_list:
        native_name_mark = '_contrast_' + condition + '_native.nii.gz'
        FS1mm_name_mark = native_name_mark.replace('_native.nii.gz', '_FS1mm.nii.gz')
        MNI1mm_name_mark = native_name_mark.replace('_native.nii.gz', '_MNI1mm.nii.gz')
        aggregate_util(subject, task_path, files_list, native_name_mark)
        aggregate_util(subject, task_path, files_list, FS1mm_name_mark)
        aggregate_util(subject, task_path, files_list, MNI1mm_name_mark)

        for surf in ['native', 'fs6', 'fs4']:
            native_surf_name_mark = '_contrast_' + condition + '_%s_surf.mgh' % surf
            aggregate_util_surf(subject, task_path, viz_path, files_list, native_surf_name_mark)

    if len(conditions_list) < 2:
        return

    for condition in conditions_list:
        condition_minus_list = conditions_list[:]
        condition_minus_list.remove(condition)
        for condition_minus in condition_minus_list:
            native_name_mark = '_contrast_' + condition + '_' + condition_minus + '_native.nii.gz'
            FS1mm_name_mark = native_name_mark.replace('_native.nii.gz', '_FS1mm.nii.gz')
            MNI1mm_name_mark = native_name_mark.replace('_native.nii.gz', '_MNI1mm.nii.gz')
            aggregate_util(subject, task_path, files_list, native_name_mark)
            aggregate_util(subject, task_path, files_list, FS1mm_name_mark)
            aggregate_util(subject, task_path, files_list, MNI1mm_name_mark)

            for surf in ['native', 'fs6', 'fs4']:
                native_surf_name_mark = '_contrast_' + condition + '_' + condition_minus + '_%s_surf.mgh' % surf
                aggregate_util_surf(subject, task_path, viz_path, files_list, native_surf_name_mark)


def aggregate_util(subject, task_path, files_list, name_mark):
    '''
    Summarize util.
    If name_mark in file, summarize the files into one.
    '''
    data_all = []
    aff = None
    hdr = None
    for file in files_list:
        if name_mark in file:
            img = nib.load(file)
            data = img.get_data()
            if aff is None:
                aff = img.affine
            if hdr is None:
                hdr = img.header
            data_all.append(data)
    if len(data_all) > 0:
        data_mean = np.mean(data_all, axis=0)
    else:
        return
    out_path = task_path / subject + name_mark
    out_img = nib.Nifti1Image(data_mean, aff, hdr)
    nib.save(out_img, out_path)


def aggregate_util_surf(subject, task_path, viz_path, files_list, name_mark):
    '''
    Summarize surf util.
    If name_mark in file, summarize the files into one.
    '''
    fnames = []
    for hemi in ['lh', 'rh']:
        data_all = []
        aff = None
        hdr = None
        for file in files_list:
            if name_mark in file and '%s_%s_' % (hemi, subject) in file:
                img = nib.load(file)
                data = img.get_data().flatten(order='F')
                if aff is None:
                    aff = img.affine
                if hdr is None:
                    hdr = img.header
                data_all.append(data)

        if len(data_all) > 0:
            data_mean = np.mean(data_all, axis=0)
        else:
            return
        out_path = task_path / hemi + '_' + subject + '_mean' + name_mark
        out_img = nib.MGHImage(data_mean, aff, hdr)
        nib.save(out_img, out_path)

        # Write fs6 / native surf zvalues result to .txt files for front.
        if '_fs6_surf.mgh' in out_path:
            fname = viz_path / (path_A(out_path).name.replace('_fs6_surf.mgh', '_fs6_surf.txt'))
        elif '_native_surf.mgh' in out_path:
            fname = viz_path / (path_A(out_path).name.replace('_native_surf.mgh', '_native_surf.txt'))
        else:
            continue
        with open(fname, 'w') as f:
            f.writelines(['%s\n' % item for item in map(str, data_mean)])
        fnames.append(fname)
    if len(fnames) > 0:
        # GZIP and Fuse fsaverage4 values.
        fnames.append(path_A(fnames[0].replace('lh_%s_mean_contrast' % subject, 'lhrh_%s_mean_contrast' % subject)))
        os.system('cat %s %s > %s' % tuple(fnames))
        for fname in fnames:
            (fname + '.gz').remove_p()
            sh.gzip(fname, _out=sys.stdout)


def prepare_output(subject, common_obj):
    task_path = common_obj['task_path']

    # Make task mean contrast files for download
    new_files = []
    for file in task_path.files():
        if file.endswith('.nii.gz') and '%s_mean_' % subject not in file:
            new_files.append(file)

    for file in new_files:
        sh.cp(file, file.replace('_contrast', '_mean_contrast'))


def write_cluster_metrics(subject, common_obj, conditions_list):
    task_path = common_obj['task_path']
    viz_path = common_obj['viz_path']
    fs6_path = common_obj['fs6_path']
    mni_path = common_obj['mni_path']
    recon_path = common_obj['recon_path']
    aseg_path = common_obj['aseg']
    condition_mark = get_condition_mark(conditions_list, task_path)

    for space in ['native', 'template']:
        if space == 'native':
            surf_path = recon_path
            vol_path = recon_path
        elif space == 'template':
            surf_path = fs6_path
            vol_path = mni_path

        # Get aparc anat parc.
        anat_vol_labels = {}
        anat_resolution = 'aparc'
        anat_labels = get_anat_parc_surf(surf_path, None, anat_resolution)
        anat_vol_labels_aparc = get_anat_parc_vol(vol_path, None, anat_resolution)
        anat_vol_labels[anat_resolution] = anat_vol_labels_aparc

        for condi in condition_mark:
            condi_dir = viz_path / condi
            condi_dir.makedirs_p()

            # Write matrics in native surface, T1 space / fs6 surf and MNI1mm space.
            cluster_mghs = write_cluster_metrics_surf(subject, task_path, viz_path, surf_path, condi_dir, condi, space)
            cluster_files = write_cluster_metrics_vol(subject, task_path, viz_path, aseg_path, condi_dir, condi, space)

            # Write cluster map anat in native surface, T1 space / fs6 surf and MNI1mm space.
            cluster_map_anat(subject, anat_labels, anat_vol_labels, viz_path, cluster_mghs, cluster_files, condi,
                             space)


def write_cluster_metrics_surf(subject, task_path, viz_path, surf_path, condi_dir, condi, space):
    ouput_labeldir = condi_dir / 'labels'
    ouput_labeldir.makedirs_p()
    cluster_mghs = {}
    if space == 'native':
        surf_name_mark = 'native'
        subj = subject
    elif space == 'template':
        surf_name_mark = 'fs6'
        subj = 'fsaverage6'
    for hemi in ['lh', 'rh']:
        mgh_file = task_path / ('%s_%s_mean_contrast_%s_%s_surf.mgh' % (hemi, subject, condi, surf_name_mark))
        sum_file = condi_dir / ('%s_cluster_summary_%s_%s.txt' % (hemi, condi, surf_name_mark))
        cluster_mgh = viz_path / ('%s_%s_%s_labeled_clusters_%s.mgz' % (hemi, subject, condi, surf_name_mark))
        ouput_label = ouput_labeldir / condi
        sh.mri_surfcluster('--in', mgh_file,
                           '--subject', subj,
                           '--hemi', hemi,
                           '--thmin', 1.96,
                           '--thsign', 'abs',
                           '--minarea', 30,
                           '--sum', sum_file,
                           '--ocn', cluster_mgh,
                           '--olab', ouput_label,
                           '--nofixmni')
        cluster_mghs[hemi] = cluster_mgh
        write_cluster_stats(subject, viz_path, surf_path, mgh_file, cluster_mgh, sum_file, hemi, condi, surf_name_mark)
    write_cluster_txt(subject, viz_path, cluster_mghs, condi, surf_name_mark)
    return cluster_mghs


def get_anat_parc_surf(recon_path, viz_parc_path, anat_resolution):
    lhrh_labels = {}
    fnames = []
    for hemi in ['lh', 'rh']:
        if anat_resolution == 'aparc':
            resolution_mark = 'aparc'
        elif anat_resolution == 'a2009s':
            resolution_mark = 'aparc.a2009s'

        label_path = recon_path / 'label' / ('%s.%s.annot' % (hemi, resolution_mark))
        labels, _, names = nib.freesurfer.io.read_annot(label_path)
        labels[labels <= 0] = 0
        lhrh_labels[hemi] = labels

        if viz_parc_path:
            # Write anatomical parc result to .txt files.
            fname = viz_parc_path / ('%s_anat_parc_%s.txt' % (hemi, anat_resolution))
            fnames.append(fname)
            with open(fname, 'w') as f:
                f.writelines(['%s\n' % item for item in map(str, labels)])

    if viz_parc_path:
        # GZIP and Fuse fsaverage4 values.
        fnames.append(viz_parc_path / ('lhrh_anat_parc_%s.txt' % anat_resolution))
        os.system('cat %s %s > %s' % tuple(fnames))
        for fname in fnames:
            (fname + '.gz').remove_p()
            sh.gzip(fname, _out=sys.stdout)
    return lhrh_labels


def get_anat_parc_vol(recon_path, viz_parc_path, anat_resolution):
    if anat_resolution == 'aparc':
        resolution_mark = 'aparc'
        resolution_num = 36
        start_nums = [1000, 2000]
    elif anat_resolution == 'a2009s':
        resolution_mark = 'aparc.a2009s'
        resolution_num = 76
        start_nums = [11100, 12100]

    fpath = recon_path / 'mri' / ('%s+aseg.mgz' % resolution_mark)
    if viz_parc_path:
        foutput = viz_parc_path / ('anat_parc_native_%s_aseg.nii.gz' % anat_resolution)
        sh.mri_convert(fpath, foutput)

    # Get anat parc in volume.
    img = nib.load(fpath)
    labels_asegs = img.get_data()
    labels = np.zeros_like(labels_asegs)
    for start_num in start_nums:
        for i in range(1, resolution_num):
            mask = labels_asegs == start_num + i
            labels[mask] = labels_asegs[mask] - start_num
    return labels.flatten()


def get_condition_mark(conditions_list, task_path):
    files = []
    for file in task_path.files():
        if 'native_surf.mgh' in file.name:
            files.append(file.name)
    condition_mark = []
    for condition in conditions_list:
        condition_mark.append(condition)
    for condition in conditions_list:
        condition_minus_list = conditions_list[:]
        condition_minus_list.remove(condition)
        for condition_minus in condition_minus_list:
            condition_mark_one = condition + '_' + condition_minus
            for file in files:
                if condition_mark_one in file and condition_mark_one not in condition_mark:
                    condition_mark.append(condition_mark_one)
                    break

    return condition_mark


def write_cluster_stats(subject, viz_path, surf_path, mgh_file, cluster_mgh, sum_file, hemi, condi, surf_name_mark):
    lhrh_data = nib.load(mgh_file).get_data().flatten()
    lhrh_labels = nib.load(cluster_mgh).get_data().flatten()
    cluster_output = viz_path / ('%s_%s_%s_cluster_metrics_%s.json' % (hemi, subject, condi, surf_name_mark))
    cluster_stats = read_summary_stats(sum_file, surf_path, hemi)
    cluster_num = len(cluster_stats[0])
    ct_mean, ct_max, ct_min, ct_std = _cortical_thickness_fs6(
        surf_path, lhrh_labels, hemi, cluster_num
    )
    sd_mean, sd_max, sd_min, sd_std = _sulcal_depth_fs6(
        surf_path, lhrh_labels, hemi, cluster_num
    )

    artsd = {
        'max_zvalue': cluster_stats[0],
        'max_vertex_index': cluster_stats[1],
        'cluster_area': cluster_stats[2],
        'Nvtxs': cluster_stats[4],
        'mean_zvalue': _mean_zvalues(lhrh_data, lhrh_labels),
        'mean_cortical_thickness': ct_mean,
        'max_cortical_thickness': ct_max,
        'min_cortical_thickness': ct_min,
        'std_cortical_thickness': ct_std,
        'mean_sulcal_depth': sd_mean,
        'max_sulcal_depth': sd_max,
        'min_sulcal_depth': sd_min,
        'std_sulcal_depth': sd_std
    }
    if surf_name_mark == 'native':
        artsd['peak_native_coords'] = cluster_stats[3]
    elif surf_name_mark == 'fs6':
        artsd['peak_fs6_coords'] = cluster_stats[3]
    # Write output.
    with cluster_output.open('w') as f:
        json.dump(artsd, f)


def write_cluster_stats_vol(subject, viz_path, nii_file, cluster_nii, sum_file, condi, vol_name_mark, hemi, aseg_path):
    vol_data = nib.load(nii_file).get_data().flatten()
    vol_labels = nib.load(cluster_nii).get_data().flatten()
    cluster_output = viz_path / ('vol_%s_%s_cluster_metrics_%s_%s.json' % (subject, condi, vol_name_mark, hemi))
    cluster_stats = read_summary_stats_vol(sum_file, aseg_path)
    artsd = {
        'max_zvalue': cluster_stats[0],
        'mean_zvalue': _mean_zvalues(vol_data, vol_labels),
        'cluster_volume': cluster_stats[2],
        'Nvoxels': cluster_stats[3]
    }
    if vol_name_mark == 'native':
        artsd['peak_T1_coords'] = cluster_stats[1]
    elif vol_name_mark == 'MNI1mm':
        artsd['peak_MNI_coords'] = cluster_stats[1]

    # Write output.
    with cluster_output.open('w') as f:
        json.dump(artsd, f)


def read_summary_stats(sum_file, recon_path, hemi):
    white_surf_path = recon_path / 'surf' / (hemi + '.white')
    vcoordi, _ = nib.freesurfer.io.read_geometry(white_surf_path)
    with open(sum_file, 'r') as f:
        parc_mark = False
        max_value = []
        max_vertex_index = []
        max_vertex_size = []
        coord_geometry = []
        Nvtxs = []
        for line in f:
            num_vals = list(map(str, line.split()))
            if 'ClusterNo' in line:
                parc_mark = True
                continue
            if parc_mark:
                max_value.append(float(num_vals[1]))
                max_vertex_index.append(float(num_vals[2]))
                max_vertex_size.append(float(num_vals[3]))
                coord_geometry.append(list(vcoordi[int(num_vals[2])]))
                Nvtxs.append(float(num_vals[7]))

        return max_value, max_vertex_index, max_vertex_size, coord_geometry, Nvtxs


def read_summary_stats_vol(sum_file, aseg_path):
    vox2ras = np.mat(nib.load(aseg_path).header.get_vox2ras())
    with open(sum_file, 'r') as f:
        parc_mark = False
        max_value = []
        cluster_volume = []
        coords = []
        Nvtxs = []
        for line in f:
            num_vals = list(map(str, line.split()))
            if 'VoxX' in line:
                parc_mark = True
                continue
            if parc_mark:
                max_value.append(float(num_vals[6]))
                coords.append(
                    np.dot(vox2ras, [float(num_vals[3]), float(num_vals[4]), float(num_vals[5]), 1]).tolist()[0][:3])
                cluster_volume.append(float(num_vals[2]))
                Nvtxs.append(float(num_vals[1]))

        return max_value, coords, cluster_volume, Nvtxs


def write_cluster_txt(subject, viz_path, cluster_mghs, condi, surf_name_mark):
    fnames = []
    for hemi in ['lh', 'rh']:
        hemi_labels = nib.load(cluster_mghs[hemi]).get_data().flatten()
        if hemi == 'lh':
            hemi_labels[hemi_labels > 0] = hemi_labels[hemi_labels > 0] + 1000
        elif hemi == 'rh':
            hemi_labels[hemi_labels > 0] = hemi_labels[hemi_labels > 0] + 2000

        # Write anatomical parc result to .txt files.
        fname = viz_path / ('%s_%s_%s_cluster_%s.txt' % (hemi, subject, condi, surf_name_mark))
        fnames.append(fname)
        with open(fname, 'w') as f:
            f.writelines(['%s\n' % item for item in map(str, hemi_labels)])

    # GZIP and Fuse cluster labels.
    fnames.append(viz_path / ('lhrh_%s_%s_cluster_%s.txt' % (subject, condi, surf_name_mark)))
    os.system('cat %s %s > %s' % tuple(fnames))
    for fname in fnames:
        (fname + '.gz').remove_p()
        sh.gzip(fname, _out=sys.stdout)


def _cortical_thickness_fs6(recon_path, lhrh_labels, hemi, resolution_num):
    if hemi in ['lh', 'rh']:
        thicknesses_path = recon_path / 'surf' / (hemi + '.thickness')
        thicknesses = nib.freesurfer.io.read_morph_data(thicknesses_path)

    # Compute mean thickness of each anatomical parc.
    ct_mean = []
    ct_max = []
    ct_min = []
    ct_std = []
    for parc in range(resolution_num):
        thicknesses_vals = []
        if hemi in ['lh', 'rh']:
            i_parc = lhrh_labels == parc
            thicknesses_vals += thicknesses[i_parc].tolist()
        if thicknesses_vals == []:
            ct_mean.append(0)
            ct_max.append(0)
            ct_min.append(0)
            ct_std.append(0)
        else:
            ct_mean.append(np.around(np.mean(thicknesses_vals), decimals=4))
            ct_max.append(np.around(np.max(thicknesses_vals), decimals=4))
            ct_min.append(np.around(np.min(thicknesses_vals), decimals=4))
            ct_std.append(np.around(np.std(thicknesses_vals, ddof=min(1, len(thicknesses_vals) - 1)), decimals=4))

    return ct_mean, ct_max, ct_min, ct_std


def _sulcal_depth_fs6(recon_path, lhrh_labels, hemi, resolution_num):
    if hemi in ['lh', 'rh']:
        sd_path = recon_path / 'surf' / (hemi + '.sulc')
        depths = nib.freesurfer.io.read_morph_data(sd_path)

    # Compute mean thickness of each anatomical parc.
    sd_mean = []
    sd_max = []
    sd_min = []
    sd_std = []
    for parc in range(resolution_num):
        sd_vals = []
        if hemi in ['lh', 'rh']:
            i_parc = lhrh_labels == parc
            sd_vals += depths[i_parc].tolist()
        if sd_vals == []:
            sd_mean.append(0)
            sd_max.append(0)
            sd_min.append(0)
            sd_std.append(0)
        else:
            sd_mean.append(np.around(np.mean(sd_vals), decimals=4))
            sd_max.append(np.around(np.max(sd_vals), decimals=4))
            sd_min.append(np.around(np.min(sd_vals), decimals=4))
            sd_std.append(np.around(np.std(sd_vals, ddof=min(1, len(sd_vals) - 1)), decimals=4))
    return sd_mean, sd_max, sd_min, sd_std


def _mean_zvalues(lhrh_data, lhrh_labels):
    mean_zvalues = []
    for cluster_index in range(int(lhrh_labels.max())):
        mask = lhrh_labels == cluster_index
        mean_zvalues.append(np.mean(lhrh_data[mask]).astype(np.float))
    return mean_zvalues


def write_cluster_metrics_vol(subject, task_path, viz_path, aseg_path, condi_dir, condi, space):
    if space == 'native':
        vol_name_mark = 'native'
    elif space == 'template':
        vol_name_mark = 'MNI1mm'
    ouput_labeldir = condi_dir / 'vol_labels'
    ouput_labeldir.makedirs_p()
    nii_file = task_path / ('%s_mean_contrast_%s_%s.nii.gz' % (subject, condi, vol_name_mark))
    hemi_nii_files = separate_hemi_nii(nii_file, aseg_path, condi_dir, condi)
    cluster_files = {}
    for hemi in ['lh', 'rh']:
        hemi_nii_file = hemi_nii_files[hemi]
        sum_file = condi_dir / ('vol_cluster_summary_%s_%s_%s.txt' % (condi, vol_name_mark, hemi))
        hemi_cluster_nii = viz_path / (
                'vol_%s_%s_labeled_clusters_%s_%s.nii.gz' % (subject, condi, vol_name_mark, hemi))
        cluster_files[hemi] = hemi_cluster_nii
        ouput_label = ouput_labeldir / condi
        sh.mri_volcluster('--in', hemi_nii_file,
                          '--thmin', 1.96,
                          '--sign', 'abs',
                          '--minsize', 150,
                          '--sum', sum_file,
                          '--ocn', hemi_cluster_nii,
                          '--labelbase', ouput_label
                          )
        write_cluster_stats_vol(subject, viz_path, hemi_nii_file, hemi_cluster_nii, sum_file, condi, vol_name_mark,
                                hemi, aseg_path)
    cluster_files['wb'] = merge_cluster_nii(cluster_files)
    return cluster_files


def separate_hemi_nii(nii_file, aseg_path, condi_dir, condi):
    nii_img = nib.load(nii_file)
    nii_data = nii_img.get_data()
    aff = nii_img.affine
    hdr = nii_img.header
    aseg_mask = nib.load(aseg_path).get_data()
    hemi_nii_files = {}
    for hemi in ['lh', 'rh']:
        hemi_mask_init = aseg_mask.copy()
        if hemi == 'lh':
            hemi_mask = np.logical_or(hemi_mask_init == 2, hemi_mask_init == 3)
        elif hemi == 'rh':
            hemi_mask = np.logical_or(hemi_mask_init == 41, hemi_mask_init == 42)
        hemi_data = np.zeros_like(nii_data)
        hemi_data[hemi_mask] = nii_data[hemi_mask]
        hemi_nii_name = condi_dir / ('%s_%s.nii.gz' % (hemi, condi))
        hemi_nii_files[hemi] = hemi_nii_name
        hemi_img = nib.Nifti1Image(hemi_data, aff, hdr)
        nib.save(hemi_img, hemi_nii_name)
    return hemi_nii_files


def merge_cluster_nii(hemi_cluster_files):
    cluster_nii = hemi_cluster_files['lh'].replace('_lh.nii.gz', '.nii.gz')
    for hemi in ['lh', 'rh']:
        hemi_img = nib.load(hemi_cluster_files[hemi])
        hemi_data = hemi_img.get_data()
        if hemi == 'lh':
            aff = hemi_img.affine
            hdr = hemi_img.header
            hemi_data[hemi_data > 0] = hemi_data[hemi_data > 0] + 1000
            cluster_data = hemi_data
        elif hemi == 'rh':
            hemi_data[hemi_data > 0] = hemi_data[hemi_data > 0] + 2000
            cluster_data += hemi_data
    wb_img = nib.Nifti1Image(cluster_data, aff, hdr)
    nib.save(wb_img, cluster_nii)
    return cluster_nii


def cluster_map_anat(subject, anat_labels, anat_vol_labels, viz_path, cluster_mghs, hemi_cluster_files, condi, space):
    cluster_map_anat_surf_parc(subject, anat_labels, viz_path, cluster_mghs, condi, space)
    cluster_map_anat_vol_parc(subject, anat_vol_labels, viz_path, hemi_cluster_files, condi, space)


def cluster_map_anat_surf_parc(subject, anat_labels, viz_path, cluster_mghs, condi, space):
    cluster_lhrh = {}
    labels_lhrh = {}
    for hemi in ['lh', 'rh']:
        nets = nib.load(cluster_mghs[hemi]).get_data().flatten()
        labels = anat_labels[hemi]
        cluster_lhrh[hemi] = nets
        labels_lhrh[hemi] = labels
    cluster_map_anat_surf(subject, cluster_lhrh, labels_lhrh, viz_path, condi, space)


def cluster_map_anat_vol_parc(subject, anat_vol_labels, viz_path, hemi_cluster_files, condi, space):
    vol_anat_resolution = 'aparc'
    anat_vol_label_one = anat_vol_labels[vol_anat_resolution]

    for hemi in ['lh', 'rh', 'wb']:
        cluster_nii = hemi_cluster_files[hemi]
        cluster_parc = nib.load(cluster_nii).get_data().flatten()
        cluster_map_anat_vol(subject, cluster_parc, anat_vol_label_one, viz_path, hemi, condi, vol_anat_resolution,
                             space)


def cluster_map_anat_surf(subject, nets_lhrh, labels_lhrh, viz_parc_path, condi, space):
    resolution_num = 36
    counts_lhrh = {}
    for hemi in ['lh', 'rh']:
        counts = []
        nets = nets_lhrh[hemi]
        labels = labels_lhrh[hemi]
        if np.max(nets) == 0:
            counts_lhrh[hemi] = [np.zeros((resolution_num), np.int)]
            continue
        for i_net in range(1, int(np.max(nets) + 1)):
            count_init = np.zeros((resolution_num), np.int)
            func_one = nets == i_net
            if not func_one.max():
                counts.append(count_init)
            else:
                count = np.bincount(labels[func_one])
                count_init[:count.shape[0]] = count
                counts.append(count_init)
        counts_lhrh[hemi] = counts
    counts_lhrh['lhrh'] = counts_lhrh['lh'] + counts_lhrh['rh']

    # Write map to .txt files
    fnames = []
    for hemi in ['lh', 'rh', 'lhrh']:
        fname = viz_parc_path / ('%s_%s_%s_cluster_map_anat_%s.txt' % (hemi, subject, condi, space))
        fnames.append(fname)
        counts = counts_lhrh[hemi]
        with open(fname, 'w') as f:
            for count in counts:
                for item in count:
                    f.write(str(item) + '\t')
                f.write('\n')
    for fname in fnames:
        (fname + '.gz').remove_p()
        sh.gzip(fname, _out=sys.stdout)


def cluster_map_anat_vol(subject, nets, anat_vol_label_one, viz_path, hemi, condi, vol_anat_resolution, space):
    if vol_anat_resolution == 'aparc':
        resolution_num = 36
    if hemi in ['lh', 'rh']:
        counts = []
        for i_net in range(1, int(np.max(nets) + 1)):
            count_init = np.zeros((resolution_num), np.int)
            func_one = nets == i_net
            if not func_one.max():
                counts.append(count_init)
            else:
                count = np.bincount(anat_vol_label_one[func_one])
                count_init[:count.shape[0]] = count
                counts.append(count_init)
        # Write map to .txt files
        fname = viz_path / (
                'vol_%s_%s_cluster_map_anat_%s_%s_%s.txt' % (subject, condi, space, vol_anat_resolution, hemi))
        with open(fname, 'w') as f:
            for count in counts:
                for item in count:
                    f.write(str(item) + '\t')
                f.write('\n')
    elif hemi == 'wb':
        counts = []
        exist_clusters = np.unique(nets)
        for i_net in exist_clusters:
            count_init = np.zeros((resolution_num), np.int)
            func_one = nets == i_net
            if not func_one.max():
                counts.append(count_init)
            else:
                count = np.bincount(anat_vol_label_one[func_one])
                count_init[:count.shape[0]] = count
                counts.append(count_init)

        fname = viz_path / ('vol_%s_%s_cluster_map_anat_%s_%s.txt' % (subject, condi, space, vol_anat_resolution))
        with open(fname, 'w') as f:
            for exist_cluster, count in zip(exist_clusters, counts):
                f.write(str(exist_cluster) + '\t')
                for item in count:
                    f.write(str(item) + '\t')
                f.write('\n')

    (fname + '.gz').remove_p()
    sh.gzip(fname, _out=sys.stdout)


def preprocess_task(data_path, subj):
    preprocess_common(data_path, subj)


def detect_niftis(subject, tmp_path):
    run_regex = re.compile(r'^\d+$')
    anat_regex = re.compile(r'^anat\/.*_mpr[0-9]{3}.nii(\.gz)?$')
    bold_regex = re.compile(r'^bold\/.*_bld[0-9]{3,4}_rest.nii(\.gz)?$')
    anat_t2_regex = re.compile(r'^anat-t2\/.*_mpr[0-9]{3}_t2w.nii(\.gz)?$')
    for dr in tmp_path.walkdirs():
        if dr.name not in ['anat', 'bold', 'anat-t2']:
            continue
        for rundir in sorted(dr.dirs()):
            if not run_regex.match(rundir.name):
                continue

            for f in rundir.walkfiles():
                f_rel = f.relpath(dr.parent)
                number = f_rel.parent.name
                if ((not bold_regex.match(f_rel)) or number not in f_rel.name) and (f_rel.startswith('bold/')):
                    rename_file_to_standard(f, subject, f_rel, '_rest.nii', '_rest.nii.gz', '_bld')

                if ((not anat_regex.match(f_rel)) or number not in f_rel.name) and (f_rel.startswith('anat/')):
                    rename_file_to_standard(f, subject, f_rel, '.nii', '.nii.gz', '_mpr')

                if ((not anat_t2_regex.match(f_rel)) or number not in f_rel.name) and (f_rel.startswith('anat-t2/')):
                    rename_file_to_standard(f, subject, f_rel, '_t2w.nii', '_t2w.nii.gz', '_mpr')


def rename_file_to_standard(src_path, subject, rel_path, nii_name, gz_name, middle_name):
    file_type = ""
    if rel_path.endswith('nii'):
        file_type = nii_name
    elif rel_path.endswith('nii.gz'):
        file_type = gz_name
    if "" != file_type:
        run_num = rel_path.parent.basename()
        f_rel_new_name = subject + middle_name + run_num + file_type
        f_renamed = src_path.parent / f_rel_new_name
        src_path.rename(f_renamed)


if __name__ == '__main__':
    set_envrion()
    # data_path = Path('data').absolute()
    # data_path = path_D('/home/zhenyu/workdata/App/MSC_app/sub-MSC01')
    # subj = 'MSC01_func01'


    NUM_NOISE_DICTIONARY_ = 500
    NUM_MAX_ITER_ = 100
    NUM_TRAIN_NOISE_ = 10
    NUM_COMBINE_DOWNSAMPLE_ = 1000
    NUM_BRAIN_NOISE_ = 100
    NUM_REPEAT_DENOISE_ = 3


    sub_list_file = '/mnt/ngshare/DeepPrep/MSC/derivatives/list.txt'
    sess_list_file = '/mnt/ngshare/DeepPrep/MSC/derivatives/sess_list.txt'

    sub_list = []
    with open(sub_list_file) as f:
        for line in f:
            sub_list.append(line.strip())
    sess_list = []
    with open(sess_list_file) as f:
        for line in f:
            sess_list.append(line.strip())

    args_list = []
    data_dir = path_D('/mnt/ngshare2/App/MSC_app')
    for sub in sub_list:
        sub_path = data_dir / sub
        for ses in sess_list:
            new_sub = sub.split('-')[1] + '_' + ses.split('-')[1]
            subj = new_sub
            preprocess(sub_path, subj)
    #         args_list.append([sub_path, subj])


    # pool = Pool(10)
    # pool.starmap(discover_upload_step, args_list)
    # pool.starmap(preprocess, args_list)
    # pool.close()
    # pool.join()



    # discover_upload_step(data_path, subj)
    # data_path = path_D('/mnt/ngshare2/App/MSC_app/sub-MSC01')
    # subj = 'MSC01_func01'
    # preprocess(data_path, subj)
    # res_proj(data_path, subj)
    # snm_task(path_A(data_path), subj)

