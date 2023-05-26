import os
import sys
import glob
import sh
import nibabel as nb
from path import Path
import math
import numpy as np
from ..utils.utils import timing_func

FRAMES_PER_PART = 40
SUB_FRAMES = 3
# CLOSE_NUM must larger than one
CLOSE_NUM = 2
SUB_RUNS = 2
# CLOSE_RUN_NUM must larger than one
CLOSE_RUN_NUM = 2


@timing_func
def T1_to_templates(data_path, subject, outdir_path):
    '''
    Wrapper of T1_to_FSNL and T1_to_MNI152.
    '''

    T1_to_FSNL(data_path, subject, outdir_path)
    T1_to_MNI152(outdir_path)


def T1_to_FSNL(data_path, subject, outdir_path):
    '''
    Project T1 into Freesurfer nonlinear volumetric space for checking purpose.

    subject     - str.
    outdir_path - Path.

    Files created in outdir_path:
    norm_fsaverage_space1mm.nii.gz
    norm_fsaverage_space2mm.nii.gz
    '''

    datadir_path = data_path / subject
    freesurfer_path = Path(os.environ['FREESURFER_HOME'])
    recon_path = datadir_path / 'recon'
    input_path = recon_path / subject / 'mri/norm.mgz'
    output_path = outdir_path / 'norm_fsaverage_space1mm.nii.gz'
    targ_path = freesurfer_path / 'average/mni305.cor.mgz'
    sh.mri_vol2vol(
        '--mov', input_path,
        '--s', subject,
        '--targ', targ_path,
        '--m3z', 'talairach.m3z',
        '--o', output_path,
        '--no-save-reg',
        '--interp', 'trilin',
        _out=sys.stdout
    )

    input_path = output_path
    output_path = outdir_path / 'norm_FSnonlinear_2mm.nii.gz'
    targdir_path = Path(__file__).parent.parent / 'resources' / 'FS_nonlinear_volumetric_space_4.5'
    targ_path = targdir_path / 'gca_mean2mm.nii.gz'
    sh.mri_vol2vol(
        '--mov', input_path,
        '--s', subject,
        '--targ', targ_path,
        '--o', output_path,
        '--regheader',
        '--no-save-reg',
        _out=sys.stdout
    )


def T1_to_MNI152(data_path):
    '''
    Project T1 to MNI152 space for checking purpose.

    data_path   - Path. Directory containing 'norm_fsaverage_space1mm.nii.gz',
                  as produced via T1_to_FSNL().

    Files created in data_path:
    norm_MNI152_1mm.nii.gz
    norm_MNI152_2mm.nii.gz
    '''

    inputdir_path = Path(__file__).parent.parent / 'resources' / 'FSL_MNI152_FS4.5.0' / 'mri'
    input_path = inputdir_path / 'norm.mgz'
    targ_path = data_path / 'norm_fsaverage_space1mm.nii.gz'
    output_path = data_path / 'norm_MNI152_1mm.nii.gz'
    sh.mri_vol2vol(
        '--mov', input_path,
        '--s', 'FSL_MNI152_FS',
        '--targ', targ_path,
        '--m3z', 'talairach.m3z',
        '--o', output_path,
        '--no-save-reg',
        '--inv-morph',
        '--interp', 'trilin',
        _out=sys.stdout
    )

    input_path = output_path
    targ_path = inputdir_path / 'norm2mm.nii.gz'
    output_path = data_path / 'norm_MNI152_2mm.nii.gz'
    sh.mri_vol2vol(
        '--mov', input_path,
        '--s', 'FSL_MNI152_FS',
        '--targ', targ_path,
        '--o', output_path,
        '--regheader',
        '--no-save-reg',
        _out=sys.stdout
    )


@timing_func
def indi_to_FS2mm(subject, fpath, reg_fpath, bld_run, indi_path):
    '''
    Project fMRI to Freesurfer nonliner volumetric space at
    1mm resolution.

    subject     - str. Subject ID.
    fpath       - Path. Path to fMRI file to project.
    reg_fpath   - Path. Path to fMRI registration .dat file.
    bld_run     - str. Number of bold run.
    indi_path   - str. Path to fMRI file to project.

    Creates:
    Folder. $DATA_DIR/preprocess/<subject>/residuals/<bld_run>+'_indi'
           include many '<subject>_bld<bld_run>_...resid_part%d.nii.gz' files

           part_num: the number of files in this folder is
           int(math.ceil(resid.shape[3])/FRAMES_PER_PART)

    Folder. $DATA_DIR/preprocess/<subject>/residuals/<bld_run> + '_FS2mm'
           include many
           '<subject>_bld<bld_run>_...resid_FS1mm_FS2mm_part%d.nii.gz' files

           the number of files is  part_num too

    return: part_num
    '''
    freesurfer_path = Path(os.environ['FREESURFER_HOME'])
    targ_path = freesurfer_path / 'average/mni305.cor.mgz'

    targdir_path = Path(__file__).parent.parent / 'resources' / 'FS_nonlinear_volumetric_space_4.5'
    targ_path_FS2 = targdir_path / 'gca_mean2mm.nii.gz'

    resid_path = fpath.parent

    # BOLD to FS2 by linear transformation
    FS2mm_dirpath = resid_path / (bld_run + '_FS2mm')
    FS2mm_dirpath.mkdir(exist_ok=True)
    list_FS2_files = []

    for file in glob.glob(str(indi_path / '*')):
        output_file = file.replace('resid_', 'resid_FS1mm_FS2mm_untrans_')
        output_file = output_file.replace(bld_run + '_indi', bld_run + '_FS2mm')
        list_FS2_files.append(output_file)

        shargs = [
            '--mov', file,
            '--s', subject,
            '--targ', targ_path_FS2,
            '--reg', reg_fpath,
            '--o', output_file,
            '--no-save-reg']

        mri_vol2vol_py(shargs)

    # FS2_untrans to FS2 by unlinear transformation
    for file in list_FS2_files:
        output_file = file.replace('resid_FS1mm_FS2mm_untrans_', 'resid_FS1mm_FS2mm_')
        shargs = [
            '--mov', file,
            '--s', subject,
            '--targ', targ_path,
            '--m3z', 'talairach.m3z',
            '--o', output_file,
            '--no-save-reg',
            '--interp', 'trilin']

        mri_vol2vol_ext_py(shargs)
        os.remove(file)


@timing_func
def mri_vol2vol_py(shargs):
    sh.mri_vol2vol(*shargs, _out=sys.stdout)


@timing_func
def mri_vol2vol_ext_py(shargs):
    sh.mri_vol2vol_ext(*shargs, _out=sys.stdout)


@timing_func
def get_brain_mask_func_2mm(data_path, subject, fpath, bld_run):
    freesurfer_path = Path(os.environ['FREESURFER_HOME'])
    targ_path = freesurfer_path / 'average/mni305.cor.mgz'

    # anat_native-->FS_1mm
    recon_path = data_path / subject / 'recon' / subject
    brain_mask_anat = recon_path / 'mri/brainmask.mgz'
    brain_mask_func_fs1mm = fpath.parent / 'brain_mask_func_fs1mm.nii.gz'
    shargs = [
        '--mov', targ_path,
        '--s', subject,
        '--targ', brain_mask_anat,
        '--m3z', 'talairach.m3z',
        '--inv',
        '--o', brain_mask_func_fs1mm,
        '--no-save-reg',
        '--interp', 'trilin']
    sh.mri_vol2vol(*shargs, _out=sys.stdout)

    # FS_1mm-->FS_2mm
    targdir_path = Path(__file__).parent.parent / 'resources' / 'FS_nonlinear_volumetric_space_4.5'
    targ_path = targdir_path / 'gca_mean2mm.nii.gz'
    brain_mask_func_fs1mm_fs2mm = str(brain_mask_func_fs1mm).replace('fs1mm', 'fs1mm_fs2mm')
    shargs = [
        '--mov', brain_mask_func_fs1mm,
        '--s', subject,
        '--targ', targ_path,
        '--o', brain_mask_func_fs1mm_fs2mm,
        '--regheader',
        '--no-save-reg']
    sh.mri_vol2vol(*shargs, _out=sys.stdout)

    # FS_1mm-->MNI_1mm
    inputdir_path = Path(__file__).parent.parent / 'resources' / 'FSL_MNI152_FS4.5.0' / 'mri'
    input_path = inputdir_path / 'norm.mgz'
    brain_mask_func_mni1mm = str(brain_mask_func_fs1mm).replace('fs1mm', 'mni1mm')
    shargs = [
        '--mov', input_path,
        '--s', 'FSL_MNI152_FS',
        '--targ', brain_mask_func_fs1mm,
        '--m3z', 'talairach.m3z',
        '--o', brain_mask_func_mni1mm,
        '--no-save-reg',
        '--inv-morph',
        '--interp', 'trilin']
    sh.mri_vol2vol(*shargs, _out=sys.stdout)

    # MNI_1mm-->MNI_2mm
    targdir_path = Path(__file__).parent.parent / 'resources' / 'FSL_MNI152_FS4.5.0' / 'mri'
    targ_path = targdir_path / 'norm2mm.nii.gz'
    brain_mask_func_mni1mm_mni2mm = str(brain_mask_func_mni1mm).replace('mni1mm', 'mni1mm_mni2mm')
    shargs = [
        '--mov', brain_mask_func_mni1mm,
        '--s', 'FSL_MNI152_FS',
        '--targ', targ_path,
        '--o', brain_mask_func_mni1mm_mni2mm,
        '--regheader',
        '--no-save-reg']
    sh.mri_vol2vol(*shargs, _out=sys.stdout)

    # get binary brain mask in fs2mm and mni2mm
    brain_mask_func_fs1mm_fs2mm_binary_path = \
        convert_mask_to_binary(brain_mask_func_fs1mm_fs2mm, 'fs1mm_fs2mm', 'fs1mm_fs2mm_binary')
    brain_mask_func_mni1mm_mni2mm_binary_path = \
        convert_mask_to_binary(brain_mask_func_mni1mm_mni2mm, 'mni1mm_mni2mm', 'mni1mm_mni2mm_binary')

    return brain_mask_func_fs1mm_fs2mm_binary_path, brain_mask_func_mni1mm_mni2mm_binary_path


def convert_mask_to_binary(brain_mask_multi_value, src_name, dst_name):
    brain_mask_multi_value_img = nb.load(brain_mask_multi_value)
    brain_mask_multi_value_data = brain_mask_multi_value_img.get_data()
    affine = brain_mask_multi_value_img.affine
    header = brain_mask_multi_value_img.header
    brain_mask_multi_value_data = brain_mask_multi_value_data > 0
    brain_mask_binary_img = nb.Nifti1Image(brain_mask_multi_value_data, affine, header)
    brain_mask_binary_path = str(brain_mask_multi_value).replace(src_name, dst_name)
    nb.save(brain_mask_binary_img, brain_mask_binary_path)

    return brain_mask_binary_path


def mri_fwhm_ng(input_path, output_path, other_shargs=None):
    shargs = [
        '--i', input_path,
        '--o', output_path]
    if other_shargs is not None:
        shargs += other_shargs
    sh.mri_fwhm(*shargs)


def mri_fwhm(without_smooth_path, fpath, brain_mask_binary):
    '''
    Use 6mm gauss filter to smooth the image.

    without_smooth_path  -path. Path of nii.gz before smooth
    fpath -path. Path of nii.gz after smooth.
    brain_mask_binary -path. limited area
    '''
    shargs = [
        '--fwhm', '6',
        '--mask', brain_mask_binary]
    mri_fwhm_ng(without_smooth_path, fpath, shargs)

@timing_func
def sub_nii(fpath, resid_path, bld_run):
    '''
    Divided the resid file to many sub files.

    fpath      -path.Resid file path .
    tmp_path   -path.Folder parent path.
    bld_run    -str.Number of bold run.

    return:
    indi_path  -path.Path of divided files of resid.
    part_num   -int.Number of divided files.
    '''

    _, f_name = os.path.split(fpath)
    indi_path = resid_path / (bld_run + '_indi')
    run = nb.load(str(fpath))
    run_data = run.get_data()

    hdr = run.header
    aff = run.affine

    indi_path.mkdir(exist_ok=True)
    each = FRAMES_PER_PART
    part_num = int(math.ceil(run_data.shape[3] * 1.0 / FRAMES_PER_PART))
    for i in range(part_num):
        run_one = run_data[:, :, :, int(i * each):int((i + 1) * each)]
        new_image = nb.Nifti1Image(run_one, aff, hdr)
        output_path = f_name.replace('.nii.gz', '_part%d.nii.gz' % i)
        nb.save(new_image, str(indi_path / output_path))

    return indi_path, part_num


@timing_func
def group_nii(dir_path, output_path, part_num, brain_mask_binary):
    '''
    Gruop many sub files to one.

    dir_path      -path. Dir path of sub files.
    output_path   -path. Path the outfile.
    part_num      -int. Number of divided files.

    '''
    aff = None
    hdr = None
    sub_files_num = int(math.ceil(part_num / (SUB_FRAMES * 1.0)))
    last_frames = part_num % SUB_FRAMES
    if last_frames < (SUB_FRAMES / 2) and last_frames != 0:
        sub_files_num -= 1
    _, f_name = os.path.split(output_path)
    sub_frames_name = f_name.replace('_sm6.nii.gz', '_subframes')

    # Make many sub_frames, each sub_frames includes 60 frames.
    sub_num = 0
    sub_frames = 0
    nii_one = None
    data_all = None
    for part_id in range(part_num):
        part_name = f_name.replace('_sm6.', '_part%d.' % part_id)
        part_path = dir_path / str(part_name)

        nii_one = nb.load(str(part_path))
        data_one = nii_one.get_data()
        # nibabel will save ((128,128,128,1))->((128,128,128))
        # so we need to manually add a dimension
        if len(data_one.shape) == 3:
            data_one = data_one.reshape((
                data_one.shape[0], data_one.shape[1], data_one.shape[2], 1))
        if sub_num == 0:
            data_all = data_one
        else:
            data_all = np.concatenate((data_all, data_one), axis=3)
        # part_path.remove()
        sub_num += 1

        # part_num-1 means the last nii in the folder.
        if sub_num >= SUB_FRAMES and (part_num - 1) - part_id >= CLOSE_NUM or part_id == part_num - 1:
            if aff is None:
                aff = nii_one.affine
            if hdr is None:
                hdr = nii_one.header

            sub_frame_img = nb.Nifti1Image(data_all, aff, hdr)
            one_name = sub_frames_name + '_%d.nii.gz' % sub_frames
            one_path = dir_path / str(one_name)
            nb.save(sub_frame_img, str(one_path))

            # Smooth each sub_frames use 6mm gauss filter.
            smooth_path = str(one_path).replace('_subframes', '_subframes_sm6_all')
            mri_fwhm(one_path, Path(smooth_path), brain_mask_binary)
            # one_path.remove()

            # limit the smooth in the aea of brain
            residual_smoothed_img = nb.load(smooth_path)
            residual_smoothed_data = residual_smoothed_img.get_data()
            brain_mask_binary_img = nb.load(brain_mask_binary)
            brain_mask_binary_data = brain_mask_binary_img.get_data()
            brain_mask_3d = brain_mask_binary_data == 0
            for n in range(residual_smoothed_data.shape[3]):
                residual_smoothed_data[:, :, :, n][brain_mask_3d] = 0
            residual_smoothed_in_brain_img = nb.Nifti1Image(residual_smoothed_data, aff, hdr)
            residual_smoothed_in_brain_path = str(smooth_path).replace('_subframes_sm6_all', '_subframes_sm6')
            nb.save(residual_smoothed_in_brain_img, str(residual_smoothed_in_brain_path))

            sub_num = 0
            sub_frames += 1

    # Group each sub_frames into one nii.gz file.
    concat_args = []
    sub_run = 0
    for sub_frame in range(sub_frames):
        sub_name = f_name.replace('_sm6.', '_subframes_sm6_%d.' % sub_frame)
        sub_path = dir_path / str(sub_name)
        concat_args.append(sub_path)

        if len(concat_args) >= SUB_RUNS and abs(sub_frame - (sub_frames - 1)) >= \
                CLOSE_RUN_NUM or sub_frame == sub_frames - 1:
            sub_output_path = str(output_path).replace('2mm_sm6', '2mm_sm6_subrun%d' % sub_run)
            concat_args = concat_args + ['--o', sub_output_path]
            concat_args.insert(0, '--i')
            sh.mri_concat(*concat_args, _out=sys.stdout)
            sub_run += 1
            concat_args = []


@timing_func
def indi_FS2mm_to_MNI2mm(fpath_mni2, part_num, bld_run, brain_mask_binary):
    '''
    Downsample fMRI from MNI152 1 mm resolution to 2 mm resolution.

    fpath_mni2       - Path. Path to file to downsample.
    part_num    - int. Number of files in each new folder.
    bld_run     - str. Number of bold run.
    brain_mask_binary - str. Path to file to mask.

    Creates:
    Folder. $DATA_DIR/preprocess/<subject>/residuals/<bld_run>+'_MNI2mm'
           include many '<subject>_bld<bld_run>_...
           resid_FS1mm_MNI1mm_MNI2mm_part%d.nii.gz' files
           --the number of files is  part_num too

    File. In $DATA_DIR/preprocess/<subject>/residuals/
      <subject>_bld<bld_run>_..._resid_FS1mm_MNI1mm_MNI2mm.nii.gz
    '''

    inputdir_path = Path(__file__).parent.parent / 'resources' / 'FSL_MNI152_FS4.5.0' / 'mri'
    input_path_MNI1 = inputdir_path / 'norm.mgz'

    targ_path_MNI2 = inputdir_path / 'norm2mm.nii.gz'

    FS2mm_dirpath = fpath_mni2.parent / (bld_run + '_FS2mm')
    MNI2mm_dirpath = fpath_mni2.parent / (bld_run + '_MNI2mm')
    MNI2mm_dirpath.mkdir(exist_ok=True)

    # FS2 to NMI2_untrans by linear transformation
    list_MNI2_files = []
    for file in glob.glob(str(FS2mm_dirpath / '*')):
        output_file = file.replace('FS2mm_', 'MNI1mm_MNI2mm_untrans_')
        output_file = output_file.replace(bld_run + '_FS2mm', bld_run + '_MNI2mm')
        list_MNI2_files.append(output_file)
        shargs = [
            '--mov', file,
            '--s', 'FSL_MNI152_FS',
            '--targ', targ_path_MNI2,
            '--o', output_file,
            '--no-save-reg']

        mri_vol2vol_py(shargs)

    # NMI2_untrans to NMI2 by unlinear transformation
    for file in list_MNI2_files:
        output_file = file.replace('MNI1mm_MNI2mm_untrans_', 'MNI1mm_MNI2mm_')

        shargs = [
            '--mov', input_path_MNI1,
            '--s', 'FSL_MNI152_FS',
            '--targ', file,
            '--m3z', 'talairach.m3z',
            '--o', output_file,
            '--no-save-reg',
            '--inv-morph',
            '--interp', 'trilin']

        mri_vol2vol_ext_py(shargs)
        os.remove(file)

    smooth_path = Path(str(fpath_mni2).replace('_MNI2mm', '_MNI2mm_sm6'))
    group_nii(MNI2mm_dirpath, smooth_path, part_num, brain_mask_binary)


def fs2mm_to_t12mm(subject, T1_path, fs2mm_path, t12mm_path):
    template_fs1mm_path = fs2mm_path.replace('_fs2mm.nii.gz', '_fs1mm.nii.gz')

    # FS2mm-->FS1mm
    freesurfer_path = Path(os.environ['FREESURFER_HOME'])
    fs1mm_target_path = freesurfer_path / 'average/mni305.cor.mgz'
    sh.mri_vol2vol(
        '--mov', fs2mm_path,
        '--s', subject,
        '--targ', fs1mm_target_path,
        '--o', template_fs1mm_path,
        '--regheader',
        '--no-save-reg',
        '--interp', 'nearest',
        _out=sys.stdout)

    # FS1mm-->T1_native(1mm)
    template_t1_native_path = template_fs1mm_path.replace('_fs1mm.nii.gz', '_t1_native.nii.gz')
    sh.mri_vol2vol(
        '--mov', T1_path,
        '--s', subject,
        '--targ', template_fs1mm_path,
        '--m3z', 'talairach.m3z',
        '--o', template_t1_native_path,
        '--inv-morph',
        '--no-save-reg',
        '--interp', 'nearest',
        _out=sys.stdout)

    # T1_native-->T12mm
    sh.mri_convert(template_t1_native_path, t12mm_path, '-ds', [2, 2, 2], '-rt', 'nearest')


def fs2mm_to_t1(subject, T1_path, fs2mm_path, t1_path):
    # Make tmp dir.
    root_path = fs2mm_path.parent / 'tmp'
    root_path.makedirs_p()

    # FS2mm -> FS1mm.
    template_fs1mm_path = root_path / 'fs1mm.nii.gz'
    fs2mm_to_fs1mm(subject, fs2mm_path, template_fs1mm_path)

    # FS1mm -> t1.
    template_t1_native_path = template_fs1mm_path.parent / 't1_native.nii.gz'
    fs1mm_to_t1(subject, T1_path, template_fs1mm_path, template_t1_native_path)

    # Output file.
    sh.cp(template_t1_native_path, t1_path)

    # Clean tmp dir.
    root_path.rmtree_p()


def fs2mm_to_fs1mm(subject, fs2mm_path, template_fs1mm_path):
    # FS2mm-->FS1mm
    freesurfer_path = Path(os.environ['FREESURFER_HOME'])
    fs1mm_target_path = freesurfer_path / 'subjects/fsaverage/mri/T1.mgz'
    sh.mri_vol2vol(
        '--mov', fs2mm_path,
        '--s', subject,
        '--targ', fs1mm_target_path,
        '--o', template_fs1mm_path,
        '--regheader',
        '--no-save-reg',
        '--interp', 'trilin',
        _out=sys.stdout)


def fs1mm_to_t1(subject, T1_path, template_fs1mm_path, template_t1_native_path):
    sh.mri_vol2vol(
        '--mov', T1_path,
        '--s', subject,
        '--targ', template_fs1mm_path,
        '--m3z', 'talairach.m3z',
        '--o', template_t1_native_path,
        '--inv-morph',
        '--no-save-reg',
        '--interp', 'trilin',
        _out=sys.stdout)


def fs2mm_to_mni2mm(abnorm_fs2mm, output_abnorm_mni2mm):
    # Make tmp dir.
    root_path = abnorm_fs2mm.parent / 'tmp'
    root_path.makedirs_p()
    input_path_MNI1 = CODE_DIR() / 'templates/volume/FSL_MNI152_FS4.5.0/mri/norm.mgz'
    targ_path_MNI2 = CODE_DIR() / 'templates/volume/FSL_MNI152_FS4.5.0/mri/norm2mm.nii.gz'

    # Make untrans file.
    tmp_file = root_path / 'tmp_untrans.nii.gz'
    shargs = [
        '--mov', abnorm_fs2mm,
        '--s', 'FSL_MNI152_FS',
        '--targ', targ_path_MNI2,
        '--o', tmp_file,
        '--no-save-reg']

    mri_vol2vol_py(shargs)

    # Make MNI2mm file.
    shargs = [
        '--mov', input_path_MNI1,
        '--s', 'FSL_MNI152_FS',
        '--targ', tmp_file,
        '--m3z', 'talairach.m3z',
        '--o', output_abnorm_mni2mm,
        '--no-save-reg',
        '--inv-morph',
        '--interp', 'trilin']

    mri_vol2vol_ext_py(shargs)

    # Clean tmp dir.
    root_path.rmtree_p()
