import os
import sys
from pathlib import Path
import json
import shutil
import time
import bids
import sh
import ants
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from app.surface_projection import surface_projection as sp
from app.volume_projection import volume_projection as vp
from app.utils.utils import timing_func
from app.filters.filters import gauss_nifti, bandpass_nifti
from app.regressors.regressors import compile_regressors, regression


def set_envrion():
    # FreeSurfer recon-all env
    os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
    os.environ['FREESURFER'] = '/usr/local/freesurfer'
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

    # FSL
    os.environ['PATH'] = '/usr/local/fsl/bin:' + os.environ['PATH']


def run_with_timing(cmd):
    start = time.time()
    os.system(cmd)
    print('=' * 50)
    print(cmd)
    print('=' * 50, ' ' * 3, time.time() - start)


@timing_func
def fastsurfer_seg(t1_input: str, fs_home: Path, sub_mri_dir: Path):
    orig_o = sub_mri_dir / 'orig.mgz'
    aseg_o = sub_mri_dir / 'aparc.DKTatlas+aseg.deep.mgz'

    fastsurfer_eval = fs_home / 'FastSurferCNN' / 'eval.py'
    weight_dir = fs_home / 'checkpoints'

    cmd = f'python3 {fastsurfer_eval} ' \
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
    cmd = f'ln -sf {t1} {rawavg}'
    run_with_timing(cmd)


@timing_func
def creat_aseg_noccseg(fs_bin: Path, sub_mri_dir: Path):
    # reduce labels to aseg, then create mask (dilate 5, erode 4, largest component), also mask aseg to remove outliers
    # output will be uchar (else mri_cc will fail below)
    py = fs_bin / 'reduce_to_aseg.py'
    mask = sub_mri_dir / 'mask.mgz'
    cmd = f'python {py} ' \
          f'-i {sub_mri_dir}/aparc.DKTatlas+aseg.orig.mgz ' \
          f'-o {sub_mri_dir}/aseg.auto_noCCseg.mgz --outmask {mask}'
    run_with_timing(cmd)


@timing_func
def creat_talairach_and_nu(sub_mri_dir: Path):
    # orig_nu 44 sec nu correct
    cmd = f"mri_nu_correct.mni --no-rescale --i {sub_mri_dir}/orig.mgz --o {sub_mri_dir}/orig_nu.mgz " \
          f"--n 1 --proto-iters 1000 --distance 50 --mask {sub_mri_dir}/mask.mgz"
    run_with_timing(cmd)

    # talairach.xfm: compute talairach full head (25sec)
    cmd = f"talairach_avi --i {sub_mri_dir}/orig_nu.mgz --xfm {sub_mri_dir}/transforms/talairach.xfm"
    run_with_timing(cmd)

    # talairach.lta:  convert to lta
    freesufer_home = os.environ['FREESURFER_HOME']
    cmd = f"lta_convert --inmni {sub_mri_dir}/transforms/talairach.xfm " \
          f"--outlta {sub_mri_dir}/transforms/talairach.lta " \
          f"--src {sub_mri_dir}/orig.mgz --trg {freesufer_home}/average/mni305.cor.mgz --ltavox2vox"
    run_with_timing(cmd)

    # create better nu.mgz using talairach transform
    # TODO 修改为可以设置1.5T和3T的脚本参数
    nu_iterations = 2  # default 1.5T
    nu_iterations = "1 --proto-iters 1000 --distance 50"  # default 3T
    cmd = f"mri_nu_correct.mni --i {sub_mri_dir}/orig.mgz --o {sub_mri_dir}/nu.mgz " \
          f"--uchar {sub_mri_dir}/transforms/talairach.xfm --n {nu_iterations} --mask {sub_mri_dir}/mask.mgz"
    run_with_timing(cmd)

    # Add xfm to nu
    cmd = f"mri_add_xform_to_header -c {sub_mri_dir}/transforms/talairach.xfm {sub_mri_dir}/nu.mgz {sub_mri_dir}/nu.mgz"
    run_with_timing(cmd)


@timing_func
def creat_brainmask(sub_mri_dir: Path, need_t1=True):
    # create norm by masking nu 0.7s
    cmd = f"mri_mask {sub_mri_dir}/nu.mgz {sub_mri_dir}/mask.mgz {sub_mri_dir}/norm.mgz"
    run_with_timing(cmd)

    if need_t1:  # T1.mgz 相比 orig.mgz 更平滑，对比度更高
        # create T1.mgz from nu 96.9s
        cmd = f"mri_normalize -g 1 -mprage {sub_mri_dir}/nu.mgz {sub_mri_dir}/T1.mgz"
        run_with_timing(cmd)

        # create brainmask by masking T1
        cmd = f"mri_mask {sub_mri_dir}/T1.mgz {sub_mri_dir}/mask.mgz {sub_mri_dir}/brainmask.mgz"
        run_with_timing(cmd)
    else:
        cmd = f"ln -sf {sub_mri_dir}/norm.mgz {sub_mri_dir}/brainmask.mgz"
        run_with_timing(cmd)


@timing_func
def update_aseg(fs_bin: Path, sub_mri_dir: Path, sub_stats_dir: Path):
    # create aseg.auto including cc segmentation and add cc into aparc.DKTatlas+aseg.deep;
    # 46 sec: (not sure if this is needed), requires norm.mgz
    subj_id = 'recon'
    cmd = f"mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz " \
          f"-lta {sub_mri_dir}/transforms/cc_up.lta {subj_id}"
    run_with_timing(cmd)

    # 0.8s
    seg = sub_mri_dir / 'aparc.DKTatlas+aseg.orig.mgz'
    cmd = f"python {fs_bin}/paint_cc_into_pred.py -in_cc {sub_mri_dir}/aseg.auto.mgz -in_pred {seg} " \
          f"-out {sub_mri_dir}/aparc.DKTatlas+aseg.deep.withCC.mgz"
    run_with_timing(cmd)

    # if ["$vol_segstats" == "1"]
    # Calculate volume-based segstats for deep learning prediction (with CC, requires norm.mgz as invol)
    freesufer_home = os.environ['FREESURFER_HOME']
    cmd = f"mri_segstats --seg {sub_mri_dir}/aparc.DKTatlas+aseg.deep.withCC.mgz " \
          f"--sum {sub_stats_dir}/aparc.DKTatlas+aseg.deep.volume.stats " \
          f"--pv {sub_mri_dir}/norm.mgz --empty --brainmask {sub_mri_dir}/brainmask.mgz " \
          f"--brain-vol-from-seg --excludeid 0 --subcortgray --in {sub_mri_dir}/norm.mgz " \
          f"--in-intensity-name norm --in-intensity-units MR --etiv " \
          f"--id 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, " \
          f"52, 53, 54, 58, 60, 63, 77, 251, 252, 253, 254, 255, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, " \
          f"1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, " \
          f"1028, 1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, " \
          f"2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, " \
          f"2031, 2034, 2035 --ctab {freesufer_home}/FreeSurferColorLUT.txt --subject {subj_id}"
    run_with_timing(cmd)


@timing_func
def create_filled_from_brain(sub_mri_dir: Path, fsthreads: int = 1):
    subj_id = 'recon'

    cmd = f'cp {sub_mri_dir}/aseg.auto.mgz {sub_mri_dir}/aseg.presurf.mgz'
    run_with_timing(cmd)

    # cmd = f"recon-all -s {subj_id} -asegmerge -normalization2 -maskbfs -segmentation -fill -time"
    # run_with_timing(cmd)

    cmd = f"recon-all -s {subj_id} -asegmerge"
    run_with_timing(cmd)
    cmd = f"recon-all -s {subj_id} -normalization2"
    run_with_timing(cmd)
    cmd = f"recon-all -s {subj_id} -maskbfs"
    run_with_timing(cmd)
    cmd = f"recon-all -s {subj_id} -segmentation"
    run_with_timing(cmd)
    cmd = f"recon-all -s {subj_id} -fill"
    run_with_timing(cmd)


if __name__ == '__main__':
    start_time = time.time()

    home = Path.home()
    pwd = Path.cwd()
    data_path = Path(f'{home}/workdata/DeepPrep/BoldPipeline/TestData')

    layout = bids.BIDSLayout(str(data_path), derivatives=True)
    subjs = layout.get_subjects()

    # DeepPrep dataset_description
    derivative_deepprep_path = data_path / 'derivatives' / 'deepprep'
    derivative_deepprep_path.mkdir(exist_ok=True)
    dataset_description = dict()
    dataset_description['Name'] = 'DeepPrep Outputs'
    dataset_description['BIDSVersion'] = '1.4.0'
    dataset_description['DatasetType'] = 'derivative'
    dataset_description['GeneratedBy'] = [{'Name': 'deepprep', 'Version': '0.0.1'}]
    dataset_description_file = derivative_deepprep_path / 'dataset_description.json'
    with open(dataset_description_file, 'w') as jf:
        json.dump(dataset_description, jf, indent=4)

    set_envrion()
    deepprep_path = Path(layout.derivatives['deepprep'].root)

    atlas_type = 'MNI152_T1_2mm'
    for subj in subjs:
        subject_id = f'sub-{subj}'

        deepprep_subj_path = deepprep_path / subject_id
        deepprep_subj_path.mkdir(exist_ok=True)
        subj_recon_path = deepprep_subj_path / 'recon'
        subj_recon_path.mkdir(exist_ok=True)

        # FreeSurfer Subject Path
        os.environ['SUBJECTS_DIR'] = str(deepprep_subj_path)

        # fastsurfer seg
        fs_home = Path.cwd() / 'FastSurfer'
        fs_recon_bin = fs_home / 'recon_surf'
        subj_surf = subj_recon_path / 'surf'
        subj_mri = subj_recon_path / 'mri'
        subj_label = subj_recon_path / 'label'
        subj_stats = subj_recon_path / 'stats'

        subj_stats.mkdir(exist_ok=True)

        bids_t1s = layout.get(subject=subj, suffix='T1w', extension='.nii.gz')
        sub_t1 = bids_t1s[0].path
        fastsurfer_seg(sub_t1, fs_home, subj_mri)

        # Creating orig and rawavg from input
        creat_orig_and_rawavg(subj_mri)

        # Create noccseg
        creat_aseg_noccseg(fs_recon_bin, subj_mri)

        # Computing Talairach Transform and NU (bias corrected)
        creat_talairach_and_nu(subj_mri)

        # Creating brainmask from aseg and norm, and update aseg
        creat_brainmask(subj_mri)

        # update aseg
        update_aseg(fs_recon_bin, subj_mri, subj_stats)

        # Creating filled from brain
        create_filled_from_brain(subj_mri, fsthreads=1)

    print('time: ', time.time() - start_time)
