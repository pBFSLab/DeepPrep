import os
import shutil
from pathlib import Path
import tempfile
import sh
import bids
import cv2
import numpy as np


def set_environ():
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

    # workbench
    os.environ['PATH'] = '/opt/workbench/bin_linux64:' + os.environ['PATH']
    os.environ['PATH'] = '/usr/local/workbench/bin_linux64:' + os.environ['PATH']


def screenshot_vol_fc(data_path, pipeline):
    save_path = data_path / 'derivatives' / 'analysis' / pipeline
    save_path.mkdir(parents=True, exist_ok=True)
    layout = bids.BIDSLayout(str(data_path))
    subjs = sorted(layout.get_subjects())

    workdir = Path('/tmp/tmp_wb_command')
    workdir.mkdir(exist_ok=True)
    set_environ()
    scene_file = Path(__file__).parent / 'MNI152_T1_2mm.scene'
    for subj in subjs:
        subj_fc_path = data_path / 'derivatives' / 'analysis' / pipeline / f'sub-{subj}'
        fc_files = subj_fc_path.glob('*.nii.gz')
        for src_fc_file in fc_files:
            dst_fc_file = workdir / 'fc.nii.gz'
            if dst_fc_file.exists():
                dst_fc_file.unlink()
            shutil.copy(src_fc_file, dst_fc_file)
            screen_fc_file = subj_fc_path / src_fc_file.name.replace('.nii.gz', '.png')
            sh.wb_command('-show-scene', scene_file, 1, screen_fc_file, 1200, 600)
            print(f'screen_fc_file >>> {screen_fc_file}')
    shutil.rmtree(workdir)


def montage_fsaverage_img(img_lh_lateral, img_rh_lateral, img_lh_medial, img_rh_medial, surf):
    if surf == 'white' or surf == 'pial':
        img_up = np.hstack((img_lh_lateral[135:435, 85:515, :], img_rh_lateral[135:435, 85:515, :]))
        img_down = np.hstack((img_lh_medial[135:435, 85:515, :], img_rh_medial[135:435, 85:515, :]))
        img = np.vstack((img_up, img_down))
    elif surf == 'inflated':
        img_up = np.hstack((img_lh_lateral[100:475, 50:550, :], img_rh_lateral[100:475, 50:550, :]))
        img_down = np.hstack((img_lh_medial[100:475, 50:550, :], img_rh_medial[100:475, 50:550, :]))
        img = np.vstack((img_up, img_down))
    else:
        raise Exception('surf type error')
    # cv2.imshow('test', img)
    # cv2.waitKey()
    return img


def montage_fsaverage_file(lh_lateral, rh_lateral, lh_medial, rh_medial, montage_file, surf='white'):
    img_lh_lateral = cv2.imread(str(lh_lateral))
    img_lh_medial = cv2.imread(str(lh_medial))
    img_rh_lateral = cv2.imread(str(rh_lateral))
    img_rh_medial = cv2.imread(str(rh_medial))

    img_montage = montage_fsaverage_img(img_lh_lateral, img_rh_lateral, img_lh_medial, img_rh_medial, surf=surf)
    cv2.imwrite(str(montage_file), img_montage)


def screenshot_surf_fc(data_path, pipeline):
    save_path = data_path / 'derivatives' / 'analysis' / pipeline
    save_path.mkdir(parents=True, exist_ok=True)
    layout = bids.BIDSLayout(str(data_path))
    subjs = sorted(layout.get_subjects())

    workdir = Path('workdir')
    workdir.mkdir(exist_ok=True)
    set_environ()
    # fsaverage4
    # motor
    lh_motor_idx = 644
    rh_motor_idx = 220
    # ACC
    lh_ACC_idx = 1999
    rh_ACC_idx = 1267
    # PCC
    lh_PCC_idx = 1803
    rh_PCC_idx = 355

    lh_seeds = list()
    rh_seeds = list()
    lh_seeds.append({'name': 'LH_Motor', 'index': lh_motor_idx})
    rh_seeds.append({'name': 'RH_Motor', 'index': rh_motor_idx})
    lh_seeds.append({'name': 'LH_ACC', 'index': lh_ACC_idx})
    rh_seeds.append({'name': 'RH_ACC', 'index': rh_ACC_idx})
    lh_seeds.append({'name': 'LH_PCC', 'index': lh_PCC_idx})
    rh_seeds.append({'name': 'RH_PCC', 'index': rh_PCC_idx})
    for subj in subjs:
        for seed_dict in lh_seeds + rh_seeds:
            seed_name = seed_dict['name']
            surf_fc_file = save_path / f'sub-{subj}' / f'lh_{seed_name}_fc.mgh'
            min = 0.2
            max = 0.6
            sh.tksurfer('fsaverage4', 'lh', 'pial', '-overlay', surf_fc_file, '-fminmax', min, max, '-tcl',
                        'tksurfer_auto_screenshot.tcl')
            lh_lateral = workdir / f'lh_{seed_name}_fc_lateral.tiff'
            shutil.move('1.tiff', lh_lateral)
            lh_medial = workdir / f'lh_{seed_name}_fc_medial.tiff'
            shutil.move('2.tiff', lh_medial)

            surf_fc_file = save_path / f'sub-{subj}' / f'rh_{seed_name}_fc.mgh'
            min = 0.2
            max = 0.6
            sh.tksurfer('fsaverage4', 'rh', 'pial', '-overlay', surf_fc_file, '-fminmax', min, max, '-tcl',
                        'tksurfer_auto_screenshot.tcl')
            rh_lateral = workdir / f'rh_{seed_name}_fc_lateral.tiff'
            shutil.move('1.tiff', rh_lateral)
            rh_medial = workdir / f'rh_{seed_name}_fc_medial.tiff'
            shutil.move('2.tiff', rh_medial)

            surf_fc_file = save_path / f'sub-{subj}' / f'fc_{seed_name}.tiff'
            montage_fsaverage_file(lh_lateral, rh_lateral, lh_medial, rh_medial, surf_fc_file)
            print(f'>>> {surf_fc_file}')
    shutil.rmtree(workdir)


# /usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz
if __name__ == '__main__':
    data_path = Path('/mnt/ngshare/DeepPrep/MSC')
    # data_path = Path('/mnt/ngshare/DeepPrep/HNU_1')
    # data_path = Path('/mnt/ngshare/Data_Mirror/SDCFlows/MSC')
    pipeline = 'DeepPrep-SDC'
    screenshot_vol_fc(data_path, pipeline)
    screenshot_surf_fc(data_path, pipeline)
