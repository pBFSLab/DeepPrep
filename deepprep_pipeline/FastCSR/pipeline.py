import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, Process, Lock
import logging
import subprocess

import numpy as np
import sh
import ants
from scipy.ndimage import binary_fill_holes, binary_dilation


def set_environ():
    # FreeSurfer
    freesurfer_home = os.environ.get('FREESURFER_HOME')
    if freesurfer_home is None:
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
        os.environ['PATH'] = '/usr/local/freesurfer/bin:/usr/local/freesurfer/mni/bin:' + os.environ['PATH']
    # nnUNet
    fastcsr_path = Path(os.path.split(__file__)[0])
    os.environ['RESULTS_FOLDER'] = str(fastcsr_path / 'model' / 'nnUNet_trained_models')
    # for nighres
    if os.environ.get('LD_LIBRARY_PATH') is None:
        os.environ['LD_LIBRARY_PATH'] = \
            '/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/lib/server'
    else:
        os.environ['LD_LIBRARY_PATH'] = \
            '/usr/lib/jvm/java-11-openjdk-amd64/lib:/usr/lib/jvm/java-11-openjdk-amd64/lib/server:' \
            + os.environ['LD_LIBRARY_PATH']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', required=True, help='Subject ID for directory inside $SUBJECTS_DIR to be created')
    parser.add_argument('--t1', help="The input T1 file path")
    parser.add_argument('--sd', default=os.environ.get('SUBJECTS_DIR'),
                        help='Output directory $SUBJECTS_DIR (pass via environment or here)')
    parser.add_argument('--parallel_scheduling', default='on', choices=['on', 'off'],
                        help="Whether to enable parallel scheduling for shortening processing time")
    parser.add_argument('--optimizing_surface', default='on', choices=['on', 'off'],
                        help='Whether to enable optimizing the white surface position')
    parser.add_argument('--pial', default=False, action='store_true', help="Whether to generate pial surface")
    parser.add_argument('--verbose', default=False, action='store_true', help="Whether to output detailed log")

    args = parser.parse_args()
    if args.sd is None:
        raise ValueError('Subjects dir need to set via $SUBJECTS_DIR environment or --sd parameter')
    else:
        os.environ['SUBJECTS_DIR'] = args.sd
    subj_dir = Path(args.sd) / args.sid
    if not os.path.exists(subj_dir) and args.t1 is None:
        raise ValueError(f'{subj_dir} is not exists and --t1 is None, please check.')
    args_dict = vars(args)
    if args_dict['parallel_scheduling'] == 'on':
        args_dict['parallel_scheduling'] = True
    else:
        args_dict['parallel_scheduling'] = False
    if args_dict['optimizing_surface'] == 'on':
        args_dict['optimizing_surface'] = True
    else:
        args_dict['optimizing_surface'] = False
    args = argparse.Namespace(**args_dict)

    return args


def config_logging(file_name=None, console_level=logging.INFO, file_level=logging.DEBUG):
    format = '[%(levelname)s] %(asctime)s PID: %(process)d %(filename)s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    handlers = list()
    if file_name is not None:
        file_handler = logging.FileHandler(file_name, mode='a', encoding="utf8")
        file_handler.setFormatter(logging.Formatter(fmt=format, datefmt=datefmt))
        file_handler.setLevel(file_level)
        handlers.append(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt=format, datefmt=datefmt))
    console_handler.setLevel(console_level)
    handlers.append(console_handler)

    if file_name is not None:
        log_level = min(console_level, file_level)
    else:
        log_level = console_level
    logging.basicConfig(level=log_level, handlers=handlers)
    logging.info('Please cite the following paper when using FastCSR:\n****************************************')


def log_msg(msg, lock, level):
    if level == logging.INFO:
        if lock is not None:
            with lock:
                logging.info(msg)
        else:
            logging.info(msg)
    elif level == logging.ERROR:
        if lock is not None:
            with lock:
                logging.error(msg)
        else:
            logging.error(msg)


def create_filled(args, lock=None):
    # prepare input
    subj_dir = Path(args.sd) / args.sid

    input_path = subj_dir / 'tmp' / 'filled' / 'tmp_input'
    output_path = subj_dir / 'tmp' / 'filled' / 'tmp_output'
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    orig = ants.image_read(str(subj_dir / 'mri' / 'orig.mgz'))
    ants.image_write(orig, str(input_path / f'{args.sid}_0000.nii.gz'))
    # nnUNet-based segmentation to create filled file
    cmd = f'nnUNet_predict -m 3d_fullres -tr nnUNetTrainerV2 -t 601 -chk final -i {input_path} -o {output_path}'.split()
    ret = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if ret.returncode == 0:
        msg = 'Filled segmentation model inference completed.'
        log_msg(msg, lock, logging.INFO)
    else:
        msg = 'Filled segmentation model inference failed.'
        log_msg(msg, lock, logging.ERROR)
        exit(-1)
    # convert segmentation label to FreeSurfer filled label
    filled = ants.image_read(str(output_path / f'{args.sid}.nii.gz'))
    filled_np = filled.numpy()
    filled_np[filled_np == 1] = 127
    filled_np[filled_np == 2] = 255
    filled_pred = ants.from_numpy(filled_np, filled.origin, filled.spacing, filled.direction)
    ants.image_write(filled_pred, str(subj_dir / 'mri' / 'filled.mgz'))
    shutil.rmtree(subj_dir / 'tmp' / 'filled')
    msg = 'The mri/filled.mgz file has been generated.'
    log_msg(msg, lock, logging.INFO)


def create_aseg_presurf(args, lock=None):
    # prepare input
    subj_dir = Path(args.sd) / args.sid

    input_path = subj_dir / 'tmp' / 'aseg_presurf' / 'tmp_input'
    output_path = subj_dir / 'tmp' / 'aseg_presurf' / 'tmp_output'
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    orig = ants.image_read(str(subj_dir / 'mri' / 'orig.mgz'))
    ants.image_write(orig, str(input_path / f'{args.sid}_0000.nii.gz'))
    # nnUNet-based segmentation to create filled file
    cmd = f'nnUNet_predict -m 3d_fullres -tr nnUNetTrainerV2 -t 602 -chk final -i {input_path} -o {output_path}'.split()
    ret = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if ret.returncode == 0:
        msg = 'Aseg_presurf segmentation model inference completed.'
        log_msg(msg, lock, logging.INFO)
    else:
        msg = 'Aseg_presurf segmentation model inference failed.'
        log_msg(msg, lock, logging.ERROR)
        exit(-1)
    # convert segmentation label to FreeSurfer aseg.presurf label
    aseg_presurf = ants.image_read(str(output_path / f'{args.sid}.nii.gz'))
    aseg_presurf_np = aseg_presurf.numpy()
    fastcsr_path = Path(os.path.split(__file__)[0])
    with open(fastcsr_path / 'model' / 'aseg_label_trans.json') as jf:
        label2aseg = json.load(jf)['label2aseg']
    aseg_pred_np = np.zeros_like(aseg_presurf_np)
    for label in label2aseg:
        aseg_pred_np[aseg_presurf_np == int(label)] = label2aseg[label]
    aseg_presurf_pred = ants.from_numpy(aseg_pred_np, aseg_presurf.origin, aseg_presurf.spacing,
                                        aseg_presurf.direction)
    ants.image_write(aseg_presurf_pred, str(subj_dir / 'mri' / 'aseg.presurf.mgz'))
    shutil.rmtree(subj_dir / 'tmp' / 'aseg_presurf')
    msg = 'The mri/aseg.presurf.mgz file has been generated.'
    log_msg(msg, lock, logging.INFO)


def create_brainmask(args, lock=None):
    subj_dir = Path(args.sd) / args.sid
    aseg_presurf = ants.image_read(str(subj_dir / 'mri' / 'aseg.presurf.mgz'))
    aseg_presurf = ants.iMath_get_largest_component(aseg_presurf)
    aseg_presurf_np = aseg_presurf.numpy()
    brain_mask = aseg_presurf_np.astype(bool)
    brain_mask = binary_fill_holes(brain_mask)
    brain_mask = binary_dilation(brain_mask, iterations=5)
    brain_mask = binary_fill_holes(brain_mask)
    brain_mask = brain_mask.astype(np.float32)
    brainmask = ants.from_numpy(brain_mask, aseg_presurf.origin, aseg_presurf.spacing, aseg_presurf.direction)
    ants.image_write(brainmask, str(subj_dir / 'mri' / 'brainmask.mgz'))
    msg = 'The mri/brainmask.mgz file has been generated.'
    log_msg(msg, lock, logging.INFO)


def create_levelset(args, lock=None):
    fastcsr_path = Path(os.path.split(__file__)[0])
    cmd_pool = list()
    cmd = f"python3 {fastcsr_path / 'fastcsr_model_infer.py'} --fastcsr_subjects_dir {args.sd} --subj {args.sid} --hemi lh".split()
    cmd_pool.append(cmd)
    cmd = f"python3 {fastcsr_path / 'fastcsr_model_infer.py'} --fastcsr_subjects_dir {args.sd} --subj {args.sid} --hemi rh".split()
    cmd_pool.append(cmd)
    if args.parallel_scheduling:
        lh_process = subprocess.Popen(cmd_pool[0], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        rh_process = subprocess.Popen(cmd_pool[1], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        lh_retcode = lh_process.wait()
        rh_retcode = rh_process.wait()
        if lh_retcode != 0 or rh_retcode != 0:
            msg = 'Levelset regression model inference failed.'
            log_msg(msg, lock, logging.ERROR)
            exit(-1)
    else:
        for cmd in cmd_pool:
            ret = subprocess.run(cmd, stdout=subprocess.DEVNULL)
            if ret.returncode != 0:
                msg = 'Levelset model regression inference failed.'
                log_msg(msg, lock, logging.ERROR)
                exit(-1)
    msg = 'Levelset model regression inference completed.'
    log_msg(msg, lock, logging.INFO)


def levelset2surf(args, lock=None, surfix='orig'):
    fastcsr_path = Path(os.path.split(__file__)[0])
    cmd_pool = list()
    cmd = f"python3 {fastcsr_path / 'levelset2surf.py'} --fastcsr_subjects_dir {args.sd} --subj {args.sid} --hemi lh --suffix {surfix}".split()
    cmd_pool.append(cmd)
    cmd = f"python3 {fastcsr_path / 'levelset2surf.py'} --fastcsr_subjects_dir {args.sd} --subj {args.sid} --hemi rh --suffix {surfix}".split()
    cmd_pool.append(cmd)
    if args.parallel_scheduling:
        lh_process = subprocess.Popen(cmd_pool[0], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        rh_process = subprocess.Popen(cmd_pool[1], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        lh_retcode = lh_process.wait()
        rh_retcode = rh_process.wait()
        if lh_retcode != 0 or rh_retcode != 0:
            msg = 'Surface generation failed.'
            log_msg(msg, lock, logging.ERROR)
            exit(-1)
    else:
        for cmd in cmd_pool:
            ret = subprocess.run(cmd, stdout=subprocess.DEVNULL)
            if ret.returncode != 0:
                msg = 'Surface generation failed.'
                log_msg(msg, lock, logging.ERROR)
                exit(-1)
    msg = 'Surface generation completed.'
    log_msg(msg, lock, logging.INFO)


def create_brain_finalsurfs(args, lock=None):
    fastcsr_path = Path(os.path.split(__file__)[0])
    cmd = f"python3 {fastcsr_path / 'brain_finalsurfs_model_infer.py'} --fastcsr_subjects_dir {args.sd} --subj {args.sid}".split()
    ret = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if ret.returncode == 0:
        msg = 'Brain_finalsurfs regression model inference completed.'
        log_msg(msg, lock, logging.INFO)
    else:
        msg = 'Brain_finalsurfs regression model inference failed.'
        log_msg(msg, lock, logging.ERROR)
        exit(-1)


def create_wm(args, lock=None):
    subj_dir = Path(args.sd) / args.sid
    filled = ants.image_read(str(subj_dir / 'mri' / 'filled.mgz'))
    aseg_presurf = ants.image_read(str(subj_dir / 'mri' / 'aseg.presurf.mgz'))
    filled_np = filled.numpy()
    aseg_presurf_np = aseg_presurf.numpy()
    wm_np = np.ones_like(filled_np)
    wm_np[filled_np == 127] = 255
    wm_np[filled_np == 255] = 255
    wm_np[aseg_presurf_np == 13] = 255
    wm = ants.from_numpy(wm_np, filled.origin, filled.spacing, filled.direction)
    ants.image_write(wm, str(subj_dir / 'mri' / 'wm.mgz'))
    msg = 'The mri/wm.mgz file has been generated.'
    log_msg(msg, lock, logging.INFO)


def create_white_surface(args, lock=None):
    cmd_pool = list()
    if args.pial:
        cmd = f"recon-all -s {args.sid} -hemi lh -white -no-isrunning".split()
        cmd_pool.append(cmd)
        cmd = f"recon-all -s {args.sid} -hemi rh -white -no-isrunning".split()
        cmd_pool.append(cmd)
    else:
        cmd = f"mris_make_surfaces -aseg aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs {args.sid} lh".split()
        cmd_pool.append(cmd)
        cmd = f"mris_make_surfaces -aseg aseg.presurf -white white.preaparc -noaparc -whiteonly -mgz -T1 brain.finalsurfs {args.sid} rh".split()
        cmd_pool.append(cmd)

    if args.parallel_scheduling:
        lh_process = subprocess.Popen(cmd_pool[0], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        rh_process = subprocess.Popen(cmd_pool[1], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        lh_retcode = lh_process.wait()
        rh_retcode = rh_process.wait()
        if lh_retcode != 0 or rh_retcode != 0:
            msg = 'Surface optimization failed.'
            log_msg(msg, lock, logging.ERROR)
            exit(-1)
    else:
        for cmd in cmd_pool:
            ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if ret.returncode != 0:
                msg = 'Surface optimization failed.'
                log_msg(msg, lock, logging.ERROR)
                exit(-1)
    msg = 'Surface optimization completed.'
    log_msg(msg, lock, logging.INFO)


def parallel_scheduling(args):
    lock = Lock()
    # make sure the mri/filled.mgz file has been created
    logging.info('-----------------------Generate mri/filled.mgz file-------------------------------')
    filled_process = None
    if not os.path.exists(subj_dir / 'mri' / 'filled.mgz'):
        filled_process = Process(target=create_filled, args=(args, lock))
        filled_process.start()
    else:
        logging.info('The mri/filled.mgz file already exists, skip this step.')

    # make sure the mri/aseg.presurf.mgz file has been created
    logging.info('--------------------Generate mri/aseg.presurf.mgz file----------------------------')
    aseg_presurf_process = None
    if not os.path.exists(subj_dir / 'mri' / 'aseg.presurf.mgz'):
        aseg_presurf_process = Process(target=create_aseg_presurf, args=(args, lock))
        aseg_presurf_process.start()
    else:
        logging.info('The mri/aseg.presurf.mgz file already exists, skip this step.')

    if filled_process is not None:
        filled_process.join()
        filled_process.close()
    if aseg_presurf_process is not None:
        aseg_presurf_process.join()
        aseg_presurf_process.close()

    # make sure the mri/brainmask.mgz file has been created
    logging.info('---------------------Generate mri/brainmask.mgz file------------------------------')
    if not os.path.exists(subj_dir / 'mri' / 'brainmask.mgz'):
        create_brainmask(args, lock)
    else:
        logging.info('The mri/brainmask.mgz file already exists, skip this step.')

    # make sure the mri/wm.mgz file has been created
    logging.info('-------------------------Generate mri/wm.mgz file---------------------------------')
    if not os.path.exists(subj_dir / 'mri' / 'wm.mgz'):
        create_wm(args, lock)
    else:
        logging.info('The mri/wm.mgz file already exists, skip this step.')

    logging.info('-------------------Generate mri/?h_levelset.nii.gz file---------------------------')
    create_levelset(args, lock)

    # make sure the mri/brain.finalsurfs.mgz file has been created
    logging.info('-------------------Generate mri/brain.finalsurfs.mgz file-------------------------')
    brain_finalsurfs_process = None
    if not os.path.exists(subj_dir / 'mri' / 'brain.finalsurfs.mgz'):
        brain_finalsurfs_process = Process(target=create_brain_finalsurfs, args=(args, lock))
        brain_finalsurfs_process.start()
    else:
        logging.info('The mri/brain.finalsurfs.mgz file already exists, skip this step.')

    # create surface
    logging.info('-----------------------------Generate surface-------------------------------------')
    levelset2surf(args, lock)

    if brain_finalsurfs_process is not None:
        brain_finalsurfs_process.join()
        brain_finalsurfs_process.close()

    # optimizing surface
    logging.info('---------------------------Surface optimization-----------------------------------')
    if args.optimizing_surface:
        create_white_surface(args, lock)


def serial_scheduling(args):
    # make sure the mri/filled.mgz file has been created
    logging.info('-----------------------Generate mri/filled.mgz file-------------------------------')
    if not os.path.exists(subj_dir / 'mri' / 'filled.mgz'):
        create_filled(args)
    else:
        logging.info('The mri/filled.mgz file already exists, skip this step.')

    # make sure the mri/aseg.presurf.mgz file has been created
    logging.info('--------------------Generate mri/aseg.presurf.mgz file----------------------------')
    if not os.path.exists(subj_dir / 'mri' / 'aseg.presurf.mgz'):
        create_aseg_presurf(args)
    else:
        logging.info('The mri/aseg.presurf.mgz file already exists, skip this step.')

    # make sure the mri/brainmask.mgz file has been created
    logging.info('---------------------Generate mri/brainmask.mgz file------------------------------')
    if not os.path.exists(subj_dir / 'mri' / 'brainmask.mgz'):
        create_brainmask(args)
    else:
        logging.info('The mri/brainmask.mgz file already exists, skip this step.')

    # make sure the mri/wm.mgz file has been created
    logging.info('-------------------------Generate mri/wm.mgz file---------------------------------')
    if not os.path.exists(subj_dir / 'mri' / 'wm.mgz'):
        create_wm(args)
    else:
        logging.info('The mri/wm.mgz file already exists, skip this step.')

    logging.info('-------------------Generate mri/?h_levelset.nii.gz file---------------------------')
    create_levelset(args)

    # make sure the mri/brain.finalsurfs.mgz file has been created
    logging.info('-------------------Generate mri/brain.finalsurfs.mgz file-------------------------')
    if not os.path.exists(subj_dir / 'mri' / 'brain.finalsurfs.mgz'):
        create_brain_finalsurfs(args)
    else:
        logging.info('The mri/brain.finalsurfs.mgz file already exists, skip this step.')

    # create surface
    logging.info('------------------------Generate surf/?h.orig file--------------------------------')
    levelset2surf(args)

    # optimizing surface
    if args.optimizing_surface:
        logging.info('---------------------------Surface optimization-----------------------------------')
        create_white_surface(args)


if __name__ == '__main__':
    args = parse_args()
    set_environ()
    config_logging()
    subj_dir = Path(args.sd) / args.sid
    # make sure the mri/orig.mgz file has been created
    logging.info('------------------------Generate mri/orig.mgz file--------------------------------')
    if not os.path.exists(subj_dir):
        logging.info(
            f'The {args.sid} folder does not exist in SUBJECTS_DIR, use the FreeSurfer recon-all command to generate.')
        sh.recon_all('-s', args.sid, '-i', args.t1, '-motioncor')
    elif not os.path.exists(subj_dir / 'mri' / 'orig.mgz'):
        logging.info(
            f'The {args.sid} folder already exists in SUBJECTS_DIR, but the mri/orig.mgz file does not exist, use the FreeSurfer reconall command to generate.')
        sh.recon_all('-s', args.sid, '-motioncor')
    else:
        logging.info('The mri/orig.mgz file already exists, skip this step.')

    if args.parallel_scheduling:
        parallel_scheduling(args)
    else:
        serial_scheduling(args)
