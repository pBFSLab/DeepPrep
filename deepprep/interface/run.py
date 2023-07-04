import os
import time
import argparse
from multiprocessing import Pool


def get_freesurfer_threads(threads: int):
    if threads and threads > 1:
        fsthreads = f'-threads {threads} -itkthreads {threads}'
    else:
        fsthreads = ''
    return fsthreads


def run_cmd_with_timing(cmd):
    print('*' * 50)
    print(cmd)
    print('*' * 50)
    start = time.time()
    os.system(cmd)
    print('=' * 50)
    print(cmd)
    print('=' * 50, 'runtime:', ' ' * 3, time.time() - start)


def multipool(cmd, Multi_Num=2):
    cmd_pool = [['lh'], ['rh']]

    pool = Pool(Multi_Num)
    pool.starmap(cmd, cmd_pool)
    pool.close()
    pool.join()


def multiregressionpool(cmd, hemi, subj_surf_path, dst_resid_file, dst_reg_file, Multi_Num=2):
    cmd_pool = []
    for i in range(len(hemi)):
        cmd_pool.append([hemi[i], subj_surf_path, dst_resid_file, dst_reg_file])

    pool = Pool(Multi_Num)
    pool.starmap(cmd, cmd_pool)
    pool.close()
    pool.join()


def multipool_run(cmd, runs, Multi_Num=2):
    cmd_pool = []
    for i in range(len(runs)):
        cmd_pool.append([runs[i]])

    pool = Pool(Multi_Num)
    pool.starmap(cmd, cmd_pool)
    pool.close()
    pool.join()


def multipool_BidsBolds(cmd, idx, bids_entities, bids_path, Multi_Num=2):
    cmd_pool = []
    for i in range(len(idx)):
        cmd_pool.append([idx[i], bids_entities[i], bids_path[i]])

    pool = Pool(Multi_Num)
    pool.starmap(cmd, cmd_pool)
    pool.close()
    pool.join()


def multipool_BidsBolds_2(cmd, bids_entities, bids_path, Multi_Num=2):
    cmd_pool = []
    for i in range(len(bids_entities)):
        cmd_pool.append([bids_entities[i], bids_path[i]])

    pool = Pool(Multi_Num)
    pool.starmap(cmd, cmd_pool)
    pool.close()
    pool.join()


def set_envrion(freesurfer_home='/usr/local/freesurfer720',
                java_home='/usr/lib/jvm/java-11-openjdk-amd64',
                fsl_home='/usr/local/fsl',
                subjects_dir='',
                threads: int = 8):
    # FreeSurfer recon-all env
    os.environ['FREESURFER_HOME'] = f'{freesurfer_home}'
    os.environ['FREESURFER'] = f'{freesurfer_home}'
    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['PATH'] = f'{freesurfer_home}/bin:{freesurfer_home}/mni/bin:{freesurfer_home}/tktools:' + \
                         f'{freesurfer_home}/fsfast/bin:' + os.environ['PATH']
    os.environ['MINC_BIN_DIR'] = f'{freesurfer_home}/mni/bin'
    os.environ['MINC_LIB_DIR'] = f'{freesurfer_home}/mni/lib'
    os.environ['PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
    os.environ['MNI_PERL5LIB'] = f'{freesurfer_home}/mni/share/perl5'
    # FreeSurfer fsfast env
    os.environ['FSF_OUTPUT_FORMAT'] = 'nii.gz'
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

    # FastCSR
    if os.environ.get('LD_LIBRARY_PATH') is None:
        os.environ['LD_LIBRARY_PATH'] = f'{java_home}/lib:{java_home}/lib/server'
    else:
        os.environ['LD_LIBRARY_PATH'] = f'{java_home}/lib:{java_home}/lib/server:' + os.environ['LD_LIBRARY_PATH']

    # FSL
    # os.environ['PATH'] = f'{fsl_home}/bin:' + os.environ['PATH']

    # set FreeSurfer threads
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['FS_OMP_NUM_THREADS'] = str(threads)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads)


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
