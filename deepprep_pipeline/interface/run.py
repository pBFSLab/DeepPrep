import os
import time


def run_cmd_with_timing(cmd):
    print('*' * 50)
    print(cmd)
    print('*' * 50)
    start = time.time()
    os.system(cmd)
    print('=' * 50)
    print(cmd)
    print('=' * 50, 'runtime:', ' ' * 3, time.time() - start)


def set_envrion(threads: int = 1):
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
