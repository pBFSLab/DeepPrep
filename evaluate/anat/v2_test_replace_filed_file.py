import os
from pathlib import Path
from shutil import copyfile, rmtree


def clear_dir(dir_path):
    os.system(f'rm -rf {dir_path}/*')


def replace_file(source_dir: str, target_dir: str, file_paths: list):
    subjects = os.listdir(target_dir)
    for subject_id in subjects:
        if 'sub' not in subject_id:
            continue
        # 如果已经存在数据，进行清理
        target_path = os.path.join(target_dir, subject_id)
        clear_dir(target_path)

        for file_path in file_paths:
            source_file = os.path.join(source_dir, subject_id, file_path)
            target_file = os.path.join(target_dir, subject_id, file_path)

            if os.path.exists(target_file):
                if os.path.islink(target_file):
                    os.unlink(target_file)
                elif os.path.isfile(target_file):
                    os.remove(target_file)

            last_dir = os.path.dirname(target_file)
            if not os.path.exists(last_dir):
                os.makedirs(last_dir)

            copyfile(source_file, target_file)
            print(f'copyfile: {source_file}  >>>  {target_file}')


def delete_file(source_dir: str, file_paths: list):
    subjects = os.listdir(source_dir)
    for subject_id in subjects:
        if 'sub' not in subject_id:
            continue

        for file_path in file_paths:
            source_file = os.path.join(source_dir, subject_id, file_path)

            if os.path.exists(source_file):
                if os.path.islink(source_file):
                    os.unlink(source_file)
                elif os.path.isfile(source_file):
                    os.remove(source_file)

            print(f'delete file: {source_file} ')


if __name__ == '__main__':
    # 用freesufer的filed替换deepprep的filed结果
    freesurfer_recon_dir = '/mnt/ngshare/DeepPrep/MSC/derivatives/FreeSurfer'
    # for sub in ['MSC1', 'MSC2', 'MSC3', 'MSC4', 'MSC5']:
    # for sub in ['MSC6']:
    #     deepprep_recon_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastCSR/{sub}/derivatives/deepprep/Recon'
    #
    #     file_paths = [
    #         'mri/orig/001.mgz',
    #         'mri/orig.mgz',
    #         'mri/filled.mgz',
    #         'mri/aseg.presurf.mgz',
    #         'mri/brain.finalsurfs.mgz',
    #         'mri/wm.mgz',
    #         'label/lh.aparc.annot',
    #         'label/rh.aparc.annot',
    #     ]
    #     replace_file(freesurfer_recon_dir, deepprep_recon_dir, file_paths)

    # for sub in ['MSC1', 'MSC2', 'MSC3', 'MSC4', 'MSC5', 'MSC6', 'MSC7', 'MSC8', 'MSC9', 'MSC10']:
    # # for sub in ['MSC6']:
    #     deepprep_recon_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFeatReg/{sub}/derivatives/deepprep/Recon'
    #
    #     file_paths = [
    #         # 'mri/orig/001.mgz',
    #         # 'mri/orig.mgz',
    #         'mri/filled.mgz',
    #         'mri/aseg.presurf.mgz',
    #         'mri/brain.finalsurfs.mgz',
    #         'mri/wm.mgz',
    #         'surf/lh.orig',
    #         'surf/rh.orig',
    #         'surf/lh.smoothwm',
    #         'surf/rh.smoothwm',
    #         'surf/lh.white.preaparc',
    #         'surf/rh.white.preaparc',
    #         'surf/lh.curv',
    #         'surf/rh.curv',
    #         'surf/lh.sulc',
    #         'surf/rh.sulc',
    #         'surf/lh.pial',
    #         'surf/rh.pial',
    #         'surf/lh.sphere',
    #         'surf/rh.sphere',
    #         'label/lh.cortex.label',
    #         'label/rh.cortex.label',
    #     ]
    #     replace_file(freesurfer_recon_dir, deepprep_recon_dir, file_paths)

    # for sub in ['MSC1', 'MSC2', 'MSC3', 'MSC4', 'MSC5', 'MSC6', 'MSC7', 'MSC8', 'MSC9', 'MSC10']:
    # for sub in ['MSC1']:
    #     deepprep_recon_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurfer/{sub}/derivatives/deepprep/Recon'
    #
    #     file_paths = [
    #         # 'mri/orig/001.mgz',
    #         # 'mri/orig.mgz',
    #     ]
    #     replace_file(freesurfer_recon_dir, deepprep_recon_dir, file_paths)

    for sub in ['MSC1', 'MSC2', 'MSC3', 'MSC4', 'MSC5', 'MSC6', 'MSC7', 'MSC8', 'MSC9', 'MSC10']:
    # for sub in ['MSC1']:
        deepprep_recon_dir = f'/mnt/ngshare/Data_Mirror/FreeSurferFastSurferFastCSRFeatReg/{sub}/derivatives/deepprep/Recon'

        file_paths = [
            # 'surf/lh.curv',
            # 'surf/rh.curv',
            # 'surf/lh.thickness',
            # 'surf/rh.thickness',
            'surf/lh.sphere.reg',
            'surf/rh.sphere.reg',
            # 'label/lh.aparc.annot',
            # 'label/rh.aparc.annot',
        ]
        delete_file(deepprep_recon_dir, file_paths)
