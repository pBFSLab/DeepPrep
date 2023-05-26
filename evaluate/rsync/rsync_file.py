import os
from pathlib import Path


def rsync_upload(source_file, target_file, user=None, host=None):
    if user is not None and host is not None:
        cmd = f'rsync -rav {source_file} {user}@{host}:{target_file}'
    else:
        cmd = f'rsync -rav {source_file} {target_file}'
    os.system(cmd)


def rsync_download(source_file, target_file, user=None, host=None):
    if user is not None and host is not None:
        cmd = f'rsync -rav {user}@{host}:{source_file} {target_file}'
    else:
        cmd = f'rsync -rav {source_file} {target_file}'
    os.system(cmd)


def download_surf_for_reatreg():
    user = 'anning'
    host = '30.30.30.141'
    source_recon_dir = Path('/mnt/ngshare/public/share/ProjData/DeepPrep/MSC/FreeSurfer')
    target_recon_dir = Path('/mnt/ngshare/DeepPrep/ReatReg/MSC_ReatReg_smooth')
    subject_list_file = '/mnt/ngshare/DeepPrep/MSC_subject_list.txt'

    with open(subject_list_file, 'r') as f:
        subject_list = f.readlines()

    subject_list = [i.strip() for i in subject_list]

    for subject in subject_list:
        for hemi in ['lh', 'rh']:
            files = [
                f'surf/{hemi}.sulc',
                f'surf/{hemi}.curv',
                f'surf/{hemi}.thickness',
                f'surf/{hemi}.sphere',
            ]
            for file_name in files:
                source_file = source_recon_dir / subject / file_name
                target_file = target_recon_dir / subject / file_name
                if not target_file.parent.exists():
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                rsync_download(source_file, target_file, user=user, host=host)


def copy_fMRIPrep_fsaverage6():
    bold_preprocess_dir_src = Path('/mnt/ngshare/testfMRIPrep_space/Bold_Preprocess')
    bold_preprocess_dir_dst = Path('/mnt/ngshare2/weiweiMSC_all/MSC_output')

    sub_dirs = bold_preprocess_dir_src.iterdir()
    for sub_dir in sub_dirs:
        if not sub_dir.is_dir():
            continue
        if 'sub-' not in sub_dir.name:
            continue
        subject_id = sub_dir.name
        subj = subject_id.split('-')[1]
        sess = sub_dir.iterdir()
        for ses_dir in sess:
            if 'ses-func' not in ses_dir.name:
                continue
            ses = ses_dir.name.split('-')[1]
            src_func_path_dir = bold_preprocess_dir_src / f'sub-{subj}' / f'ses-{ses}' / 'func'
            dst_func_path_dir = bold_preprocess_dir_dst / f'sub-{subj}' / f'ses-{ses}' / 'func'
            for lr in ['R', 'L']:
                src_fsaverage_gii = src_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.gii'
                dst_fsaverage_gii = dst_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.func.gii'
                rsync_upload(src_fsaverage_gii, dst_fsaverage_gii)

                src_fsaverage_json = src_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.json'
                dst_fsaverage_json = dst_func_path_dir / f'sub-{subj}_ses-{ses}_task-rest_hemi-{lr}_space-fsaverage6_bold.json'
                rsync_upload(src_fsaverage_json, dst_fsaverage_json)


if __name__ == '__main__':
    # user = 'pbfs20'
    # host = '30.30.30.73'
    # src_bold_dir = Path('/mnt/ngshare2/DeepPrep_UKB_500/UKB_BoldPreprocess')
    # dst_bold_dir = Path('/mnt/ngshare2/DeepPrep_UKB_1500/UKB_BoldPreprocess')
    # rsync(src_bold_dir, dst_bold_dir, user, host)
    # copy_deepprep_1500_to_pbfs20()
    # download_surf_for_reatreg()
    copy_fMRIPrep_fsaverage6()
