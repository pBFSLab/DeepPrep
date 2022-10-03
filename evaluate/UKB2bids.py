import os.path
import shutil
from pathlib import Path
import zipfile
import pandas as pd


def unzip(zip_file, outpath):
    src_path = zip_file
    target_path = outpath
    z = zipfile.ZipFile(src_path)
    z.extractall(path=target_path)
    z.close()


if __name__ == '__main__':
    UKB_dir = '/mnt/ngshare2'
    T1_dir = '/media/pbfs19/c97fb2f7-1015-4387-bd54-b6ff6a54b856/UKB/T1'
    rest_dir = '/media/pbfs19/c97fb2f7-1015-4387-bd54-b6ff6a54b856/UKB/rfMRI'
    tmp_dir = os.path.join(UKB_dir, 'tmp')
    bids_outpath = os.path.join(UKB_dir, 'BIDS')
    sublist_csv = '/home/pbfs19/workspace/f.20191.filtered_T1_01.csv'

    df = pd.read_csv(sublist_csv)
    df_data = pd.DataFrame(df)
    subjects_list = list(df_data['sub'])

    if os.path.exists(tmp_dir) is False:
        os.makedirs(tmp_dir)
    if os.path.exists(bids_outpath) is False:
        os.makedirs(bids_outpath)

    for sub in subjects_list:
        sub = str(sub)
        sub_tmp_path = os.path.join(tmp_dir, sub)
        if os.path.exists(sub_tmp_path) is True:
            print(sub + "： 已解压!")
            continue
        else:
            os.makedirs(sub_tmp_path)

        sub_T1_zip = os.path.join(T1_dir,  f'{sub}_20252_2_0.zip')
        sub_rest_zip = os.path.join(rest_dir, f'{sub}_20227_2_0.zip')

        if os.path.exists(sub_T1_zip) is False or os.path.exists(sub_rest_zip) is False:
            print(sub + '没有！！！！！！！！！！！！！')
            continue

        unzip(sub_T1_zip, sub_tmp_path)
        unzip(sub_rest_zip, sub_tmp_path)

        print(sub + "： 已解压完成！")

    for sub in subjects_list:
        sub = str(sub)
        sub_tmp_T1_file = os.path.join(tmp_dir, sub, 'T1', 'T1.nii.gz')
        sub_tmp_rest_file = os.path.join(tmp_dir, sub, 'fMRI', 'rfMRI.nii.gz')

        if os.path.exists(sub_tmp_T1_file) is False:
            sub_tmp_T1_file = os.path.join(tmp_dir, sub, 'T1', 'unusable', 'T1_orig_defaced.nii.gz')
            if os.path.exists(sub_tmp_T1_file) is False:
                sub_tmp_T1_file = os.path.join(tmp_dir, sub, 'T1', 'T1_orig_defaced.nii.gz')
        if os.path.exists(sub_tmp_rest_file) is False:
            sub_tmp_rest_file = os.path.join(tmp_dir, sub, 'fMRI', 'unusable', 'rfMRI.nii.gz')

        if os.path.exists(sub_tmp_rest_file) is False:
            continue


        new_sub = f'sub-{sub}'

        sub_bids_T1_anat = os.path.join(bids_outpath, new_sub, 'ses-02', 'anat')
        sub_bids_rest_bold = os.path.join(bids_outpath, new_sub, 'ses-02', 'func')

        if os.path.exists(sub_bids_T1_anat) is False:
            os.makedirs(sub_bids_T1_anat)
        if os.path.exists(sub_bids_rest_bold) is False:
            os.makedirs(sub_bids_rest_bold)

        sub_bids_T1_path = os.path.join(sub_bids_T1_anat, f'{new_sub}_ses-02_T1w.nii.gz')
        sub_bids_rest_path = os.path.join(sub_bids_rest_bold, f'{new_sub}_ses-02_task-rest_run-01_bold.nii.gz')

        if os.path.exists(sub_bids_T1_path) is False:
            shutil.move(sub_tmp_T1_file, sub_bids_T1_path)
        if os.path.exists(sub_bids_rest_path) is False:
            shutil.move(sub_tmp_rest_file, sub_bids_rest_path)

        print(sub + "： 已转为bids！")
