import os.path
import shutil
from pathlib import Path
import zipfile


def unzip(zip_file, outpath):
    src_path = zip_file
    target_path = outpath
    z = zipfile.ZipFile(src_path)
    z.extractall(path=target_path)
    z.close()


if __name__ == '__main__':
    UKB_dir = Path('/mnt/ngshare/Data_Orig/UKB')
    T1_dir = UKB_dir / 'T1'
    rest_dir = UKB_dir / 'rfMRI'
    tmp_dir = UKB_dir / 'tmp'
    bids_outpath = UKB_dir / 'BIDS'
    sublist_txt = '/mnt/ngshare/Data_Orig/UKB/sublist.txt'

    # subjects_list = ['1000037']
    subjects_list = []
    with open(sublist_txt) as f:
        for line in f:
            line = line.split('_')[0]
            subjects_list.append(line.strip())

    tmp_dir.mkdir(exist_ok=True)
    bids_outpath.mkdir(exist_ok=True)

    for sub in subjects_list:
        sub_tmp_path = tmp_dir / sub

        if os.path.exists(sub_tmp_path) is True:
            print(sub + "： 已解压!")
            continue

        sub_tmp_path.mkdir(exist_ok=True, parents=True)

        sub_T1_zip = T1_dir / f'{sub}_20252_2_0.zip'
        sub_rest_zip = rest_dir / f'{sub}_20227_2_0.zip'

        unzip(str(sub_T1_zip), sub_tmp_path)
        unzip(str(sub_rest_zip), sub_tmp_path)

        print(sub + "： 已解压完成！")

    for sub in subjects_list:
        sub_tmp_T1_file = tmp_dir / sub / 'T1' / 'T1.nii.gz'
        sub_tmp_rest_file = tmp_dir / sub / 'fMRI' / 'rfMRI.nii.gz'

        if os.path.exists(str(sub_tmp_T1_file)) is False:
            sub_tmp_T1_file = tmp_dir / sub / 'T1' / 'unusable' / 'T1_orig_defaced.nii.gz'
            sub_tmp_rest_file = tmp_dir / sub / 'fMRI' / 'unusable' / 'rfMRI.nii.gz'

        new_sub = f'sub-{sub}'

        sub_bids_T1_anat = bids_outpath / new_sub / 'ses-02' / 'anat'
        sub_bids_rest_bold = bids_outpath / new_sub / 'ses-02' / 'func'

        sub_bids_T1_anat.mkdir(exist_ok=True, parents=True)
        sub_bids_rest_bold.mkdir(exist_ok=True, parents=True)

        sub_bids_T1_path = sub_bids_T1_anat / f'{new_sub}_ses-02_T1w.nii.gz'
        sub_bids_rest_path = sub_bids_rest_bold / f'{new_sub}_ses-02_task-rest_run-01_bold.nii.gz'

        if os.path.exists(str(sub_tmp_T1_file)) is False:
            shutil.move(sub_tmp_T1_file, sub_bids_T1_path)
            shutil.move(sub_tmp_rest_file, sub_bids_rest_path)

        print(sub + "： 已转为bids！")
