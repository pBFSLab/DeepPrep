import os
from pathlib import Path
import shutil
import time
import pandas
import pandas as pd


def ADNI1():
    # part
    data_path = Path('/home/weiwei/workdata/DeepPrep/ADNI1/data')
    logs_path = Path('/home/weiwei/workdata/DeepPrep/ADNI1/logs')
    save_path = Path('/home/weiwei/workdata/DeepPrep/ADNI1/input')

    files = sorted([file for file in data_path.iterdir() if file.is_file()])
    data_list = list()
    start_idx = 0
    for idx, file in enumerate(files):
        from_file_name = file.name
        to_file_name = f"{start_idx + idx:05}{''.join(file.suffixes)}"
        item = dict()
        item['from'] = from_file_name
        item['to'] = to_file_name
        data_list.append(item)
        shutil.copy(file, save_path / to_file_name)
    df = pd.DataFrame(data_list)
    log_file = logs_path / f'log_{time.strftime("%Y-%m-%d_%H:%M:%S")}.csv'
    df.to_csv(log_file, index=False)
    pass


def ADNI_3T():
    data_path = Path('/home/weiwei/workdata/DeepPrep/ADNI/data')
    logs_path = Path('/home/weiwei/workdata/DeepPrep/ADNI/logs')
    save_path = Path('/home/weiwei/workdata/DeepPrep/ADNI/input')
    logs_path.mkdir(exist_ok=True)
    save_path.mkdir(exist_ok=True)

    # df = pd.read_csv(data_path / 'ADNI-ALL-T1-3T-MPRAGE_NiFTI_5_22_2022.csv')
    files = sorted(list((data_path / 'ADNI-ALL-T1-3T-MPRAGE-NiFTI' / 'ADNI').glob('*/*/*/*/*.nii')))
    data_list = list()
    for idx, file in enumerate(files):
        print(f'{idx}/{len(files)}')
        from_file_name = file.name
        to_file_name = from_file_name.split('_')[-1]
        item = dict()
        item['from'] = from_file_name
        item['to'] = to_file_name
        data_list.append(item)
        shutil.copy(file, save_path / to_file_name)
    df = pd.DataFrame(data_list)
    log_file = logs_path / f'log_{time.strftime("%Y-%m-%d_%H:%M:%S")}.csv'
    df.to_csv(log_file, index=False)
    pass


if __name__ == '__main__':
    ADNI_3T()
