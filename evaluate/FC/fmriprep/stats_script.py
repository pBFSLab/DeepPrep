import os
from pathlib import Path
from collections import defaultdict
import re
import pandas as pd


def stats_fmriprep():
    data_path = Path('/home/weiwei/workdata/DeepPrep/fmriprep')
    ds_list = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(len(ds_list))

    data_list = list()
    for ds in ds_list:
        item = dict()
        item['ds'] = ds.name
        fmriprep_dir = ds / 'fmriprep'
        if fmriprep_dir.exists():
            fmriprep_subjs = sorted([d for d in (ds / 'fmriprep').iterdir() if d.is_dir()])
        else:
            fmriprep_subjs = list()
        freesurfer_subjs = sorted(list((ds / 'freesurfer').glob('sub-*')))
        item['n_fmriprep'] = len(fmriprep_subjs)
        item['n_freesurfer'] = len(freesurfer_subjs)
        data_list.append(item)
    df = pd.DataFrame(data_list)
    df.to_csv('fmriprep.csv', index=False)


def stats_openneuro():
    data_path = Path('/home/weiwei/workdata/DeepPrep/openneuro')
    ds_list = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(len(ds_list))

    data_list = list()
    for ds in ds_list:
        item = dict()
        item['ds'] = ds.name
        subjs = sorted(list(ds.glob('sub-*')))
        item['n_subj'] = len(subjs)
        data_list.append(item)
    df = pd.DataFrame(data_list)
    df.to_csv('openneuro.csv', index=False)


def stats_task_fmriprep():
    data_path = Path('/home/weiwei/workdata/DeepPrep/fmriprep')
    ds_list = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(len(ds_list))

    data_list = list()
    for ds in ds_list:
        fmriprep_dir = ds / 'fmriprep'
        if fmriprep_dir.exists():
            fmriprep_subjs = sorted([d for d in (ds / 'fmriprep').iterdir() if d.is_dir()])
            for subj_path in fmriprep_subjs:
                ses_list = [d for d in subj_path.glob('ses-*') if d.is_dir()]
                if len(ses_list) == 0:
                    ses_list.append(subj_path)
                for ses_path in ses_list:
                    ses_name = ses_path.name
                    if ses_name.startswith('ses-'):
                        ses_name = ses_name[ses_name.find('-') + 1:]
                    else:
                        ses_name = ''
                    task_dict = defaultdict(list)
                    func_dir = ses_path / 'func'
                    if not func_dir.exists():
                        break
                    for file in func_dir.iterdir():
                        task = re.match(r'(.*)_task-([a-zA-Z0-9]+)_', file.name).group(2)
                        run = re.match(r'(.*)_run-([a-zA-Z0-9]+)_', file.name)
                        if run is not None:
                            run = run.group(2)
                        else:
                            run = '001'
                        task_dict[task].append(run)
                    for task in task_dict:
                        task_dict[task] = sorted(list(set(task_dict[task])))
                        item = dict()
                        item['ds'] = ds.name
                        item['subject'] = subj_path.name
                        item['ses'] = ses_name
                        item['task'] = task
                        item['n_num'] = len(task_dict[task])
                        data_list.append(item)
    df = pd.DataFrame(data_list)
    df.to_csv('fmriprep_task.csv', index=False)


def stats_task_openneuro():
    data_path = Path('/home/weiwei/workdata/DeepPrep/openneuro')
    ds_list = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(len(ds_list))

    data_list = list()
    for ds in ds_list:
        subjs = sorted([d for d in ds.glob('sub-*') if d.is_dir()])
        for subj_path in subjs:
            ses_list = [d for d in subj_path.glob('ses-*') if d.is_dir()]
            if len(ses_list) == 0:
                ses_list.append(subj_path)
            for ses_path in ses_list:
                ses_name = ses_path.name
                if ses_name.startswith('ses-'):
                    ses_name = ses_name[ses_name.find('-') + 1:]
                else:
                    ses_name = ''
                task_dict = defaultdict(list)
                func_dir = ses_path / 'func'
                if not func_dir.exists():
                    break
                for file in func_dir.iterdir():
                    task = re.match(r'(.*)_task-([a-zA-Z0-9]+)_', file.name).group(2)
                    run = re.match(r'(.*)_run-([a-zA-Z0-9]+)_', file.name)
                    if run is not None:
                        run = run.group(2)
                    else:
                        run = '001'
                    task_dict[task].append(run)
                for task in task_dict:
                    task_dict[task] = sorted(list(set(task_dict[task])))
                    item = dict()
                    item['ds'] = ds.name
                    item['subject'] = subj_path.name
                    item['ses'] = ses_name
                    item['task'] = task
                    item['n_num'] = len(task_dict[task])
                    data_list.append(item)
    df = pd.DataFrame(data_list)
    df.to_csv('openneuro_task.csv', index=False)


def stats_subj_bold_openneuro():
    df = pd.read_csv('openneuro_task.csv')
    datasets = sorted(list(set(df['ds'])))
    data_list = list()
    for ds in datasets:
        item = dict()
        item['ds'] = ds
        df_ds = df[df['ds'] == ds]
        item['n_subj'] = len(set(df_ds['subject']))
        item['n_bold'] = df_ds['n_num'].sum()
        data_list.append(item)
    df_stats = pd.DataFrame(data_list)
    df_stats.to_csv('openneuro_stats.csv', index=False)
    print(df_stats)


if __name__ == '__main__':
    # stats_fmriprep()
    # stats_openneuro()
    # stats_task_fmriprep()
    # stats_task_openneuro()
    stats_subj_bold_openneuro()
