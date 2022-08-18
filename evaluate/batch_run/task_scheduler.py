import os
from pathlib import Path
import time
import math
import datetime

if __name__ == '__main__':
    data_path = Path('/mnt/nfs')
    tasks_file = data_path / 'tasks.txt'
    task_files = list()
    with open(tasks_file) as f:
        for line in f:
            task_files.append(line.strip())
    before_tasks_mt = os.path.getmtime(tasks_file)
    tasks = [task_file[:task_file.find('.')] for task_file in task_files]
    tasks_dict = {task: task_file for task, task_file in zip(tasks, task_files)}
    need_update = True
    n_node = 0
    while True:
        curr_tasks_mt = os.path.getmtime(tasks_file)
        if curr_tasks_mt > before_tasks_mt:
            task_files = list()
            with open(tasks_file) as f:
                for line in f:
                    task_files.append(line.strip())
            tasks = [task_file[:task_file.find('.')] for task_file in task_files]
            tasks_dict = {task: task_file for task, task_file in zip(tasks, task_files)}
            need_update = True
        processed = os.listdir(data_path / 'output')
        processed = sorted(list(set(tasks) & set(processed)))
        with open(data_path / 'processed.txt', 'w') as f:
            f.write('\n'.join(processed))
        precessing_files = sorted([f.name for f in (data_path / 'status').glob('*/*')])
        with open(data_path / 'processing.txt', 'w') as f:
            f.write('\n'.join(precessing_files))
        precessing = [file[:file.find('.')] for file in precessing_files]
        nodes = os.listdir(data_path / 'node')
        if len(nodes) == 0:
            time.sleep(10)
            continue
        need_preocess = sorted(list(set(tasks) - set(precessing) - set(processed)))
        n_task_per_node = math.ceil(len(need_preocess) / len(nodes))
        if n_node != len(nodes) or need_update:
            n_node = len(nodes)
            need_update = False
            for idx, node in enumerate(nodes):
                node_tasks = need_preocess[idx * n_task_per_node:(idx + 1) * n_task_per_node]
                node_files = [tasks_dict[task] for task in node_tasks]
                task_file = data_path / 'task' / node
                with open(task_file, 'w') as f:
                    f.write('\n'.join(node_files))
            time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print(f'{time_str} update task')
            print(nodes)
        time.sleep(30)
