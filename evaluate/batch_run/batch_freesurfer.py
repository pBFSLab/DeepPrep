import os
import sys
from pathlib import Path
import socket
import multiprocessing as mp
import shutil
import time
import datetime
import logging
import tempfile
import sh


def config_logging(log_file, console_level: int = logging.INFO, file_level: int = logging.INFO):
    format = '[%(levelname)s] %(asctime)s PID: %(process)d %(filename)s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    file_handler = logging.FileHandler(log_file, mode='a', encoding="utf8")
    file_handler.setFormatter(logging.Formatter(fmt=format, datefmt=datefmt))
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt=format, datefmt=datefmt))
    console_handler.setLevel(console_level)

    logging.basicConfig(
        level=min(console_level, file_level),
        handlers=[file_handler, console_handler],
    )


def set_environ():
    # FreeSurfer
    if os.environ.get('FREESURFER_HOME') is None:
        os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
        os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
        os.environ['PATH'] = '/usr/local/freesurfer/bin:/usr/local/freesurfer/mni/bin:' + os.environ['PATH']
        os.environ['MINC_BIN_DIR'] = '/usr/local/freesurfer/mni/bin'
        os.environ['MINC_LIB_DIR'] = '/usr/local/freesurfer/mni/lib'
        os.environ['PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'
        os.environ['MNI_PERL5LIB'] = '/usr/local/freesurfer/mni/share/perl5'


def one_reconall(node_id, data_path, task_queue):
    file = task_queue.get()
    subj = file[:file.find('.')]
    t1_file = data_path / 'input' / file
    temp_dir = tempfile.gettempdir()
    os.environ['SUBJECTS_DIR'] = temp_dir
    status_file = data_path / 'status' / node_id / file
    with open(status_file, 'w') as f:
        pass
    sh.recon_all('-s', subj, '-i', t1_file, '-all', '-parallel')
    shutil.copytree(Path(temp_dir) / subj, data_path / 'output' / subj)
    shutil.rmtree(Path(temp_dir) / subj)
    logging.info(f'Processing completed: {file}')
    status_file.unlink()


if __name__ == '__main__':
    data_path = Path('/mnt/nfs')
    n_process = 4

    hostname = socket.gethostname()
    username = os.getlogin()
    node_id = f'{hostname}_{username}'

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_file = data_path / 'log' / f'{node_id}_{time_str}.txt'
    config_logging(log_file)

    logging.info(f'Node ID: {node_id}')

    node_file = data_path / 'node' / node_id
    if not node_file.exists():
        with open(node_file, 'w') as f:
            pass
    status_path = data_path / 'status' / node_id
    if status_path.exists():
        shutil.rmtree(status_path)
    status_path.mkdir(exist_ok=True)

    task_queue = mp.Queue()
    before_mtime = 0.0
    pool = [None] * n_process
    set_environ()
    while True:
        task_file = data_path / 'task' / node_id
        while not task_file.exists():
            time.sleep(5)
        curr_mtime = os.path.getmtime(task_file)
        if curr_mtime > before_mtime:
            before_mtime = curr_mtime
            task_queue.empty()
            with open(task_file) as f:
                for line in f:
                    task_queue.put(line.strip())
            logging.info(f'Task queue updated')
        for i in range(n_process):
            if pool[i] is None:
                pool[i] = mp.Process(target=one_reconall, args=(node_id, data_path, task_queue,))
                pool[i].start()
            else:
                if pool[i].is_alive():
                    pass
                else:
                    exitcode = pool[i].exitcode
                    pid = pool[i].pid
                    if exitcode != 0:
                        logging.error(f'PID: {pid}, abnormal exit')
                    pool[i] = mp.Process(target=one_reconall, args=(node_id, data_path, task_queue,))
                    pool[i].start()
        time.sleep(5)
