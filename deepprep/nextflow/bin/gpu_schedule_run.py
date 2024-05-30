#! /usr/bin/env python3
import os
import sys
import subprocess
import redis_lock
from redis_lock import StrictRedis

from gpu_manage import GPUManager, check_gpus


if __name__ == '__main__':
    print(f'INFO: Device select strategy: {sys.argv[1]}')
    pipeline_min_gpu_memory_required = 11000
    cuda_version_min = 11.8

    if sys.argv[1].isdigit():
        check_gpus()
        gpu = sys.argv[1]
    elif sys.argv[1].lower() == 'cpu':
        gpu = ''
    else:
        print(f'INFO: Auto select device by ： {sys.argv[1]}')
        try:
            gpu_manager = GPUManager()
            gpu = gpu_manager.auto_choice()
            print(f'WARNNING: Auto select: GPU {gpu}')
        except Exception as e:
            print(e)
            print(f'WARNNING: Auto select: cpu')
            gpu = ''

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print(f'INFO: GPU: {gpu}')
    print(f'INFO: sys.argv : {sys.argv}')
    run_count = 'one'
    if gpu:
        gpu_available, info, gpu_memory = check_gpus(cuda_version_min, pipeline_min_gpu_memory_required)
        if not gpu_available:
            raise ImportError(info)

        process_memory_required = int(sys.argv[2])
        if gpu_memory > process_memory_required * 2:
            run_count = 'two'
        elif gpu_memory < process_memory_required:
            gpu = ''
    assert os.path.exists(sys.argv[4]), f"{sys.argv[4]}"

    conn = None
    try:
        conn = StrictRedis()
        conn.lpop('lock_index')
        print('connect to StrictRedis DB')
    except Exception as why:
        conn = None
        print(f'Cant connect to StrictRedis DB: {why}')

    if gpu and conn:
        print(f'INFO: run_count : {run_count}')
        if run_count == 'two':
            if 'lh' in sys.argv:
                lock_name = 'nextflow-local-gpu-1'
            elif 'rh' in sys.argv:
                lock_name = 'nextflow-local-gpu-2'
            else:
                lock_i = int(conn.lpop('lock_index'))
                lock_name = ['nextflow-local-gpu-1', 'nextflow-local-gpu-2'][lock_i]
            print(f"lock_name: {lock_name}")
            with redis_lock.Lock(conn, lock_name):
                cmd = f"python3 " + " ".join(sys.argv[4:])
                print(f'INFO: GPU: {cmd}')
                status, results = subprocess.getstatusoutput(cmd)
                print(results)
                assert status == 0
        else:
            with redis_lock.Lock(conn, 'nextflow-local-gpu-1'), redis_lock.Lock(conn, 'nextflow-local-gpu-2'):
                cmd = f"python3 " + " ".join(sys.argv[4:])
                print(f'INFO: GPU: {cmd}')
                status, results = subprocess.getstatusoutput(cmd)
                print(results)
                assert status == 0
    else:
        cmd = f"python3 " + " ".join(sys.argv[4:])
        print(f'INFO: GPU: {cmd}')
        status, results = subprocess.getstatusoutput(cmd)
        print(results)
        assert status == 0
