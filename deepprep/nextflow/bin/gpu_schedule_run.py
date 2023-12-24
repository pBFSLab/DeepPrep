#! /usr/bin/env python3
import os
import sys
import redis_lock
from redis_lock import StrictRedis

from gpu_manage import GPUManager


if __name__ == '__main__':
    gpu_manager = GPUManager()
    gpu = gpu_manager.auto_choice()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print(sys.argv)
    assert os.path.exists(sys.argv[3]), f"{sys.argv[3]}"
    if sys.argv[2].lower() == 'local':
        conn = StrictRedis()
        if sys.argv[1] == 'double':
            with redis_lock.Lock(conn, 'nextflow-local-gpu-1'), redis_lock.Lock(conn, 'nextflow-local-gpu-2'):
                cmd = "python3 " + " ".join(sys.argv[3:])
                print(cmd)
                results = os.popen(cmd).readlines()
                print(results)
        else:
            if 'lh' in sys.argv:
                lock_name = 'nextflow-local-gpu-1'
            elif 'rh' in sys.argv:
                lock_name = 'nextflow-local-gpu-2'
            else:
                lock_i = int(conn.lpop('lock_index'))
                lock_name = ['nextflow-local-gpu-1', 'nextflow-local-gpu-2'][lock_i]
            print(f"lock_name: {lock_name}")
            with redis_lock.Lock(conn, lock_name):
                cmd = "python3 " + " ".join(sys.argv[3:])
                print(cmd)
                os.system(cmd)
    else:
        cmd = "python3 " + " ".join(sys.argv[3:])
        print(cmd)
        os.system(cmd)
