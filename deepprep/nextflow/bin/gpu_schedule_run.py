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
    if sys.argv[2].lower() == 'local':
        conn = StrictRedis()
        if sys.argv[1] == 'double':
            with redis_lock.Lock(conn, 'nextflow-local-gpu-1'), redis_lock.Lock(conn, 'nextflow-local-gpu-2'):
                cmd = "python3 " + " ".join(sys.argv[3:])
                print(cmd)
                os.system(cmd)
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
        conn = StrictRedis()
        hostname = os.popen('hostname').readlines()[0].strip()
        redis_lock.Lock(conn, f'hostname-{hostname}-gpu-{gpu}')
        with redis_lock.Lock(conn, f'hostname-{hostname}-gpu-{gpu}'):
            cmd = "python3 " + " ".join(sys.argv[3:])
            print(cmd)
            os.system(cmd)
