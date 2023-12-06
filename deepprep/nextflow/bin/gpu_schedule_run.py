import os
import sys
import redis_lock
from redis_lock import StrictRedis


if __name__ == '__main__':
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
        cmd = "python3 " + " ".join(sys.argv[3:])
        print(cmd)
        os.system(cmd)
