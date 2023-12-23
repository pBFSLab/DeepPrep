import sys
import redis_lock
from redis_lock import StrictRedis
from gpu_manage import check_gpus


if __name__ == '__main__':

    if sys.argv[1].lower() == 'local':
        gpu_available, info = check_gpus(11.8, 23500)
        if not gpu_available:
            raise ImportError(info)

        print("set gpu lock")
        conn = StrictRedis()
        redis_lock.reset_all(conn)
        lock1 = redis_lock.Lock(conn, 'nextflow-local-gpu-1')
        lock2 = redis_lock.Lock(conn, 'nextflow-local-gpu-2')

        conn.delete('lock_index')
        for i in range(0, 10000):
            conn.lpush('lock_index', i % 2)
