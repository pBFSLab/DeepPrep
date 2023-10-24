import sys
import redis_lock
from redis_lock import StrictRedis


if __name__ == '__main__':
    if sys.argv[1].lower() == 'local':
        print("set gpu lock")
        conn = StrictRedis()
        redis_lock.reset_all(conn)
        lock = redis_lock.Lock(conn, 'nextflow-local-gpu')
