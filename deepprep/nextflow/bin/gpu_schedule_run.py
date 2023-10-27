import os
import sys
import redis_lock
from redis_lock import StrictRedis


if __name__ == '__main__':
    if sys.argv[2].lower() == 'local':
        conn = StrictRedis()
        with redis_lock.Lock(conn, 'nextflow-local-gpu'):
            cmd = "python3 " + " ".join(sys.argv[3:])
            print(cmd)
            os.system(cmd)
    else:
        cmd = "python3 " + " ".join(sys.argv[3:])
        print(cmd)
        os.system(cmd)
