#! /usr/bin/env python3
import sys
import redis_lock
from redis_lock import StrictRedis


if __name__ == '__main__':

    print("set gpu lock")
    try:
        conn = StrictRedis()
        print('connect to StrictRedis DB')
        redis_lock.reset_all(conn)
        lock1 = redis_lock.Lock(conn, 'nextflow-local-gpu-1')
        lock2 = redis_lock.Lock(conn, 'nextflow-local-gpu-2')

        conn.delete('lock_index')
        for i in range(0, 10000):
            conn.lpush('lock_index', i % 2)
    except Exception as why:
        print('Cant connect to StrictRedis DB', why)
