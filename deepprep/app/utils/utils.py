import time
from functools import wraps


def timing_func(function):
    @wraps(function)
    def timer(*args, **kwargs):
        tic = time.time()
        result = function(*args, **kwargs)
        toc = time.time()
        print('[Finished func: {func_name} in {time:.4f}s]'.format(func_name=function.__name__, time=toc - tic))
        return result

    return timer
