"""
Toolkit for data processing and model.

"""

import time
import functools
from datetime import timedelta
import sys
 
def log_exec_time(func):
    """wrapper for log the execution time of function
    
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        print('Current Func : {}...'.format(func.__name__))
        start=time.perf_counter()
        res=func(*args,**kwargs)
        end=time.perf_counter()
        print('Func {} took {:.2f}s'.format(func.__name__,(end-start)))
        return res
    return wrapper

def get_time_dif(start_time):
    """calculate the time cost from the start point.
    
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def red_str(s, tofile=False):
    s = str(s)
    if tofile:
        # s = f'**{s}**'
        pass
    else:
        s = f'\033[1;31;40m{s}\033[0m'
    return s

def get_time_str():
    return time.strftime('%Y-%m-%d_%H.%M.%S') + str(time.time() % 1)[1:6]

class Logger:
    def __init__(self, fn='xx.log', verbose=1):
        self.pre_time = time.time()
        self.fn = fn
        self.verbose = verbose

    def __str__(self):
        return self.fn

    def log(self, s='', end='\n', red=False):
        s = str(s)
        if self.verbose == 1:
            p = red_str(s) if red else s
            print(p, end=end)
        # elif self.verbose == 2:
        #     p = red_str(s, tofile=True) if red else s
        #     print(p, end=end)
        # now_time = time.time()
        # s = s + end
        # if now_time - self.pre_time > 30 * 60:
        s = get_time_str() + '\n' + s
        #     self.pre_time = now_time

        with open(self.fn, 'a') as f:
            fs = red_str('* '+s, tofile=True) if red else s
            f.write(fs)
        sys.stdout.flush()