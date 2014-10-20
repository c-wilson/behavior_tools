__author__ = 'chris'

import inspect
from multiprocessing import Pool
import numpy as np
import numpy as np

def a_func(x):
    u = x
    def _another_func():
        print u
    _another_func()
    return

if __name__ == '__main__':

    a_func(3)