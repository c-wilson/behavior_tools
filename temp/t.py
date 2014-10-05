__author__ = 'chris'

import inspect

class a(object):
    u = 332
    def __init__(self):
        print inspect.getfile(self.__class__)
        return

if __name__ == '__main__':
    u = a()
    print inspect.getfile(u.__class__)
