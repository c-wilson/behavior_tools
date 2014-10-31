__author__ = 'chris'


def tt(int a):
    return a**2


if __name__ == '__main__':
    r = xrange(300000)
    for i in r:
        a = tt(i)
    print a