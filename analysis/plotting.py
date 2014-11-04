__author__ = 'chris'


import matplotlib.pyplot as plt
import numpy as np
import stats


def plot_binomial(x, k, n, ci_fn='clopper_pearson', conf_interval=0.95, *args, **kwargs):
    """
    Plot binomial performance with confidence intervals from the number successes.

    :param x: array of x-axis values
    :param k: array of number of successes.
    :param n: array of number of total tests. (ie performace = k/n)
    :param args:
    :param kwargs:
    :return:
    """

    if np.isscalar(x):
        x = [x]
    if np.isscalar(k):
        k = [k]
    if np.isscalar(n):
        n = [n]
    ci_calcs = {'clopper_pearson': stats.clopper_pearson,
                'jeffreys_interval': stats.jeffreys_interval}
    try:
        ci_calc = ci_calcs[ci_fn]
    except KeyError:
        raise KeyError('Confidence interval function not found. Valid functions are: %s' % ci_calcs.keys())
    x = np.array(x)
    k = np.array(k).astype(float)
    n = np.array(n).astype(float)
    pc = k / n
    ranges = (ci_calc(k, n, conf_interval, *args, **kwargs))
    ranges = np.array(ranges)
    er_l = np.abs(ranges[0, :]-pc)
    er_h = np.abs(ranges[1, :]-pc)
    plt.errorbar(x, pc, [er_l, er_h], *args, **kwargs)


