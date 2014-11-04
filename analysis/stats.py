__author__ = 'chris'

import numpy as np
from scipy import stats


def clopper_pearson(n_correct, n, conf_interval=0.95, *args, **kwargs):
    """
    Clopper-Pearson exact method of binomial confidence interval calculation.

    :param n_correct: number of correct trials
    :param n: number of observations total.
    :param conf_interval: Confidence interval in percent (default = 0.95 for 95% confidence interval)
    :return: low CI bound, high CI bound
    """
    alpha = 1. - conf_interval
    lo = stats.beta.ppf(alpha / 2.,
                        n_correct,
                        n - n_correct + 1)
    hi = stats.beta.isf(alpha / 2.,
                        n_correct + 1,
                        n - n_correct)
    return lo, hi


def jeffreys_interval(n_correct, n, conf_interval=0.95, *args, **kwargs):
    """
    Jeffreys method of binomial confidence interval calculation.

    :param n_correct:
    :param n:
    :param conf_interval:
    :return:
    """
    alpha = 1. - conf_interval
    lo, hi = stats.beta.interval(1 - alpha, n_correct + 0.5,
                                 n - n_correct + 0.5)
    return lo, hi
