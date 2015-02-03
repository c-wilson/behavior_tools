from __future__ import division
__author__ = 'chris'

import numpy as np
import numba

def weibull(i, alpha, beta, guess, lapse):
    """

    :param i: parameter value for the stimulus (ie intensity)
    :param alpha:
    :param beta:
    :param guess:
    :param lapse:
    :return:
    """
    return ((1.-lapse)-(1.-guess-lapse) *
            np.exp(-(i/alpha)**beta))

@numba.autojit('f8[:](f8[:], f8, f8, f8, f8)')
def logistic(i, alpha, beta, guess, lapse):
    temp = (1.-lapse-guess)
    result = np.zeros(i.shape)
    for ii in xrange(len(i)):
        result[ii] = guess + temp / (1. + np.exp(-beta*(i[ii]-alpha)))
    return result

p_funcs = {'Weibull' : weibull,
           'logistic' : logistic,}