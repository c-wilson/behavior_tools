__author__ = 'chris'

import numpy as np

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

def logistic(i, alpha, beta, guess, lapse):
    return (guess+(1.-lapse-guess)/
            (1.+np.exp(-beta*(i-alpha))))



p_funcs = {'Weibull' : weibull,
           'logistic' : logistic,}