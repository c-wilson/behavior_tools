from __future__ import division
__author__ = 'chris'


import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom
from psycho_fns import p_funcs
from functools import partial
from scipy.misc import logsumexp

def fit_p_func(data, stim, p_func, bounds=((0, None), (0, None), (0., 1.), (0., 1.)),
               x0=None, *args, **kwargs):
    """

    :param data: 2xn array of [number correct, number trials]. Each row of the array are trials for a single stimulus
    parameter.
    :param stim: 1xn array of stimulus parameter (ie stimulus intensity) corresponding to the data array.
    :param p_func: psychometric function object or string indicating the function to use (ie 'logistic' or 'Weibull').
    FUNCTION SHOULD EXPECT 5 VARIABLES (i, alpha, beta, guess, lapse).
    :param bounds: list or tuple of constraints for parameters with order ->
                   [(alpha_low, alpha_high), (beta_low, beta_high), (guess_low, guess_high), (lapse_low, lapse_high)].
                    if any bound is not needed, use None. For example [(0, None), ... ] will have a minimum bound at 0
                    and no maximum bound for the alpha parameter.
    :param x0: List of initial conditions for optimizer ([alpha, beta, gamma, lapse]). Defaults to [100, 6, .5, 1].
    :return: PsycometricModel object.
    """

    default_x0 = (100, 6, .5, .1)

    if type(p_func) == str:
        p_func = p_funcs[p_func]
    n = data[:, 0]  # num correct.
    m = data[:, 1]  # num trials total.

    # ----- GENERATE OBJECTIVE FUNCTION ------
    # first, generate a negative log likelihood objective function with a function generator:

    def nll((a, b, g, l)):  # expects a tuple "x" from the minimizer
        """
        negative log likelihood function.
        """

        res = p_func(stim, a, b, g, l)
        # p = np.nan_to_num(binom.pmf(n, m, res))
        # log_p = np.nan_to_num(np.log(p))  # underflow of 'p' causes this to go to -infinity, which I hate.

        return -np.sum(binom.logpmf(n, m , res))
    # then create an instance of the objective function and minimize.

    # ----- Make initial guess ---------

    if x0 is None:
        x0 = np.zeros(4)
        for i in xrange(4):
            bound = bounds[i]
            if bound is None:
                x0[i] = default_x0[i]
            else:
                l = bound[0]
                h = bound[1]
                if l is None:
                    x0[i] = h
                elif h is None:
                    x0[i] = l
                else:
                    if x0[i] > l and x0[i] < h:
                        x0[i] = default_x0[i]
                    else:
                        x0[i] = l + (h - l)/2.

    # ----- MINIMIZE OBJECTIVE FUNCTION -----
    res = minimize(nll, x0, bounds=bounds, *args, **kwargs)

    if res.success:
        alpha, beta, gamma, lamb = res.x
    else:
        alpha = beta = gamma = lamb = None

    return PsychometricModel(p_func,
                             alpha=alpha,
                             beta=beta,
                             guess=gamma,
                             lapse=lamb,
                             x0=x0,
                             bounds=bounds,
                             minimize_result=res)


def bootstrap_analysis(observer, n_samples=20000):
    """

    :param observer:
    :param n_samples:
    :return:
    :type observer: observers.Observer
    """

    #parse this into a tuple for the _draw_and_fit function below:
    if not hasattr(observer, 'model'):
        print 'Observer object must have compute model prior to bootstrapping.'
        return


    sub_func = partial(_draw_and_fit, observer=observer)

    # p = Pool(4)

    # models = p.map(sub_func, range(n_samples))

    models = map(sub_func, range(n_samples))

    alphas = []
    betas = []
    gammas = []
    lambdas = []
    for mod in models:
        alphas.append(mod.alpha)
        betas.append(mod.beta)
        gammas.append(mod.guess)
        lambdas.append(mod.lapse)
    alphas = np.array(alphas)
    betas = np.array(betas)
    gammas = np.array(gammas)
    lambdas = np.array(lambdas)
    raw_results = {'alpha': alphas,
                   'beta': betas,
                   'gamma': gammas,
                   'lamda': lambdas}
    means = {}
    variances = {}
    standard_deviations = {}
    confidences={}
    for k, v in raw_results.iteritems():
        means[k] = np.mean(v)
        variances[k] = np.var(v)
        standard_deviations[k] = np.std(v)
        confidences[k] = np.percentile(v, (2.5, 97.5))
    results = {'raw_results': raw_results,
               'means': means,
               'variances': variances,
               'sd':standard_deviations,
               'CI_95': confidences}
    observer.bootstrap = results
    return observer



def _plot_bootstrap(observer, evaluation_range, parameter):
    # TODO: IMPLEMENT THIS.

    """

    :param observer:
    :param evaluate:
    :param parameter:
    :return:
    """

    try:
        results = observer.model.evaluate(evaluation_range)
    except AttributeError:
        raise AttributeError('The observer has no psychometric model. Please run fit_p_func to fit a model.')


    try:
        value = observer.bootstrap
    except:
        pass




def _draw_and_fit(_iter, observer):
    """

    :param stim:
    :param p:
    :param num_trials:
    :return:
    """
    stims = observer.stim_i
    samples = observer.samples
    bounds = observer.model.bounds
    p = samples[:,0] / samples[:,1]
    n = samples [:,1]
    ic = observer.model.x0


    bs_data = np.zeros(samples.shape)

    for i, lvl in enumerate(samples):
        n = lvl[1]
        p = lvl[0] / n
        bs_data[i,0] = np.random.binomial(n, p)
        bs_data[i,1] = n
    return fit_p_func(bs_data, stims, observer.model.p_func, bounds=bounds, initial_conditions=ic)


class PsychometricModel(object):
    def __init__(self, p_func, alpha, beta, guess, lapse, x0, bounds, minimize_result=None):
        self.p_func = p_func
        self.alpha = alpha
        self.beta = beta
        self.guess = guess
        self.lapse = lapse
        self.result = minimize_result
        self.bounds = bounds
        self.x0 = x0

    def evaluate(self, i):
        """
        Evaluates the psycometric function with the embedded parameters at an array of stimulus intensities.

        :param i: np.array of stimulus parameters with which to evaluate the model.
        :return: np.array of predicted performance values
        """

        for param in [self.alpha, self.beta, self.guess, self.lapse]:
            if param is None:
                print 'Parameter is set to None, model did not converge!'
                a = np.empty(len(i))
                a[:] = np.nan
                return a

        if np.isscalar(i):
            i = np.array([i])
        else:
            i = np.asarray(i)
        p_func = self.p_func
        ret = p_func(i, self.alpha, self.beta, self.guess, self.lapse)
        return ret

    def __str__(self):
        return ("Psycometric model fit to %s function. \nParameters: alpha: %.4f, beta: %.4f, gamma: %.4f, lambda: %.4f"
                % (self.p_func.func_name, self.alpha, self.beta, self.guess, self.lapse))


