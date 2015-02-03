from __future__ import division
__author__ = 'chris'


import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom
from psycho_fns import p_funcs
from functools import partial

def fit_p_func(data, stim, p_func, bounds=(None, None, None, None),
               search_grid_size=50, initial_conditions=None):
    """

    :param data: 2xn array of [number correct, number trials]. Each row of the array are trials for a single stimulus
    parameter.
    :param stim: 1xn array of stimulus parameter (ie stimulus intensity) corresponding to the data array.
    :param p_func: psychometric function object or string indicating the .
    FUNCTION SHOULD EXPECT 5 VARIABLES (i, alpha, beta, guess, lapse).
    :param bounds: list or tuple of constraints for parameters order -> [alpha, beta, guess, lapse]. If constraint
    is scalar, the parameter will be treated as set (not free), elseif the constraint is a 2 member tuple,the function
    will constrain the parameter within the bounds specified (low, high). If None, the parameter is COMPLETELY un-
    constrained.
    :return: PsycometricModel object.
    """

    if type(p_func) == str:
        p_func = p_funcs[p_func]
    n = data[:,0]  # num correct.
    m = data[:,1]  # num trials total.
    if initial_conditions:  # then we don't need to search,
        search_grid_size = 1  # so lets make the grid size small so we don't allocare memory for this.

    # ----- REPARAMETERIZE a, b, g, l -----
    # based on constraints by constructing functions returning these constraints.
    # by setting some of these reparameterized functions to a constant, the minimizer will not modify them!
    # Also, generate guess space which will be used to evaluate the objective function to determine initial conditions.

    reparam_fns = []
    guess_space = []

    for bound in bounds:
        if bound is None:
            reparam_fns.append(lambda x: x)
            guess_space.append(np.linspace(np.nan_to_num(-np.inf),
                                            np.nan_to_num(np.inf),
                                            num=search_grid_size))
        elif np.isscalar(bound):
            def func_g1(constant):
                def fn(x):
                    #do nothing, return a constant.
                    return constant
                return fn
            reparam_fn = func_g1(bound)
            reparam_fns.append(reparam_fn)
            guess_space.append(np.array([0.]))
        elif len(bound) == 2:
            def func_g(high,low):
                # generate a function OBJECT with the high and low set as constants:
                def tmpfunc(x):
                    return low + (high-low)/(1.+np.exp(-x))
                def inv_tmpfunc(x):
                    return -np.log((high-low)/(x-low)-1.)
                return tmpfunc, inv_tmpfunc
            h = max(bound)
            l = min(bound)
            reparam_fn, inv_reparam_fn = func_g(h,l)
            reparam_fns.append(reparam_fn)
            guess_space.append(np.nan_to_num(inv_reparam_fn(np.linspace(h,l, num=search_grid_size))))
            # print guess_space
        else:
            print 'alpha value is not scalar or of length 2, so this cannot be preformed.'
            #TODO: raise exemption
            return



    # ----- GENERATE OBJECTIVE FUNCTION ------
    # first, generate a negative log likelihood objective function with a function generator:
    def ob_gen(stim_i, model_params):
        # model parameters here are put into an objective function as CONSTANTS
        alpha = model_params[0]
        beta  = model_params[1]
        guess = model_params[2]
        lapse = model_params[3]
        def ob_fun((a, b, g, l)):  # expects a tuple "x" from the minimizer
            res=p_func(stim_i, alpha(a), beta(b), guess(g), lapse(l))
            return -np.sum(n * np.nan_to_num(np.log(binom.pmf(n, m, res))) +
                           (m-n) * np.nan_to_num(np.log(1.-binom.pmf(n, m, res))))
        return ob_fun
    # then create an instance of the objective function and minimize.
    nll = ob_gen(stim, reparam_fns)

    # ----- GENERATE START CONDITIONS -----
    # use brute force to find start conditions (x0) that nearly maximize the objective function.
    #
    x0 = []
    if not initial_conditions:
        nll_mat = np.zeros((guess_space[0].size,
                            guess_space[1].size,
                            guess_space[2].size,
                            guess_space[3].size))

        for i, a in enumerate(guess_space[0]):
            for j, b in enumerate(guess_space[1]):
                for k, g in enumerate(guess_space[2]):
                    for ii, l in enumerate(guess_space[3]):
                        nll_mat[i,j,k,ii] = nll((a, b, g, l))

        guess_idxes = np.where(nll_mat == np.min(nll_mat))

        # #### DEBUG ######
        # import matplotlib.pyplot as plt
        # import matplotlib
        # plt.pcolormesh(guess_space[0] ,guess_space[1], np.clip(-np.nan_to_num(nll_mat[:,:,0,0]),0 ,5e4, ),
        #                norm=matplotlib.colors.LogNorm())
        # #### DEBUG ######

        for g_ax, idx, rp_fn in zip(guess_space, guess_idxes, reparam_fns):
            x0.append(g_ax[idx][0])
    else:
        x0 = initial_conditions
    # print x0

    # ----- MINIMIZE OBJECTIVE FUNCTION -----
    res = minimize(nll, x0, method='Nelder-Mead', options={'maxiter':int(1e9), 'maxfev':int(1e4)})
    return PsychometricModel(p_func,
                             alpha=reparam_fns[0](res.x[0]),
                             beta=reparam_fns[1](res.x[1]),
                             guess=reparam_fns[2](res.x[2]),
                             lapse=reparam_fns[3](res.x[3]),
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

        if np.isscalar(i):
            i = np.array([i])
        elif isinstance(i, list) or isinstance(i, tuple):
            i = np.array(i)
        p_func = self.p_func
        ret = p_func(i, self.alpha, self.beta, self.guess, self.lapse)
        return ret

    def __str__(self):
        return ("Psycometric model fit to %s function. \nParameters: alpha: %.4f, beta: %.4f, gamma: %.4f, lambda: %.4f"
                % (self.p_func, self.alpha, self.beta, self.guess, self.lapse))


