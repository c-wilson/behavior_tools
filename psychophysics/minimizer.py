__author__ = 'chris'


import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom


def fit_p_func(observer, p_func, bounds=[None, None, None, None]):
    """

    :param observer: an Observer object with a samples attribute.
    :param p_func: psychometric function object. SHOULD EXPECT 5 VARIABLES (i, alpha, beta, guess, lapse).
    :param constraints: list or tuple of constraints for parameters order -> [alpha, beta, guess, lapse]. If constraint
    is numeric, the parameter will be treated as set (not free), elseif the constraint is a 2 member tuple,the function
    will constrain the parameter within the bounds specified (low, high). If None, the parameter is COMPLETELY un-
    constrained.
    :return: PsycometricModel object.
    """

    data = observer.samples
    stim_i = observer.stim_i
    n = data[:,0]  # num correct.
    m = data[:,1]  # num trials total.

    # ----- REPARAMETERIZE a, b, g, l -----
    # based on constraints by constructing functions returning these constraints.
    # by setting some of these reparameterized functions to a constant, the minimizer will not modify them!
    # Also, generate guess space which will be used to do some sweet math later.

    reparam_fns = []
    guess_space = []
    guess_grid_size = 200
    for bound in bounds:
        if bound is None:
            reparam_fns.append(lambda x: x)
            guess_space.append(np.linspace(np.nan_to_num(-np.inf),
                                            np.nan_to_num(np.inf),
                                            num=guess_grid_size))
        elif np.isscalar(bound):
            def func_g1(constant):
                def fn(x):
                    #do nothing, return a constant.
                    return constant
                return fn
            reparam_fn = func_g1(bound)
            reparam_fns.append(reparam_fn)
            guess_space.append(np.array([0]))
        elif len(bound) == 2:
            def func_g(high,low):
                # generate a function OBJECT with the high and low set as constants:
                def tmpfunc(x):
                    return low + (high-low)/(1+np.exp(-x))
                def inv_tmpfunc(x):
                    return -np.log((high-low)/(x-low)-1)
                return tmpfunc, inv_tmpfunc
            h = max(bound)
            l = min(bound)
            reparam_fn, inv_reparam_fn = func_g(h,l)
            reparam_fns.append(reparam_fn)
            guess_space.append(np.nan_to_num(inv_reparam_fn(np.linspace(h,l, num=guess_grid_size))))
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
            return -np.sum(n * np.nan_to_num(np.log(binom.pmf(n, m, p_func(stim_i, alpha(a), beta(b), guess(g), lapse(l))))) +
                           (m-n) * np.nan_to_num(np.log(1-binom.pmf(n, m, p_func(stim_i, alpha(a),
                                                                                 beta(b), guess(g), lapse(l))))))
        return ob_fun
    # then create an instance of the objective function and minimize.
    nll = ob_gen(stim_i, reparam_fns)

    # ----- GENERATE START CONDITIONS -----
    # use brute force to find start conditions (x0) that nearly maximize the objective function.
    #

    nll_mat = np.zeros((guess_space[0].size,
                        guess_space[1].size,
                        guess_space[2].size,
                        guess_space[3].size))

    for i, a in enumerate(guess_space[0]):
        for j, b in enumerate(guess_space[1]):
            for k, g in enumerate(guess_space[2]):
                for ii, l in enumerate(guess_space[3]):
                    nll_mat[i,j,k,ii] = nll((a,b,g,l))

    guess_idxes = np.where(nll_mat == np.min(nll_mat))
    x0 = []
    for g_ax, idx, rp_fn in zip(guess_space, guess_idxes, reparam_fns):
        x0.append(g_ax[idx][0])


    # ----- MINIMIZE OBJECTIVE FUNCTION -----
    res = minimize(nll, x0, method='Nelder-Mead', options={'maxiter':int(1e9), 'maxfev':int(1e4)})

    observer.model = PsychometricModel(p_func,
                                       alpha=reparam_fns[0](res.x[0]),
                                       beta=reparam_fns[1](res.x[1]),
                                       guess=reparam_fns[2](res.x[2]),
                                       lapse=reparam_fns[3](res.x[3]),
                                       bounds=bounds,
                                       minimize_result=res)
    return observer.model


class PsychometricModel(object):
    def __init__(self, p_func, alpha, beta, guess, lapse, bounds, minimize_result=None):
        self.p_func = p_func
        self.alpha = alpha
        self.beta = beta
        self.guess = guess
        self.lapse = lapse
        self.result = minimize_result
        self.bounds = bounds

    def evaluate(self, i):
        """
        Evaluates the psycometric function with the embedded parameters at an array of stimulus intensities.

        :param i: np.array of stimulus parameters with which to evaluate the model.
        :return: np.array of predicted performance values
        """

        if isinstance(i, int):
            i = np.array([i])
        ret = np.zeros(i.shape)
        p_func = self.p_func
        for idx, stim in enumerate(i):
            ret[idx] = p_func(stim, self.alpha, self.beta, self.guess, self.lapse)
        return ret