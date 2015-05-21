from minimizer import fit_p_func, PsychometricModel
from psycho_fns import p_funcs
import numpy as np
import matplotlib.pyplot as plt



class Observer(object):
    """
    Base class for observers
    """

    # define functions that will be used for all observer instances.

    def __init__(self, stimuli=[], samples=[]):
        self.samples = stimuli
        self.stim_i = samples
        self.model = None
        self.bootstrap = {}


    def fit_p_func(self, p_func, bounds=[None, None, None, None], **kwargs):
        """
        Makes class method from static method of fit_p_func.

        :param p_func:
        :param bounds:
        :param search_grid_size:
        :return None
        """

        data = self.samples
        stim = np.array(self.stim_i, dtype=np.float)
        self.model = fit_p_func(data, stim, p_func, bounds=bounds, **kwargs)
        return self  # allows use with multiprocessing schemes.

    def plot_bins(self, binsize=50, axis=plt, *args, **kwargs):

        nt = np.sum(self.samples[:, 1])
        nbins = int(nt) / int(binsize)
        results = np.zeros((nbins, 2))
        count = 0
        n_c = 0
        i = 0
        for rx, res in zip(self.stim_i, self.samples):
            count += res[1]
            n_c += res[0]
            if count >= binsize:
                results[i, 0] = rx
                results[i, 1] = float(n_c) / float(count)
                count = 0
                n_c = 0
                i += 1
        results = results[:i, :]
        axis.plot(results[:, 0], results[:, 1], *args, **kwargs)





class SimObserver(Observer):
    """
    Simulated observer object class.
    """
    def __init__(self, alpha, beta, guess, lapse, p_func='Weibull'):
        """

        :param alpha:
        :param beta:
        :param guess:
        :param lapse:
        :param p_func: Psychometric function to use to GENERATE samples.
        :return: none
        """

        self.alpha = alpha
        self.beta = beta
        self.guess = guess
        self.lapse = lapse
        self.p_func = p_funcs[p_func]
        super(SimObserver, self).__init__()

    def sample(self, i, n):
        """
        Generates samples based on the paramaters of the observer and the intensities of the trials.

        :param i: np.array of stimulus values (ie intensities).
        :param n: np.array OR scalar int of number of samples for each stimulus value.
        :return: len(i) by 2 matrix with first column the number correct for the intensity
         and the second column the total number of samples for that trial
        """

        ret = np.zeros((len(i), 2))
        if isinstance(n,int):
            n = np.ones((len(i),1)) * n
        for idx, (stim, num) in enumerate(zip(i,n)):
            p = self.p_func(stim, self.alpha, self.beta, self.guess, self.lapse)
            ret[idx, 0] = np.random.binomial(num, p)
            ret[idx, 1] = num
        self.samples = ret
        self.stim_i = i
        return


