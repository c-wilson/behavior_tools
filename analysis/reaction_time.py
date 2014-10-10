__author__ = 'chris'

import numpy as np
import matplotlib.pyplot as plt
from .. import psychophysics
import sniff_processing


def rxn_time_trial(behavior_trial):
    """
    Finds the first lick start following the first inhalation after stimulus onset.

    :param behavior_trial: a behavioral trial.
    :return: np.float64
    """

    sniff = behavior_trial.streams['sniff']
    # check that sniff is good:  check max > 200, min < -200, no breaks:
    if np.max(sniff) < 200 or np.min(sniff) > -200 or np.sum(sniff == 0) > 399:
        return np.NaN, np.NaN
    # FIND first inhalation following stimulus onset:
    stim_on = behavior_trial.trials['fvOnTime']
    sniff_on = behavior_trial.events['sniff_inh']
    l = sniff_on > stim_on[0]  # returns all the inhalation onsets after stimulus onset.
    sniff_starts = sniff_on[l]  #all the timestamps of inhalation starts after stim onset.
    if sniff_starts.any():
        first_sniff = sniff_starts[0]  # returns only the first sniff time stamp.
    else:
        return np.NaN, np.NaN

    # FIND first lick after first sniff:
    licks = []
    first_licks = []
    licks.append(behavior_trial.events['lick1'])
    licks.append(behavior_trial.events['lick2'])

    for lick in licks:
        l = lick > first_sniff
        if l.any():
            first_licks.append(lick[l][0])
        else:
            first_licks.append(np.NaN)

    return np.nanmin(first_licks) - first_sniff, first_sniff


def rxn_time_epoch(behavior_epoch):
    """

    Simple wrapper for rxn_time_trial function that runs it for all trials and creates a reaction_times attribute
    for the behavior epoch.

    :param behavior_epoch:
    :return: None
    """
    if not 'sniff_inh' in behavior_epoch.events.keys():
        sniff_processing.make_sniff_events(behavior_epoch)
    n_trials = behavior_epoch.trials.size
    reaction_times = []
    first_sniffs =[]
    for i in range(n_trials):
        trial = behavior_epoch.return_trial(i)
        if trial:
            rt, fs = rxn_time_trial(trial)
            reaction_times.append(rt)
            first_sniffs.append(fs)
        else:
            reaction_times.append(np.NaN)
            first_sniffs.append(np.NaN)
    behavior_epoch.reaction_times = np.array(reaction_times)
    behavior_epoch.first_sniffs = np.array(first_sniffs)


def plot_rxn(behavior_epochs):
    """

    :param behavior_epochs:
    :return:
    """
    c_correct = np.array([], dtype=np.bool)
    c_rxn = np.array([],dtype=np.float64)
    for epoch in behavior_epochs:
        rxn = epoch.reaction_times
        result = epoch.trials['result']
        valid = (result <5) * (result > 0)
        correct = result < 3
        rxn = rxn[valid]
        correct = correct[valid]
        c_rxn = np.concatenate([c_rxn, rxn])
        c_correct = np.concatenate([c_correct, correct])
    print sum(np.invert(c_correct))
    print sum(c_correct)
    plt.plot(c_rxn, c_correct, '.')
    return c_rxn, c_correct


def _sort_rxn(behavior_epochs):
    """

    :param behavior_epochs:
    :return:
    """
    c_correct = np.array([], dtype=np.bool)
    c_rxn = np.array([],dtype=np.float64)
    for epoch in behavior_epochs:
        if not hasattr(epoch, 'reaction_times'):
            rxn_time_epoch(epoch)
        rxn = epoch.reaction_times
        result = epoch.trials['result']
        valid = (result <5) * (result > 0)
        correct = result < 3
        rxn = rxn[valid]
        correct = correct[valid]
        c_rxn = np.concatenate([c_rxn, rxn])
        c_correct = np.concatenate([c_correct, correct])
    # combine, sort by rxn time, and break back into two arrays.
    c = np.hstack((c_rxn[:,np.newaxis],
                          c_correct[:,np.newaxis]))
    c = c[c[:,0].argsort()]
    rx = c[:,0]
    cor = c[:,1]
    # remove nans from this stuff:
    not_nans = ~np.isnan(rx)
    not_nans = ~np.isnan(cor) * not_nans
    rx = rx[not_nans]
    cor = cor[not_nans]
    # Collapse duplicate observations:
    u_r = np.unique(rx)
    rx2 = np.zeros(u_r.size)
    res = np.zeros((u_r.size,2))
    for i, r in enumerate(u_r):
        idx = rx==r
        n_obs = np.sum(idx)
        n_cor = np.sum(cor[idx])
        res[i,:] = n_cor, n_obs
        rx2[i] = r

    return rx2, res


def rxn_make_observer(behavior_epochs):
    """
    Helper function to parse reaction time data from behavior epochs.

    :param behavior_epochs:
    :return:
    """
    rxn, cor = _sort_rxn(behavior_epochs)
    # cor = cor[:, np.newaxis]
    # dat = np.hstack((cor, np.ones(cor.shape)))
    dat = cor
    obs = psychophysics.Observer(dat, rxn)
    return obs


def rxn_concentration_make_observers(behavior_epochs):
    concentrations = []
    conc = np.array([])
    for b in behavior_epochs:
        conc = np.concatenate((conc, b.trials['odorconc']))
    conc = np.unique(conc)

    observers = []

    for con in conc:
        c_correct = np.array([], dtype=np.bool)
        c_rxn = np.array([],dtype=np.float64)
        for epoch in behavior_epochs:
            if not hasattr(epoch, 'reaction_times'):
                rxn_time_epoch(epoch)
            rxn = epoch.reaction_times
            result = epoch.trials['result']
            concentrations = epoch.trials['odorconc']
            valid = (result <5) * (result > 0) * (concentrations == con)
            correct = result < 3
            rxn = rxn[valid]
            correct = correct[valid]
            c_rxn = np.concatenate([c_rxn, rxn])
            c_correct = np.concatenate([c_correct, correct])
        # combine, sort by rxn time, and break back into two arrays.
        c = np.hstack((c_rxn[:,np.newaxis],
                              c_correct[:,np.newaxis]))
        c = c[c[:,0].argsort()]
        rx = c[:,0]
        cor = c[:,1]
        # remove nans from this stuff:
        not_nans = ~np.isnan(rx)
        not_nans = ~np.isnan(cor) * not_nans
        rx = rx[not_nans]
        cor = cor[not_nans]
        # Collapse duplicate observations:
        u_r = np.unique(rx)
        rx2 = np.zeros(u_r.size)
        res = np.zeros((u_r.size,2))
        for i, r in enumerate(u_r):
            idx = rx==r
            n_obs = np.sum(idx)
            n_cor = np.sum(cor[idx])
            res[i,:] = n_cor, n_obs
            rx2[i] = r
        obs = psychophysics.Observer(res, rx2)
        obs.concentration = con
        observers.append(obs)
    return observers






