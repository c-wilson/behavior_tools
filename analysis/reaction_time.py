__author__ = 'chris'

import numpy as np
import matplotlib.pyplot as plt
from .. import psychophysics
import sniff_processing
from numpy.lib import recfunctions


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
    sniff_starts = sniff_on[l]  # all the timestamps of inhalation starts after stim onset.
    if sniff_starts.any():
        first_sniff = sniff_starts[0]  # returns only the first sniff time stamp.
    else:
        return np.NaN, np.NaN

    # FIND first lick after first sniff:
    licks = []
    first_licks = []
    licks.append(behavior_trial.events['lick1'])
    if 'lick2' in behavior_trial.events.keys():
        licks.append(behavior_trial.events['lick2'])

    for lick in licks:
        l = lick > first_sniff
        if l.any():
            first_licks.append(lick[l][0])
        else:
            first_licks.append(np.NaN)

    return np.nanmin(first_licks) - first_sniff, first_sniff


def rxn_time_epoch(behavior_epoch, **kwargs):
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
    first_sniffs = []
    for i in range(n_trials):
        trial = behavior_epoch.return_trial(i)
        if trial:
            rt, fs = rxn_time_trial(trial)
            reaction_times.append(rt)
            first_sniffs.append(fs)
        else:
            reaction_times.append(np.NaN)
            first_sniffs.append(np.NaN)

    assert behavior_epoch.trials.size == len(reaction_times) == len(first_sniffs), \
        'Calculated arrays did not equal the number of trials in trials array.'

    first_sniffs = np.array(first_sniffs, dtype=np.float)
    reaction_times = np.array(reaction_times, dtype=np.float)

    if not ('reaction_time' and 'first_stim_sniff') in behavior_epoch.trials.dtype.names:
        behavior_epoch.trials = recfunctions.append_fields(behavior_epoch.trials,
                                                           ['reaction_time', 'first_stim_sniff'],
                                                           [reaction_times, first_sniffs],
                                                           usemask=False)
    else:
        behavior_epoch.trials['reaction_time'] = reaction_times
        behavior_epoch.trials['first_stim_sniff'] = first_sniffs


def plot_rxn(behavior_epochs):
    """

    :param behavior_epochs:
    :return:
    """

    c_correct = np.array([], dtype=np.bool)
    c_rxn = np.array([], dtype=np.float64)
    for epoch in behavior_epochs:
        rxn = epoch.trials['reaction_time']
        result = epoch.trials['result']
        valid = epoch.trials['valid_trial']
        correct = result < 3
        rxn = rxn[valid]
        correct = correct[valid]
        c_rxn = np.concatenate([c_rxn, rxn])
        c_correct = np.concatenate([c_correct, correct])
    print sum(np.invert(c_correct))
    print sum(c_correct)
    plt.plot(c_rxn, c_correct, '.')
    return c_rxn, c_correct


def _sort_rxn(behavior_epochs, skip_first=20):
    """

    :param behavior_epochs:
    :param skip_first: number of initial trials to skip within a session.
    :return:
    """
    c_correct = np.array([], dtype=np.bool)
    c_rxn = np.array([], dtype=np.float64)
    for epoch in behavior_epochs:
        trials = epoch.trials[skip_first:]
        # if not hasattr(epoch, 'reaction_times'):
        if not 'reaction_time' in trials.dtype.names:
            rxn_time_epoch(epoch)
        rxn = trials['reaction_time']
        result = trials['result']
        valid = (result < 5) * (result > 0)
        correct = result < 3
        rxn = rxn[valid]
        correct = correct[valid]
        c_rxn = np.concatenate([c_rxn, rxn])
        c_correct = np.concatenate([c_correct, correct])
    # combine, sort by rxn time, and break back into two arrays.
    c = np.hstack((c_rxn[:, np.newaxis],
                   c_correct[:, np.newaxis]))
    c = c[c[:, 0].argsort()]
    rx = c[:, 0]
    cor = c[:, 1]
    # remove nans from this stuff:
    not_nans = ~np.isnan(rx)
    not_nans = ~np.isnan(cor) * not_nans
    rx = rx[not_nans]
    cor = cor[not_nans]
    # Collapse duplicate observations:
    u_r = np.unique(rx)
    rx2 = np.zeros(u_r.size)
    res = np.zeros((u_r.size, 2))
    for i, r in enumerate(u_r):
        idx = rx == r
        n_obs = np.sum(idx)
        n_cor = np.sum(cor[idx])
        res[i, :] = n_cor, n_obs
        rx2[i] = r
    # get rid of very long reaction time trials:
    res = res[rx2 < 250, :]
    rx2 = rx2[rx2 < 250]

    return rx2, res


def rxn_make_observer(behavior_epochs, **kwargs):
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


def rxn_first_lick_make_observer(behavior_epochs, skip_first=20, **kwargs):
    c_correct = np.array([], dtype=np.bool)
    c_rxn = np.array([], dtype=np.float64)
    for epoch in behavior_epochs:
        trials = epoch.trials[skip_first:]
        # if not hasattr(epoch, 'reaction_times'):
        if not 'reaction_time' in trials.dtype.names:
            rxn_time_epoch(epoch)
        rxn = trials['first_lick_time']
        result = trials['first_lick_correct']
        valid = trials['valid_trial']
        rxn = rxn[valid]
        correct = result[valid]
        c_rxn = np.concatenate([c_rxn, rxn])
        c_correct = np.concatenate([c_correct, correct])
    # combine, sort by rxn time, and break back into two arrays.
    c = np.hstack((c_rxn[:, np.newaxis], c_correct[:, np.newaxis]))
    c = c[c[:, 0].argsort()]
    rx = c[:, 0]
    cor = c[:, 1]
    # remove nans from this stuff:
    not_nans = ~np.isnan(rx)
    not_nans = ~np.isnan(cor) * not_nans
    rx = rx[not_nans]
    cor = cor[not_nans]
    # Collapse duplicate observations:
    u_r = np.unique(rx)
    rx2 = np.zeros(u_r.size)
    res = np.zeros((u_r.size, 2))
    for i, r in enumerate(u_r):
        idx = rx == r
        n_obs = np.sum(idx)
        n_cor = np.sum(cor[idx])
        res[i, :] = n_cor, n_obs
        rx2[i] = r
    # get rid of very long reaction time trials:
    # res = res[rx2 < 250, :]
    # rx2 = rx2[rx2 < 250]

    return psychophysics.Observer(res, rx2)


def rxn_concentration_make_observers(behavior_epochs, skip_first=20):
    concentrations = []
    conc = np.array([])
    for b in behavior_epochs:
        conc = np.concatenate((conc, b.trials['odorconc']))
    conc = np.unique(conc)

    observers = []

    for con in conc:

        c_correct = np.array([], dtype=np.bool)
        c_rxn = np.array([], dtype=np.float64)
        for epoch in behavior_epochs:
            trials = epoch.trials[skip_first:]
            if not hasattr(epoch, 'reaction_times'):
                rxn_time_epoch(epoch)
            rxn = epoch.reaction_times[skip_first:]
            result = trials['result']
            concentrations = trials['odorconc']
            valid = (result < 5) * (result > 0) * (concentrations == con)
            correct = result < 3
            rxn = rxn[valid]
            correct = correct[valid]
            c_rxn = np.concatenate([c_rxn, rxn])
            c_correct = np.concatenate([c_correct, correct])
        # combine, sort by rxn time, and break back into two arrays.
        c = np.hstack((c_rxn[:, np.newaxis],
                       c_correct[:, np.newaxis]))
        c = c[c[:, 0].argsort()]
        rx = c[:, 0]
        cor = c[:, 1]
        # remove nans from this stuff:
        not_nans = ~np.isnan(rx)
        not_nans = ~np.isnan(cor) * not_nans
        rx = rx[not_nans]
        cor = cor[not_nans]
        # Collapse duplicate observations:
        u_r = np.unique(rx)
        rx2 = np.zeros(u_r.size)
        res = np.zeros((u_r.size, 2))
        for i, r in enumerate(u_r):
            idx = rx == r
            n_obs = np.sum(idx)
            n_cor = np.sum(cor[idx])
            res[i, :] = n_cor, n_obs
            rx2[i] = r
        res = res[rx2 < 250, :]
        rx2 = rx2[rx2 < 250]
        obs = psychophysics.Observer(res, rx2)
        obs.concentration = con
        observers.append(obs)

    return observers


def rxn_first_lick_concentration_make_observers(behavior_epochs, skip_first=20):
    concentrations = []
    conc = np.array([])
    for b in behavior_epochs:
        conc = np.concatenate((conc, b.trials['odorconc']))
    conc = np.unique(conc)

    observers = []

    for con in conc:

        c_correct = np.array([], dtype=np.bool)
        c_rxn = np.array([], dtype=np.float64)
        for epoch in behavior_epochs:
            trials = epoch.trials[skip_first:]
            # if not 'first_lick_time' in trials.dtype.names:
            # calc_correct_first_lick(epoch)
            concentrations = trials['odorconc']
            rxn = trials['first_lick_time']
            correct = trials['first_lick_correct']
            valid = trials['valid_trial'] * (concentrations == con)

            rxn = rxn[valid]
            correct = correct[valid]
            c_rxn = np.concatenate([c_rxn, rxn])
            c_correct = np.concatenate([c_correct, correct])
        # combine, sort by rxn time, and break back into two arrays.
        c = np.hstack((c_rxn[:, np.newaxis],
                       c_correct[:, np.newaxis]))
        c = c[c[:, 0].argsort()]
        rx = c[:, 0]
        cor = c[:, 1]
        # remove nans from this stuff:
        not_nans = ~np.isnan(rx)
        not_nans = ~np.isnan(cor) * not_nans
        rx = rx[not_nans]
        cor = cor[not_nans]
        # Collapse duplicate observations:
        u_r = np.unique(rx)
        rx2 = np.zeros(u_r.size)
        res = np.zeros((u_r.size, 2))
        for i, r in enumerate(u_r):
            idx = rx == r
            n_obs = np.sum(idx)
            n_cor = np.sum(cor[idx])
            res[i, :] = n_cor, n_obs
            rx2[i] = r
        res = res[rx2 < 250, :]
        rx2 = rx2[rx2 < 250]
        obs = psychophysics.Observer(res, rx2)
        obs.concentration = con
        observers.append(obs)

    return observers


def calc_correct_first_lick(behavior_epoch, **kwargs):
    """

    :param behavior_epoch:
    :return:
    :type behavior_epoch: behavior_data_classes.BehaviorEpoch
    """
    correct_on_first_lick_list = []
    first_lick_time_list = []

    if not 'first_stim_sniff' in behavior_epoch.trials.dtype.names:
        if not behavior_epoch.events.has_key('sniff_inh'):
            sniff_processing.make_sniff_events(behavior_epoch, **kwargs)
        rxn_time_epoch(behavior_epoch, **kwargs)

    for tr_idx in range(len(behavior_epoch.trials)):
        trial = behavior_epoch.return_trial(tr_idx)
        if not trial:
            correct_on_first_lick_list.append(False)
            first_lick_time_list.append(np.inf)
            continue
        # result = trial.trials['result']
        trial_type = trial.trials['trialtype']
        lick1 = trial.events['lick1']
        lick2 = trial.events['lick2']
        first_sniff_time = trial.trials['first_stim_sniff']

        if trial_type == 1:
            # correct is lick2, incorrect is lick 1
            if np.any(lick2):
                first_correct_lick_time = lick2[0, 0]
            else:
                first_correct_lick_time = np.inf

            if np.any(lick1):
                first_incorrect_lick_time = lick1[0, 0]
            else:
                first_incorrect_lick_time = np.inf

        elif trial_type == 0:
            if np.any(lick1):
                first_correct_lick_time = lick1[0, 0]
            else:
                first_correct_lick_time = np.inf
            if np.any(lick2):
                first_incorrect_lick_time = lick2[0, 0]
            else:
                first_incorrect_lick_time = np.inf

        else:
            raise ValueError('Trial type is not valid.')

        if first_correct_lick_time < first_incorrect_lick_time:
            correct_on_first_lick_list.append(True)
            first_lick_time_list.append(first_correct_lick_time - first_sniff_time[0])
        else:
            correct_on_first_lick_list.append(False)
            first_lick_time_list.append(first_incorrect_lick_time - first_sniff_time[0])

    assert len(correct_on_first_lick_list) == len(behavior_epoch.trials) == len(first_lick_time_list), \
        'Number of trials and number of calculated values are not consistent, something is wrong.'

    correct_on_first_lick_list = np.array(correct_on_first_lick_list, dtype=np.float)
    first_lick_time_list = np.array(first_lick_time_list, dtype=np.float)

    if not ('first_lick_correct' and 'first_lick_time') in behavior_epoch.trials.dtype.names:
        behavior_epoch.trials = recfunctions.append_fields(behavior_epoch.trials,
                                                           ['first_lick_correct', 'first_lick_time'],
                                                           [correct_on_first_lick_list, first_lick_time_list],
                                                           usemask=False)
    else:
        behavior_epoch.trials['first_lick_correct'] = correct_on_first_lick_list
        behavior_epoch.trials['first_lick_time'] = first_lick_time_list