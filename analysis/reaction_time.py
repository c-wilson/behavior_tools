__author__ = 'chris'

import numpy as np
import matplotlib.pyplot as plt

def reaction_time_trial(behavior_trial):
    """
    Finds the first lick start following the first inhalation after stimulus onset.

    :param behavior_trial: a behavioral trial.
    :return: np.float64
    """

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


def calc_rxn_time(behavior_epoch):
    """

    Simple wrapper for reaction_time_trial function that runs it for all trials and creates a reaction_times attribute
    for the behavior epoch.

    :param behavior_epoch:
    :return: None
    """

    n_trials = behavior_epoch.trials.size
    reaction_times = []
    first_sniffs =[]
    for i in range(n_trials):
        trial = behavior_epoch.return_trial(i)
        if trial:
            rt, fs = reaction_time_trial(trial)
            reaction_times.append(rt)
            first_sniffs.append(fs)
        else:
            reaction_times.append(np.NaN)
            first_sniffs.append(np.NaN)
    behavior_epoch.reaction_times = np.array(reaction_times)
    behavior_epoch.first_sniffs = np.array(first_sniffs)



def plot_rxn(behavior_epochs):
    c_correct = np.array([], dtype=np.bool)
    _rxn = np.array([],dtype=np.float64)

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
    plt.plot(rxn, correct, '.')
    return c_rxn, correct




