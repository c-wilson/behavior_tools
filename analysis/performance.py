__author__ = 'chris'

import behavior_data_classes
import numpy as np
from scipy import stats



def calc_performace(behavior_epoch):
    """

    :param behavior_epoch:
    :return:
    :type behavior_epoch: behavior_data_classes.BehaviorEpoch
    """
    results = behavior_epoch.trials['result']
    a = results > 0
    b = results < 3
    correct_array = a * b
    c = results < 5
    valid_trial_array = a * c
    behavior_epoch.percent_correct = np.float(np.sum(correct_array)) / np.float(np.sum(valid_trial_array))
    behavior_epoch.correct_trial_array = correct_array
    behavior_epoch.valid_trial_array = valid_trial_array
    return


def calc_mask_performace(behavior_epoch):
    """
    Adds a mask_performance attribute to the BehaviorEpoch object input.
    Format of mask_performace dictionary is : {concentration: {mask_latency: (num_correct, num_trials)}}

    :param behavior_epoch:
    :return:
    """
    calc_performace(behavior_epoch)
    if 'amplitude_1' in behavior_epoch.trials.dtype.names:
        mask_trials = behavior_epoch.trials['amplitude_1'] > 0
    else:
        behavior_epoch.mask_performace = {}
        return
    mask_latencies = behavior_epoch.trials['pulseOnsetDelay_1']
    mask_latencies_unique = np.unique(mask_latencies)
    odor_conc = behavior_epoch.trials['odorconc']
    mask_concs = np.unique(odor_conc[mask_trials])


    behavior_epoch.mask_concs = mask_concs

    mask_performance = {}
    for conc in mask_concs:
        c_dict = {}
        conc_mask = odor_conc == conc
        for lat in mask_latencies_unique:
            lat_mask = mask_latencies == lat
            t_mask = mask_trials * conc_mask * lat_mask * behavior_epoch.valid_trial_array
            percent_correct = np.sum(behavior_epoch.correct_trial_array * t_mask)
            n_trials = np.sum(t_mask)
            c_dict[lat] = (percent_correct, n_trials)
        mask_performance[conc] = c_dict
    behavior_epoch.mask_performace = mask_performance
    return


