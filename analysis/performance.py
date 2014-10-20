__author__ = 'chris'

import behavior_data_classes
import numpy as np


def calc_performance(behavior_epoch):
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
    percent_correct = np.float(np.sum(correct_array)) / np.float(np.sum(valid_trial_array))
    behavior_epoch.percent_correct = percent_correct
    behavior_epoch.correct_trial_array = correct_array
    behavior_epoch.valid_trial_array = valid_trial_array
    return percent_correct


def calc_mask_performance(behavior_epoch, separate_concentrations=True):
    """
    Adds a mask_performance attribute to the BehaviorEpoch object input.
    Format of mask_performace dictionary is : {concentration: {mask_latency: (num_correct, num_trials)}}

    :param behavior_epoch:
    :return:
    """
    calc_performance(behavior_epoch)
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
    umask_performance = {}

    if not separate_concentrations:
        mask_concs_true = mask_concs
        mask_concs = [None]

    for conc in mask_concs:
        print mask_concs
        c_dict = {}
        if separate_concentrations:
            conc_mask = odor_conc == conc
        else:
            conc_mask = np.zeros(odor_conc.shape, dtype=np.bool)
            for conc_2 in mask_concs_true:
                conc_mask = conc_mask + (odor_conc == conc_2)
                print sum(conc_mask)
        for lat in mask_latencies_unique:
            lat_mask = mask_latencies == lat
            t_mask = mask_trials * conc_mask * lat_mask * behavior_epoch.valid_trial_array
            percent_correct = np.sum(behavior_epoch.correct_trial_array * t_mask)
            n_trials = np.sum(t_mask)
            c_dict[lat] = (percent_correct, n_trials)
        mask_performance[conc] = c_dict
        # Calculate unmasked trial performance:
        ut_mask = np.invert(mask_trials) * conc_mask * behavior_epoch.valid_trial_array
        umask_performance[conc] = (np.sum(ut_mask * behavior_epoch.correct_trial_array),
                                   np.sum(ut_mask))
    behavior_epoch.mask_performace = mask_performance
    behavior_epoch.unmask_performance = umask_performance
    return
