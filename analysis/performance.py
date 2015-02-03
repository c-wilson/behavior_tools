__author__ = 'chris'

import behavior_data_classes
import numpy as np
from numpy.lib import recfunctions
import sniff_processing


def calc_performance(behavior_epoch):
    """

    :param behavior_epoch:
    :return:
    :type behavior_epoch: behavior_data_classes.BehaviorEpoch
    """

    if 'valid_trial' in behavior_epoch.trials.dtype.names:
        old_valid_trials = behavior_epoch.trials['valid_trial']
        # other routines can only INVALIDATE trials, but cannot guarantee that they will be
        # considered valid by this routine. Any trials considered invalid here will be marked as
        # invalid regardless of the previous status. Other routines should adhere to this guideline.
    else:
        old_valid_trials = np.empty(behavior_epoch.trials['result'].shape)
        old_valid_trials.fill(True)

    results = behavior_epoch.trials['result']
    a = results > 0
    b = results < 3
    correct_array = a * b
    correct_array = correct_array.astype(bool)
    c = results < 5
    valid_trial_array = a * c * old_valid_trials
    valid_trial_array = valid_trial_array.astype(bool)
    percent_correct = np.float(np.sum(correct_array)) / np.float(np.sum(valid_trial_array))
    behavior_epoch.percent_correct = percent_correct

    assert len(correct_array) == len(behavior_epoch.trials) == len(valid_trial_array), \
        'Trial number and array lengths are not equal. Something is wrong here.'

    if not ('correct_response' and 'valid_trial') in behavior_epoch.trials.dtype.names:
        behavior_epoch.trials = recfunctions.append_fields(behavior_epoch.trials,
                                                           ['correct_response', 'valid_trial'],
                                                           [correct_array, valid_trial_array],
                                                           usemask=False)
    else:
        behavior_epoch.trials['correct_response'] = correct_array
        behavior_epoch.trials['valid_trial'] = valid_trial_array

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
        print 'no amplitude_1'
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
        # print mask_concs
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
            t_mask = mask_trials * conc_mask * lat_mask * behavior_epoch.trials['valid_trial']
            percent_correct = np.sum(behavior_epoch.trials['correct_response'] * t_mask)
            n_trials = np.sum(t_mask)
            c_dict[lat] = (percent_correct, n_trials)
        mask_performance[conc] = c_dict
        # Calculate unmasked trial performance:
        ut_mask = np.invert(mask_trials) * conc_mask * behavior_epoch.trials['valid_trial']
        umask_performance[conc] = (np.sum(ut_mask * behavior_epoch.trials['correct_response']),
                                   np.sum(ut_mask))
    behavior_epoch.mask_performace = mask_performance
    behavior_epoch.unmask_performance = umask_performance
    return


