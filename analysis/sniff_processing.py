__author__ = 'chris'

import behavior_data_classes
import numpy as np


def make_sniff_events(behavior_epoch, inh_threshold=20, inverted=True, min_amplitude=200, **kwargs):
    """

    :param behavior_epoch: BehaviorEpoch
    :param inh_threshold: Threshold value for definining inhalation vs exhalation
    :param inverted: if True (default) inhalation is positive.
    :param min_amplitude: define minimum amplitude for sniff.
    :return:
    """

    sniff = behavior_epoch.streams['sniff'].read()

    if inverted:
        sniff = -sniff

    # check that minimum amplitude is reached.
    if np.max(sniff) < min_amplitude or np.min(sniff) > -min_amplitude:
        return np.array([]), np.array([])
    else:
        sniff_logical = sniff < inh_threshold
        edges = np.convolve([1, -1], sniff_logical, mode='same')  # returns array of 1s and -1s for logic changes
        up_edges = np.where(edges == 1)[0]  # returns tuple, must take first member from tuple
        down_edges = np.where(edges == -1)[0]

    if inverted:
        behavior_epoch.events['sniff_inh'] = down_edges
        behavior_epoch.events['sniff_exh'] = up_edges
    else:
        behavior_epoch.events['sniff_inh'] = up_edges
        behavior_epoch.events['sniff_exh'] = down_edges

    behavior_epoch.sniff_threshold = inh_threshold
