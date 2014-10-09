__author__ = 'chris'

import behavior_data_classes
import numpy as np

def sniff_thresholding(sniff, threshold=-20, **kwargs):
    """
    Rudamentary sniff analysis. Compares sniff from behavior epoch with threshold.
    Assumes that sniff is NOT inverted from ADC (inhale is negative!)!!!!
    :param behavior_epoch:
    :param threshold:
    :return:
    """

    if np.max(sniff) < 150 or np.min(sniff) > -150:
        return np.array([]), np.array([])
    else:
        sniff_logical = sniff < threshold
        edges = np.convolve([1,-1], sniff_logical, mode='same')  # returns array of 1s and -1s for logic changes
        up_edges = np.where(edges == 1)[0]  # returns tuple, must take first member from tuple
        down_edges = np.where(edges == -1)[0]
        return up_edges, down_edges


def make_sniff_events(behavior_epoch, **kwargs):
    sniff = behavior_epoch.streams['sniff'].read()
    (behavior_epoch.events['sniff_inh'], behavior_epoch.events['sniff_exh']) = sniff_thresholding(sniff, **kwargs)

