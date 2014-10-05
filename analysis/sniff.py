__author__ = 'chris'

import behavior_data_classes

def make_events(behav_obj):

    # behav_obj.events['sniff_on'] =
    # behav_obj.events['sniff_off'] =




def calculate_rxn_times(behav_obj):
    """

    :param behav_obj:
    :type behav_obj: data_classes.behavior_run
    :return:

    """
    for trial_row in behav_obj.Trials:
        start = trial_row['starttrial']
        end = trial_row['endtrial']
        if not (start or end):
            continue
        trial = behav_obj.return_trial()






