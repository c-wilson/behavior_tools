__author__ = 'chris'

import datetime, os
import numpy as np


def get_session_fns(mouse, sessions=(), base_directory='/Users/chris/Data/Behavior/'):
    """
    Returns a list of filenames corresponding to the session files for a mouse.

    :param mouse: Int animal id number
    :param sessions: List of session numbers. If empty, will return ALL sessions for given mouse.
    :param base_directory: directory where the Data structure lives.
    :type mouse: int
    :type sessions: tuple
    :type base_directory: str
    :return: list of filenames corresponding to session h5 files.
    :rtype: list
    """
    mousedirname = 'mouse_%04i' % mouse
    mousepath = os.path.join(base_directory, mousedirname)
    fns = []

    if np.isscalar(sessions):
        sessions = [sessions]
    # if you leave sessions as an empty array, go through directory and find EVERY session number.
    else:
        sessions = list(sessions)

    if len(sessions) == 0:
        d = os.listdir(mousepath)
        for fn in d:
            m, s, d = parse_h5path(fn)
            if s:
                sessions.append(s)
    # Once session list is built go through the session numbers and return the filenames that occur with them.
    for session in sessions:
        fn = _get_session_fn(mouse, session, base_directory)
        if fn:
            fns.append(fn)
    return fns


def _get_session_fn(mouse, session, base_directory='~/Data/Behavior/'):
    """
    Helper function to get filename based on mouse number and session number.

    :param mouse: int mouse number
    :param session: int session number
    :param base_directory:
    :return:
    """
    mousedirname = 'mouse_%04i' % mouse
    mousepath = os.path.join(base_directory, mousedirname)
    session_name_seed = 'mouse%i_sess%i' % (mouse, session)
    d = os.listdir(mousepath)
    file_list = []
    for fn in d:
        if fn.startswith(session_name_seed):
            file_list.append(fn)
    if len(file_list) > 1:
        file_sizes = []
        for fn in file_list:
            ffn = os.path.join(mousepath, fn)
            file_sizes.append(os.path.getsize(ffn))
        ind = file_sizes.index(max(file_sizes))
        session_path = os.path.join(mousepath, file_list[ind])
    elif len(file_list) < 1:
        return None
    else:
        session_path = os.path.join(mousepath, file_list[0])
    return session_path


def parse_h5path(path):
    """

    :param path: string to filename
    :return: tuple of ints: (mouse, session, date),
    :type path: str
    """
    fn = os.path.split(path)[1]  # this returns the filename out of the entire path, even if only filename is provided.
    if fn.find('mouse') > -1:  # CAN CHANGE THE FORMATTING HERE!
        mouse_start = fn.find('mouse') + len('mouse')
    else:
        return None, None, None
    mouse_end = fn.find('_sess')
    mouse = fn[mouse_start:mouse_end]
    sess_start = mouse_end + len('_sess')
    sess_end = fn.find('_D')
    sess = fn[sess_start:sess_end]
    date_start = sess_end + len('_D')
    date_end = fn.find('.h5')
    dt_str = fn[date_start:date_end]
    dt_l = dt_str.split('T')
    date_str = dt_l[0]
    time_str = dt_l[1]
    date_str_l = date_str.split('_')
    time_str_l = time_str.split('_')
    date_int_l = [int(x) for x in date_str_l]
    time_int_l = [int(x) for x in time_str_l]
    dt_final = date_int_l + time_int_l  # makes a list of integers: [Y, M, D, H, M, S]
    date_obj = datetime.datetime(*dt_final)  # make a datetime object by unpacking this list.
    return int(mouse), int(sess), date_obj