__author__ = 'chris'


import datetime
import os




def parse_h5path(path):
    fn = os.path.split(path)[1]
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
    date_int_l = [ int(x) for x in date_str_l ]
    time_int_l = [ int(x) for x in time_str_l ]
    dt_final = date_int_l + time_int_l  # makes a list of integers: [Y, M, D, H, M, S]
    date_obj = datetime.datetime(*dt_final)  # make a datetime object by unpacking this list.
    return int(mouse), int(sess), date_obj
