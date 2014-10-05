__author__ = 'chris'

import numpy as np
import tables
import os
from data_classes import RichData


def reformat_voyeur_files(filenames):
    '''

    :param filenames: list of strings of hdf5 file_nm paths OR list of tuples (_h5 path in, _h5 path out).
    :type filenames: list
    :return: None
    '''
    if isinstance(filenames, list):
        for fn in filenames:
            #TODO: parallelize this with multiprocess
            try: reformat_voyeur_file(*fn)
            except TypeError:  # catch if fn is not a sequence
                reformat_voyeur_file(fn)
    elif isinstance(filenames, str):
        reformat_voyeur_file(filenames)
    return


def reformat_voyeur_file(input_path, save_path=None,
                         table_names=('Trials',),
                         stream_names=('sniff', 'treadmill'),
                         event_names=('lick1', 'lick2'),
                         ):
    """
    reformats file_nm

    :param input_path: str: full path to the unprocessed Voyeur ._h5 data file_nm.
    :param save_path: str: full path to the desired output file_nm.
    :param table_names: tuple of strings pointing to tabular data nodes in original _h5 file_nm.
    :param stream_names: tuple of strings pointing to stream data nodes in original _h5 file_nm. Stream data nodes are
    assumed to be embedded in multiple /Trial####/ groups.
    :param event_names: tuple of strings pointing to event (timestamp) data nodes in original _h5 file_nm. Event data nodes
    are assumed to be embedded in multiple /Trial####/ groups.
    :type input_path: str
    :type save_path: str
    :return:
    """
    if isinstance(input_path, list):
        reformat_voyeur_files(input_path)  # this will just call this function again a for every string in the list.
        return
    elif not isinstance(input_path, str):
        print 'reformat_voyeur_file() requires a string.'
    h5 = load_h5(input_path)
    # read and extract attributes from the _h5 file_nm object. Populate stream names if possible.
    h5_attr = h5.root._v_attrs
    if hasattr(h5_attr, 'stream_names'):
        stream_names = h5_attr.stream_names
        if isinstance(stream_names, str):
            stream_names = [stream_names]
    if hasattr(h5_attr, 'event_names'):
        event_names = h5_attr.event_names
        if isinstance(event_names, str):
            event_names = [event_names]
    tbls = {}
    streams = {}
    events = {}
    #TODO: carry over attributes from streams, tables, etc. by implementing classes for rich data structure.
    for table_name in table_names:
        try:
            tbls[table_name] = h5.get_node('/', table_name)
        except tables.NoSuchNodeError:
            print 'File has no table node named "%s".' % table_name
    for stream_name in stream_names:
        streams[stream_name] = process_continuous_stream(h5, stream_name)
    for event_name in event_names:
        events[event_name] = process_event_stream(h5, event_name)
    complete = save_file(save_path, tbls, streams, events, h5)
    h5.close()



def load_h5(filename):
    if os.path.isfile(filename):
        h5 = tables.open_file(filename, mode='r')
    else:
        print '%s does not exist' % filename
    # TODO: Check basic properties of the H5 file_nm to make sure that it is ok.
    return h5


def save_file(save_path, tbls, streams, events, orig_h5, no_compression=False):
    """
    Creates new hdf5 file_nm, and adds data tabular data from original file_nm along with the flattened stream data to the
    new file_nm.

    :param save_path: pathname to save file_nm directory.
    :param _h5: HDF5 object containing original data.
    :type save_path: str
    :type tbls: dict
    :type streams: dict of [np.array]
    :type events: dict
    :type orig_h5: tables.File
    :type no_compression: bool
    :return:
    """

    #TODO: check if write path already exists and verify overwrite??
    #TODO: make directory if necessary?
    new_file = tables.open_file(save_path, mode='w')

    # copy root attributes from original file_nm:
    new_file.copy_node_attrs(orig_h5.root, '/')

    # TODO: add animal_id & session information to attributes if they are not found:
    # if new_file.root._v_attrs.animal_id == 1:



    # get sampling rate from file_nm attributes if available, else set a default (1 kHz):
    try:
        fs = int(new_file.get_node_attr('/', 'voyeur_sample_rate'))
    except AttributeError:
        fs = 1000
        new_file.set_node_attr('/', 'voyeur_sample_rate', fs)
    # add tables by copying the originals:
    tbl_group = new_file.root
    for tbl_name, tbl_node in tbls.iteritems():
        nt = tbl_node.copy(tbl_group, tbl_name)
        nt.close()
    # add streams:
    str_grp = new_file.create_group('/', 'Streams', "Continuous Data Streams")
    strm_filt = tables.Filters(complevel=5, complib='zlib')
    for stream_name, stream_data in streams.iteritems():
        if stream_data:
            stream_data.add_to_h5(new_file, str_grp, stream_name, filters=strm_filt)  # stream data is a data_containers.RichData object.
    # add events:
    ev_group = new_file.create_group('/', 'Events', "Event Data (timestamps)")
    for ev_name, ev_data in events.iteritems():
        if ev_data:
            ev_data.add_to_h5(new_file, ev_group, ev_name)  # event data is a data_containers.RichData object.
    new_file.close()

    return


def process_event_stream(h5, stream_name):
    """
    This processes event streams (specifically licks), which have an on-off event time from arduino. This returns an
    n by 2 np.array with the first column being the 'on' time and the second column being the 'off' time.

    If a stream is not present in the file_nm, it should skip it without crashing and return None.

    :param h5: tables HDF5 file_nm object
    :param stream_name: string of the stream name (as enumerated in the H5 file_nm).
    :type h5: tables.File
    :type stream_name str
    :return:
    """
    st = []
    fs = None
    for trial in h5.root:
        try:
            tr_st = trial._f_get_child(stream_name).read()
            for i in tr_st:
                if i.size % 2:
                    # Protocol convention states that event streams are sent in even length packets of [on, off]
                    # The first event is a forced off event, so we should discard this, and subsequent stream packets
                    # will be 'in-phase', meaning (ON, OFF, ON, OFF).
                    continue
                for ii in i:
                    st.append(ii)
            # Get sampling rate for individual stream, if not available, set to voyeur default (1 kHz).
            if fs is None:
                try:
                    fs = tr_st.attrs['sample_rate']
                except KeyError:
                    try:
                        h5.get_node_attr('/', 'voyeur_sample_rate')
                    except AttributeError:
                        fs = 1000  # default == 1000
        except tables.NoSuchNodeError:  # if the stream does not exist in this file_nm, return None.
            return None
        except AttributeError:  # if the table doesn't have an Events table, continue to the next trial group.
            continue
    # Reshape the array so that it will be 2 columns, column 1 is 'on' and column 2 is 'off'.
    st_arr = np.array(st)
    # this will reshape the array such that the first column is "on" events, and the second is "off events"
    st_attr = {'sample_rate' : fs}
    if stream_name.startswith('lick'):
        l = st_arr.size / 2
        st_arr.shape = (l, 2)
    stream_obj = RichData(st_arr, st_attr)
    return stream_obj


def process_continuous_stream(h5, stream_name):
    """
    Processes continuous analog acquisition streams (ie sniff,

    :param h5: tables file_nm object.
    :param stream_name: string specifying the name of the stream to parse.
    :type h5: tables.File
    :type stream_name: str
    :return: continuous sniff array.
    """
    #TODO: extract and return sampling frequency from attributes.
    st = np.zeros(5e9, dtype=np.int16)  # allocate memory for ~ 23 hrs of recording at 1 kHz.
    fs = None
    for trial in h5.root:
        try:
            tr_events = trial.Events.read()
            tr_st = trial._f_get_child(stream_name).read()
            # HANDLE EMPTY FRAMES:
            del_list=[]
            for i, ev in enumerate(tr_events):
                if ev[1] == 0:
                    del_list.append(i)
            if del_list:
                tr_events = np.delete(tr_events, del_list)
            # MOVE FROM TRIAL TO CONTINUOUS STREAM:
            for ev, st_pkt in zip(tr_events, tr_st):
                tail = ev[0]
                head = tail - ev[1]
                st[head:tail] = st_pkt[:]
            if fs is None:
                try:
                    fs = tr_st.attrs['sample_rate']
                except KeyError:
                    try:
                        h5.get_node_attr('/', 'voyeur_sample_rate')
                    except AttributeError:
                        fs = 1000  # default == 1000
        except tables.NoSuchNodeError:  # if the stream does not exist in this file_nm, return None.
            # print 'no such node'
            return None
        except AttributeError as e:  # if the table doesn't have an Events table, continue to the next trial group.
            # print 'attribute error'
            # print e
            continue
    st_attr = {'sample_rate' : fs}
    stream_obj = RichData(st[:tail], st_attr)  # last tail is the last sample we need to save
    return stream_obj

if __name__ == '__main__':
    reformat_voyeur_file('/Users/chris/Data/Behavior/mouse_0932/H5/mouse932_sess189_D2014_4_24T11_4_8.h5',
                         '/Users/chris/Data/Behavior/mouse_0932/H5/tmp_mouse931_sess148_D2014_2_25T10_14_222._h5')