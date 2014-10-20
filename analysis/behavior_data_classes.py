__author__ = 'chris'

import tables
import utils
import numpy as np


class BehaviorRun(object):
    """

    """
    def __init__(self, file_path):

        self.file_path = file_path
        self.mouse, self.session, self.date_time = utils.parse_h5path(file_path)
        try:
            h5 = tables.open_file(file_path, mode='r')
        except IOError as er:
            # problem opening file_nm.
            print er
            return
        self.events = {}
        self.streams = {}
        for event_node in h5.root.Events:
            self.events[event_node.name] = event_node
        for stream_node in h5.root.Streams:
            self.streams[stream_node.name] = stream_node
        self.trials = h5.root.Trials.read()
        # just going to read the trials table into memory, it shouldn't be too big. this
        # allows you to use nice indexing to get data out of the table (ie self.trials['starttrial'] for a column or
        # self.trials[2:30] for a slice of rows.
        self._h5 = h5  # save this for later.

    def return_time_period(self, start_time, end_time, read_streams=True):
        """

        :param start_time:
        :param end_time:
        :type start_time: int
        :type end_time: int
        :return:
        """
        if not self._h5.isopen:
            print 'WARNING, HDF5 object for behavior session is closed.\nPlease reopen using the open() method.'

        events = {}
        streams = {}
        for k, event_node in self.events.iteritems():
            try:
                ev = event_node.read()
            except AttributeError:
                ev = event_node
            ev_l = (ev >= start_time) * (ev <= end_time)
            if ev.ndim == 2:  # for lick handling, we need to worry about both the on and off columns:
                if np.any(ev_l):
                    ev_i = np.where(ev_l)  # returns 2 d array. 1st row is row number, 2nd is column number.
                    ev_i_l = np.min(ev_i[0])  # using first column, which corresponds to the row (each row is an event).
                    ev_i_h = np.max(ev_i[0])
                    events[k] = ev[ev_i_l:ev_i_h]
                else:  # Handles the case where ev_l is an empty array, in which case the min and max functions explode.
                    events[k] = np.array([], dtype=ev.dtype)
            else:
                if np.any(ev_l):
                    events[k] = ev[ev_l]
                else:
                    events[k] = np.array([],dtype=ev.dtype)
        for k, stream_node in self.streams.iteritems():
            if read_streams:
                streams[k] = stream_node[start_time:end_time]  # reads these values from the stream node into memory.
                #TODO: fix this to realize and correct for sample rate discrepancies.
            elif not read_streams:
                #TODO: implement function where we can read values that we want later instead of loading into memory now.
                pass
        # assume that all 'Trials' events occur between the 'starttrial' and 'endtrial' times.
        starts = self.trials['starttrial']
        ends = self.trials['endtrial']
        idx = (starts <= end_time) * (starts >= start_time) * (ends <= end_time) * (ends >= start_time)  # a bit redundant.
        trials = self.trials[idx]  # produces a tables.Table
        return BehaviorEpoch(start_time, end_time, trials, events, streams, self)

    def return_trial(self, trial_index, padding=(2000, 2000)):
        """

        :param trial_index: int index of the trial within the Trials table.
        :param padding:
        :type trial_index: int
        :type padding:tuple of [int]
        :return:
        """
        trial = self.trials[trial_index]
        start = trial['starttrial']
        end = trial['endtrial']  # don't really care here whether this is higher than the record: np will return only as much as it has.
        if not start or not end:
            return None
        if np.isscalar(padding):
            padding = [padding, padding]
        if start >= padding[0]:  # don't want the start time to be before 0
            start -= padding[0]
        end += padding[1]
        return self.return_time_period(start, end)

    def close(self):
        """
        Closes _h5 object linked to the behavior run.
        :return:
        """
        if self._h5.isopen:
            self._h5.close()

    def open(self):
        """
        Opens HDF5 object underlying the behavior session.
        :return:
        """
        if not self._h5.isopen:
            self._h5 = tables.open_file(self.file_path)

    def __str__(self):
        return 'BehaviorRun:  Mouse: %i, Session: %i.' % (self.mouse, self.session)


class BehaviorEpoch(object):
    def __init__(self, start_time, end_time, trials=np.ndarray([]), events={}, streams={}, parent=None, **kwargs):
        """
        Container for arbitrary epochs of behavior data. Inherits metadata from the parent.

        :param parent: Parent behavior
        :param trials: Trials object.
        :param events: Dict of events.
        :param streams: Dict of streams.
        :type parent: BehaviorRun
        :type trials: np.ndarray
        :type events: dict
        :type streams: dict
        :return:
        """
        # trials.__getitem__
        self.trials = trials
        self.events = events
        self.streams = streams
        self.parent_epoch = parent
        self.start_time = start_time
        self.end_time = end_time
        if parent:
            self._process_parent(parent)
        return

    def _process_parent(self, parent):
        """

        """
        self.mouse = parent.mouse
        self.session = parent.session
        self.date_time = parent.date_time
        self._h5 = parent._h5


class BehaviorTrial(BehaviorEpoch):
    """
    Contains behavior epoch data for a single trial.
    """
    def __init__(self, behavior_epoch):
        """



        :param behavior_epoch:
        :type behavior_epoch: BehaviorEpoch
        :return:
        """
        # just want to check that this is in fact representing a single trial, and not many.
        if behavior_epoch.trials.size == 1:
            self.trials = behavior_epoch.trials
            self.events = behavior_epoch.events
            self.streams = behavior_epoch.streams
            try:
                # do this instead of getting these data from the system.
                self._process_parent(self, behavior_epoch.parent_epoch)
            except AttributeError:
                pass
        else:
            raise ValueError