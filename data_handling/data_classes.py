__author__ = 'chris'

import tables
import numpy as np
import os.path
import datetime

class RichData(object):
    def __init__(self, data, attributes = {}):
        """

        :param data: Data object.
        :param attributes: Dictionary of attributes (metadata) that will be included with _h5 file_nm.
        :type attributes: dict
        :type data: np.array
        :return:
        """
        self.data = data
        self.attributes = attributes
        return

    def add_to_h5(self, h5_obj, where, name, **kwargs):
        """
        Help function to add data to an _h5 file_nm with metadata attributes.

        :param h5_obj: the tables.File object to write the data to. H5 object must be open and writable!!
        :param where: H5 group object to add the data leaf to.
        :param name: Str name of the new data leaf.
        :param kwargs:
        :type h5_obj: tables.File
        :type where: tables.Group
        :return:
        """

        try:
            leaf = h5_obj.create_carray(where, name,
                                        atom=tables.Atom.from_dtype(self.data.dtype),
                                        shape=self.data.shape,
                                        **kwargs)
            leaf[:] = self.data
            for attr_name, attr_val in self.attributes.iteritems():
                leaf.set_attr(attr_name, attr_val)
            leaf.close()
        except tables.NoSuchNodeError:
            # print 'Node %s does not exist.' % where
            pass
        except ValueError:
            pass
            #print self.data.shape
