__author__ = 'chris'

import os
from utilities import parse_h5path
from voyeur_reformatting import reformat_voyeur_file
import sys


def main(source_path, raw_store_path, reformatted_store_path):
    """

    :param source_path:
    :param raw_store_path:
    :param reformatted_store_path:
    :return:
    """

    new_files = os.listdir(source_path)

    for file in new_files:
        if not file.endswith('.h5'):
            continue
        orig_path = os.path.join(source_path, file)
        m, s, d = parse_h5path(file)
        mouse_flder = 'mouse_%04i' % m
        local_save_folder = os.path.join(reformatted_store_path, mouse_flder)
        local_save_path = os.path.join(local_save_folder, file)
        if not os.path.exists(local_save_folder):
            os.mkdir(local_save_folder)
        print 'Reformatting: %s.' % file
        try:
            reformat_voyeur_file(orig_path,
                                 save_path=local_save_path,
                                 stream_names=('sniff',))
            try:
                _move_to_repository(orig_path, m, raw_store_path)
            except Exception as e:
                print 'Warning: problem moving %s to repository.' % file
                print e.message
        except Exception as e:
            print 'Warning: problem reformatting file: %s. File was not moved to repository.' % file
    print 'Complete!'


def _move_to_repository(orig_path, mouse, remote_store_path):
    """

    :param orig_path:
    :param mouse:
    :param remote_store_path:
    :return:
    """
    remote_save_dir = os.path.join(remote_store_path, 'mouse_%04i' % mouse)
    if not os.path.exists(remote_save_dir):
        os.mkdir(remote_save_dir)  # should raise exemption if the folder cannot be created.
    fn = os.path.split(orig_path)[1]
    remote_save_path = os.path.join(remote_save_dir, fn)
    os.rename(orig_path, remote_save_path)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        hlp_str = ("\nUsage: python pull_new_files.py 'source_dir' 'raw_repository', 'reformatted_repository' \n\nThis "
                   "will reformat the files contained in the source directory, saving the reformatted file to the "
                   "reformatted repository, and will move the original raw h5 file to the raw repository.\n")
        sys.exit(hlp_str)
    else:
        source_directory = args[0]
        remote_repository = args[1]
        local_repository = args[2]
        main(source_directory, remote_repository, local_repository)
    sys.exit()