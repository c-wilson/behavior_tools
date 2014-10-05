__author__ = 'chris'

from data_handling import voyeur_reformatting
import os

BASEDIR = '/Users/chris/Data/Behavior'

b = os.listdir(BASEDIR)
problem_log_path = os.path.join(BASEDIR, 'reprocess_problems.txt')
complete_log_path = os.path.join(BASEDIR, 'reprocess_completed.txt')

if not os.path.exists(problem_log_path):
    problem_log = open(problem_log_path, 'w')
else:
    problem_log = open(problem_log_path, 'w')

problem_log.write('PROBLEMS WITH FILE CONVERSION:\n')

if not os.path.exists(complete_log_path):
    completed_log = open(complete_log_path, 'w')  # make new fiel
else:
    completed_log = open(complete_log_path, 'a')  # append
completed_log.write('COMPLETED FILES:\n')

for name in b:
    mouse_path = os.path.join(BASEDIR, name)
    if name.startswith('mouse_') and os.path.isdir(mouse_path):
        new_pth = os.path.join(mouse_path, 'h5_new')
        try:
            os.makedirs(new_pth)
        except OSError:
            print 'path already exists'
        old_pth = os.path.join(mouse_path,'H5')
        h5_list = os.listdir(old_pth)
        for file_nm in h5_list:
            if file_nm.endswith('.h5') and file_nm.startswith('mouse'):
                full_file_nm = os.path.join(old_pth, file_nm)
                out_file = os.path.join(new_pth, file_nm)

                try:
                    voyeur_reformatting.reformat_voyeur_file(full_file_nm, save_path=out_file,
                                                            stream_names=('sniff',))
                    print 'Processed: %s' % file_nm
                    completed_log.write('%s\n' % full_file_nm)
                    os.rename(full_file_nm, os.path.join(old_pth, 'r_' + file_nm))
                except Exception as e:
                    print e
                    print 'Problem with file: %s' % file_nm
                    problem_log.write('%s\n' % full_file_nm)
problem_log.close()
completed_log.close()
print 'COMPLETE!'
