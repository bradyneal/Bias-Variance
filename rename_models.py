# To run, import copy function and call it with required arguments

import os
import re
from shutil import copy2


def get_model_dir(username):
    #return os.path.join('/data/milatmp1', username, 'information-paths/saved/models')
    return os.path.join(os.getcwd(), 'saved/models')


def copy(old_slurm_id, new_slurm_id, load_user_id, saved_user_id):
    load_model_dir = get_model_dir(load_user_id)
    saved_model_dir = get_model_dir(saved_user_id)

    p = re.compile('(.*job)({}).pt'.format(old_slurm_id))
    for file in os.listdir(load_model_dir):
        m = p.match(file)
        if m:
            old_file_path = os.path.join(load_model_dir, file)
            print(old_file_path)

            new_file = '{}{}.pt'.format(m.group(1), new_slurm_id)
            new_file_path = os.path.join(saved_model_dir, new_file)
            print(new_file_path)

            copy2(old_file_path, new_file_path)
