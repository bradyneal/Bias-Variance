"""
Module for the saving and loading of different data
"""

from __future__ import print_function, division
import os
import torch
import getpass
import numpy as np
import pickle
from models import ShallowNet

USERNAME = getpass.getuser()

OUTPUT_DIR = os.path.join('/data/milatmp1', USERNAME, 'information-paths')
#OUTPUT_DIR = os.getcwd()  # for cedar/graham

SAVED_DIR = os.path.join(OUTPUT_DIR, 'saved')
MODEL_DIR = os.path.join(SAVED_DIR, 'models')
PROB_DIR = os.path.join(SAVED_DIR, 'probabilities')
VARIANCE_DIR = os.path.join(SAVED_DIR, 'variance')
BIAS_DIR = os.path.join(SAVED_DIR, 'variance')
DATA_MODEL_COMP_DIR = os.path.join(SAVED_DIR, 'data_model_comps')
HYPERPARAM_DIR = os.path.join(SAVED_DIR, 'hyperparam')
BITMAP_DIRS = ['train_bitmaps', 'val_bitmaps', 'test_bitmaps']
WEIGHT_DIR = os.path.join(SAVED_DIR, 'weights')
PAIRWISE_DISTS_DIR = os.path.join(SAVED_DIR, 'pairwise_dists')
PATH_DIR = os.path.join(SAVED_DIR, 'path_bitmaps')
FINE_PATH_DIR = os.path.join(SAVED_DIR, 'path_bitmaps_fine')
FINE_PATH_DIRS = [os.path.join(FINE_PATH_DIR, bitmap_dir) for bitmap_dir in BITMAP_DIRS]
PATHS = [SAVED_DIR, MODEL_DIR, WEIGHT_DIR, PAIRWISE_DISTS_DIR, PATH_DIR,
         FINE_PATH_DIR, DATA_MODEL_COMP_DIR, HYPERPARAM_DIR, PROB_DIR, VARIANCE_DIR, BIAS_DIR] + BITMAP_DIRS + FINE_PATH_DIRS

OLD_COMMON_NAMING_FORMAT = 'shallow%d_run%d_job%s.pt'
COMMON_NAMING_FORMAT = 'shallow%d_run%d_inter%d_job%s.pt'
COMMON_REGEXP_FORMAT = r'shallow%d_run\d+_job(\d+).pt'

TO_CPU_DEFAULT = False


def make_all_dirs():
    """Make all the directories if they don't already exist"""
    for path in PATHS:
        if not os.path.exists(path):
            print("Creating directory:", path)
            os.makedirs(path)

make_all_dirs()


def get_slurm_id():
    try:
        return os.environ["SLURM_JOB_ID"]
    except:
        return 0


"""Specific saving functions"""


def save_data_model_comp(data_model_comp_obj, slurm_id=get_slurm_id(), inter=0):
    data_model_comp_path = get_data_model_comp_path(data_model_comp_obj.model.num_hidden, data_model_comp_obj.run_i, slurm_id, inter)
    pickle.dump(data_model_comp_obj, open(data_model_comp_path, 'wb'))


def save_shallow_net(model, num_hidden, i, slurm_id=get_slurm_id(), inter=0):
    if isinstance(model, ShallowNet):
        return save_model(model, model.num_hidden, i, slurm_id, inter)
    else:
        raise Exception('Naming convention for saving model not implemented for models other than ShallowNet')


def save_model(model, num_hidden, i, slurm_id=get_slurm_id(), inter=0):
    return torch.save(model, get_model_path(num_hidden, i, slurm_id, inter))


def save_probabilities(slurm_id, num_hidden, probabilities):
    return np.save(get_probabilities_path(slurm_id, num_hidden), probabilities)


def save_correlations(slurm_id, correlations, matrix_num):
    return np.save(get_correlations_path(slurm_id, matrix_num), correlations)


def save_variance_data(slurm_id, variance, option):
    return np.save(get_variance_path(slurm_id, option), variance)


def save_variance_diffs(slurm_id, num_hidden, diffs):
    return np.save(get_variance_diffs_path(slurm_id, num_hidden), diffs)


def save_bias_diffs(slurm_id, num_hidden, diffs):
    return np.save(get_bias_diffs_path(slurm_id, num_hidden), diffs)


def save_weights(weights, num_hidden, i, slurm_id):
    return torch.save(weights, get_weight_path(num_hidden, i, slurm_id))


def save_bitmap(bitmap, num_hidden, i, slurm_id, type):
    return torch.save(bitmap, get_bitmap_path(num_hidden, i, slurm_id, type))


def save_pairwise_dists(pairwise_dists, num_hidden, num_runs, modifier):
    return torch.save(pairwise_dists, get_pairwise_dists_path(num_hidden, num_runs, modifier))


def save_opt_path_bitmaps(opt_path, num_hidden, i, slurm_id):
    return torch.save(opt_path, get_opt_path_bitmaps_path(num_hidden, i, slurm_id))


def save_fine_path_bitmaps(bitmap, num_hidden, i, inter, type):
    return torch.save(bitmap, get_fine_path_bitmaps_path(num_hidden, i, inter, get_slurm_id(), type))

"""Specific loading functions"""


def load_data_model_comp(num_hidden, i, slurm_id=get_slurm_id(), inter=0):
    return pickle.load(open(get_data_model_comp_path(num_hidden, i, slurm_id, inter), 'rb'))


def load_shallow_net(num_hidden, i, slurm_id, inter=0):
    return load_model(num_hidden, i, slurm_id, inter)


def load_model(num_hidden, i, slurm_id, inter=0):
    return load_torch(get_model_path(num_hidden, i, slurm_id, inter))


def load_probabilities(slurm_id, num_hidden):
    return np.load(get_probabilities_path(slurm_id, num_hidden))


def load_correlations(slurm_id, matrix_num):
    return np.load(get_correlations_path(slurm_id, matrix_num))


def load_variance_data(slurm_id, option):
    return np.load(get_variance_path(slurm_id, option))


def load_variance_diffs(slurm_id, num_hidden):
    return np.load(get_variance_diffs_path(slurm_id, num_hidden))


def load_bias_diffs(slurm_id, num_hidden):
    return np.load(get_bias_diffs_path(slurm_id, num_hidden))


def load_weights(num_hidden, i, slurm_id):
    return load_torch(get_weight_path(num_hidden, i, slurm_id))


def load_bitmap(num_hidden, i, slurm_id, type):
    return load_torch(get_bitmap_path(num_hidden, i, slurm_id, type))


def load_pairwise_dists(num_hidden, num_runs, modifier):
    return load_torch(get_pairwise_dists_path(num_hidden, num_runs, modifier))


def load_opt_path_bitmaps(num_hidden, i, slurm_id):
    return load_torch(get_opt_path_bitmaps_path(num_hidden, i, slurm_id))


def load_fine_path_bitmaps(num_hidden, i, inter, slurm_id, type):
    return load_torch(get_fine_path_bitmaps_path(num_hidden, i, inter, slurm_id, type))


def load_torch(filename, to_cpu=TO_CPU_DEFAULT):
    """Load torch object, reverting to loading to CPU if loading error"""
    # Don't even try to load normally if you know it's going to CPU
    if to_cpu:
        return load_to_cpu(filename)
    else:
        # Try to load data normally
        try:
            return torch.load(filename)
        # likely CUDA error from saving it from GPU and loading to CPU
        except RuntimeError:
            return load_to_cpu(filename)


def load_to_cpu(filename):
    """Load torch object specifically to CPU"""
    return torch.load(filename, map_location=lambda storage, loc: storage)


# Deprecated
def load_model_information():
    """
    TODO: trip this down and finish this (currently a copy paste)
    Load model information from disk
    """

    bitmap = torch.load(os.path.join(SAVED_BITMAP_DIR, load_fn))
    weight_vec = torch.load(os.path.join(SAVED_WEIGHTS_DIR, load_fn))

    save_model_filename = COMMON_NAMING_FORMAT % (num_hidden, i, slurm_id)
    save_info_fn = save_model_fn
    if load_model_inf or load_models:
        load_fn = COMMON_NAMING_FORMAT % (num_hidden, i, saved_slurm_id)
        save_info_fn = load_fn

    # Common file naming
        save_model_fn = 'shallow%d_run%d_job%s.pt' % (num_hidden, i, slurm_id)
        save_info_fn = save_model_fn
        if load_model_inf or load_models:
            load_fn = 'shallow%d_run%d_job%s.pt' % (num_hidden, i, saved_slurm_id)
            save_info_fn = load_fn

        # Load model information (fastest)
        if load_model_inf:
            bitmap = torch.load(os.path.join(SAVED_BITMAP_DIR, load_fn))
            weight_vec = torch.load(os.path.join(SAVED_WEIGHTS_DIR, load_fn))
        else:
            # Load models and test them (fast)
            if load_models:
                shallow_net = torch.load(os.path.join(SAVED_MODELS_DIR,
                                                      load_fn))
                trainer = NNTrainer(shallow_net)    # no training necessary
            # Train models (slow)
            else:
                shallow_net = ShallowNet(num_hidden)
                trainer = NNTrainer(shallow_net, lr=0.1, momentum=0.5, epochs=10)
                trainer.train(test=True)
                torch.save(shallow_net, os.path.join(SAVED_MODELS_DIR,
                                                     save_fn))
            bitmap = trainer.test()
            weight_vec = shallow_net.get_params()
            torch.save(bitmap, os.path.join(SAVED_BITMAP_DIR, save_info_fn))
            torch.save(weight_vec, os.path.join(SAVED_WEIGHTS_DIR, save_info_fn))

        # Append bitmaps and weights to output lists
        bitmaps.append(bitmap)
        weights.append(weight_vec)


"""
Functions that return the path for a specific directory
"""



def get_hyperparam_main_plot_path(first_job_id, option):
    return os.path.join(HYPERPARAM_DIR, '{}_job{}.jpg'.format(option, first_job_id))


def get_hyperparam_indi_plot_path(first_job_id, num_hidden, option):
    return os.path.join(HYPERPARAM_DIR, 'shallow{}_option{}_job{}.jpg'.format(num_hidden, option, first_job_id))


def get_probabilities_path(slurm_id, num_hidden):
    return os.path.join(PROB_DIR, 'shallow{}_job{}.npy'.format(num_hidden, slurm_id))


def get_correlations_path(slurm_id, matrix_num):
    return os.path.join(PROB_DIR, '{}_job{}.npy'.format(matrix_num, slurm_id))


def get_variance_path(slurm_id, option):
    '''
    option is either all or individual_variance
    '''
    return os.path.join(VARIANCE_DIR, '{}_{}.npy'.format(slurm_id, option))


def get_variance_diffs_path(slurm_id, num_hidden):
    return os.path.join(VARIANCE_DIR, 'diffs_shallow{}_job{}.npy'.format(num_hidden, slurm_id))


def get_bias_diffs_path(slurm_id, num_hidden):
    return os.path.join(BIAS_DIR, 'diffs_shallow{}_job{}.npy'.format(num_hidden, slurm_id))


def get_data_model_comp_path(num_hidden, i, slurm_id, inter=0):
    return get_path(DATA_MODEL_COMP_DIR, num_hidden, i, slurm_id, inter)


def get_model_path(num_hidden, i, slurm_id, inter=0):
    return get_path(MODEL_DIR, num_hidden, i, slurm_id, inter)


def get_weight_path(num_hidden, i, slurm_id):
    return get_path(WEIGHT_DIR, num_hidden, i, slurm_id)


def get_bitmap_path(num_hidden, i, slurm_id, type):  # TODO: rename type in all places
    return get_path(BITMAP_DIRS[type], num_hidden, i, slurm_id)


def get_pairwise_dists_path(num_hidden, num_runs, modifier):
    return os.path.join(PAIRWISE_DISTS_DIR,
                        'shallow{}_runs{}_{}.pt'.format(num_hidden, num_runs, modifier))


def get_opt_path_bitmaps_path(num_hidden, i, slurm_id):
    return get_path(PATH_DIR, num_hidden, i, slurm_id)


def get_fine_path_bitmaps_path(num_hidden, i, inter, slurm_id,
                               type  # 0 for train, 1 for validation, 2 for test
                               ):
    return os.path.join(FINE_PATH_DIRS[type],
        'shallow{}_run{}_inter{}_job{}.pt'.format(num_hidden, i, inter, slurm_id))


def get_path(directory, num_hidden, i, slurm_id, inter=0):
    """Get path of a file in a specific directory"""
    return os.path.join(directory, get_filename(num_hidden, i, slurm_id, inter))


def get_filename(num_hidden, i, slurm_id, inter=0):
    """
    Return filename for a specific number of hidden units, run i, and SLURM id
    """
    if int(slurm_id) > 161000:
        return COMMON_NAMING_FORMAT % (num_hidden, i, inter, slurm_id)
    return OLD_COMMON_NAMING_FORMAT % (num_hidden, i, slurm_id)


def get_train_test_modifiers(modifier=None):
    """Append the modifier to 'train' and 'test'"""
    modifier_train = 'train'
    modifier_test = 'test'
    if modifier is not None:
        modifier_train = modifier_train + '_' + modifier
        modifier_test = modifier_test + '_' + modifier
    return modifier_train, modifier_test
