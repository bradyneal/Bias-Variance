"""
Module for the saving and loading of different data
"""

from __future__ import print_function, division
import os
import re
import torch


OUTPUT_DIR = '/data/milatmp1/nealbray/information-paths/'
SAVED_DIR = os.path.join(OUTPUT_DIR, 'saved')
MODEL_DIR = os.path.join(SAVED_DIR, 'models')
TRAIN_BITMAP_DIR =  os.path.join(SAVED_DIR, 'train_bitmaps')
TEST_BITMAP_DIR =  os.path.join(SAVED_DIR, 'test_bitmaps')
WEIGHT_DIR =  os.path.join(SAVED_DIR, 'weights')
PAIRWISE_DISTS_DIR =  os.path.join(SAVED_DIR, 'pairwise_dists')

COMMON_NAMING_FORMAT = 'shallow%d_run%d_job%s.pt'
COMMON_REGEXP_FORMAT = r'shallow%d_run\d+_job(\d+).pt'

TO_CPU_DEFAULT = False

"""Specific saving functions"""
def save_model(model, num_hidden, i , slurm_id):
    return torch.save(model, get_model_path(num_hidden, i , slurm_id))

def save_weights(weights, num_hidden, i , slurm_id):
    return torch.save(weights, get_weight_path(num_hidden, i , slurm_id))

def save_train_bitmap(bitmap, num_hidden, i , slurm_id):
    return torch.save(bitmap, get_train_bitmap_path(num_hidden, i , slurm_id))
    
def save_test_bitmap(bitmap, num_hidden, i , slurm_id):
    return torch.save(bitmap, get_test_bitmap_path(num_hidden, i , slurm_id))

def save_pairwise_dists(pairwise_dists, num_hidden, num_runs, modifier):
    return torch.save(pairwise_dists, get_pairwise_dists_path(num_hidden, num_runs, modifier))

"""Specific loading functions"""
def load_model(num_hidden, i , slurm_id):
    return load_torch(get_model_path(num_hidden, i , slurm_id))

def load_weights(num_hidden, i , slurm_id):
    return load_torch(get_weight_path(num_hidden, i , slurm_id))

def load_train_bitmap(num_hidden, i , slurm_id):
    return load_torch(get_train_bitmap_path(num_hidden, i , slurm_id))
    
def load_test_bitmap(num_hidden, i , slurm_id):
    return load_torch(get_test_bitmap_path(num_hidden, i , slurm_id))

def load_pairwise_dists(num_hidden, num_runs, modifier):
    return load_torch(get_pairwise_dists_path(num_hidden, num_runs, modifier))


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
        except RuntimeError as e:
            return load_to_cpu(filename)
            
    
def load_to_cpu(filename):
    """Load torch object specifically to CPU"""
    return torch.load(filename, map_location=lambda storage, loc: storage)


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
def get_model_path(num_hidden, i , slurm_id):
    return get_path(MODEL_DIR, num_hidden, i , slurm_id)

def get_weight_path(num_hidden, i , slurm_id):
    return get_path(WEIGHT_DIR, num_hidden, i , slurm_id)

def get_train_bitmap_path(num_hidden, i , slurm_id):
    return get_path(TRAIN_BITMAP_DIR, num_hidden, i , slurm_id)

def get_test_bitmap_path(num_hidden, i , slurm_id):
    return get_path(TEST_BITMAP_DIR, num_hidden, i , slurm_id)

def get_pairwise_dists_path(num_hidden, num_runs, modifier):
    return os.path.join(PAIRWISE_DISTS_DIR,
                        'shallow{}_runs{}_{}.pt'.format(num_hidden, num_runs, modifier))


def get_path(directory, num_hidden, i , slurm_id):
    """Get path of a file in a specific directory"""
    return os.path.join(directory, get_filename(num_hidden, i, slurm_id))


def get_filename(num_hidden, i, slurm_id):
    """
    Return filename for a specific number of hidden units, run i, and SLURM id
    """
    return COMMON_NAMING_FORMAT % (num_hidden, i, slurm_id)

def get_train_test_modifiers(modifier):
    """Append the modifier to 'train' and 'test'"""
    modifier_train = 'train'
    modifier_test = 'test'
    if modifier is not None:
        modifier_train = modifier_train + '_' + modifier
        modifier_test = modifier_test + '_' + modifier
    return modifier_train, modifier_test
