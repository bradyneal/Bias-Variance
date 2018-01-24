from __future__ import print_function, division
import os
import re
from functools import partial
import torch



# from NNTrainer import NNTrainer
# from models import Linear, ShallowNet, MinDeepNet, ExampleNet
# from infmetrics import get_pairwise_hamming_diffs, get_pairwise_weight_dists_normalized, \
#     get_pairwise_pos_disagreements, get_pairwise_neg_disagreements       
# 
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt


OUTPUT_DIR = '/data/milatmp1/nealbray/information-paths/'
SAVED_DIR = os.path.join(OUTPUT_DIR, 'saved')
MODELS_DIR = os.path.join(SAVED_DIR, 'models')
MODEL_INF_DIR =  os.path.join(SAVED_DIR, 'model_inf')
BITMAP_DIR =  os.path.join(MODEL_INF_DIR, 'bitmaps')
WEIGHT_DIR =  os.path.join(MODEL_INF_DIR, 'weights')

CURRENT_SLURM_ID = os.environ["SLURM_JOB_ID"]
COMMON_NAMING_FORMAT = 'shallow%d_run%d_job%s.pt'
COMMON_REGEXP_FORMAT = r'shallow%d_run\d+_job(\d+).pt'

def load_torch(filename, to_cpu=False):
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


def get_path(directory, num_hidden, i , slurm_id):
    """Get path of a file in a specific directory"""
    return os.path.join(directory, get_filename(num_hidden, i, slurm_id))


"""
Functions that return the path for a specific directory
Args: num_hidden, i, slurm_id
"""
get_weight_path = partial(get_path, directory=WEIGHT_DIR)
get_bitmap_path = partial(get_path, directory=BITMAP_DIR)
get_model_path = partial(get_path, directory=MODEL_DIR)


def get_filename(num_hidden, i, slurm_id):
    """
    Return filename for a specific number of hidden units, run i, and SLURM id
    """
    return COMMON_NAMING_FORMAT % (num_hidden, i, slurm_id)
