"""
Module for long running scripts such as evaluation or training of many models
"""

from __future__ import print_function, division
from DataModelComp import DataModelComp           
import torch
import os
import re
from itertools import combinations
from fileio import load_model, load_train_bitmap, load_test_bitmap, get_train_test_modifiers, \
    save_model, save_weights, save_train_bitmap, save_test_bitmap, save_pairwise_dists
from models import ShallowNet
from infmetrics import hamming_diff

SLURM_ID = 116568


def eval_saved_models_and_save(hidden_sizes, num_runs, slurm_id, start_i=0):
    """
    Load all saved models according to hidden_sizes and num_runs, evaluate them
    on the training and test set, and save both bitmaps.
    """
    if not isinstance(hidden_sizes, list):
        hidden_sizes = [hidden_sizes]
    for num_hidden in hidden_sizes:
        print('num_hidden:', num_hidden)
        for i in range(start_i, num_runs):
            print('%d of %d' % (i + 1, num_runs))
            model = load_model(num_hidden, i, slurm_id)
            data_model_comp = DataModelComp(model)
            eval_model_and_save(data_model_comp, num_hidden, i, slurm_id)


def train_shallow_nns_and_save(hidden_sizes, num_runs, slurm_id=None, start_i=0):
    """
    Train many shallow nns, evaluate them, and save everything.
    """
    if slurm_id is None:
        slurm_id = os.environ["SLURM_JOB_ID"]
    if not isinstance(hidden_sizes, list):
        hidden_sizes = [hidden_sizes]
    for num_hidden in hidden_sizes:
        for i in range(start_i, num_runs):
            train_shallow_nn_and_save(num_hidden, i, slurm_id)

    
def compute_pairwise_metrics_and_save(hidden_sizes, num_runs, slurm_id, metric, modifier=None):
    """
    Compute the pairwise distance between all runs of a neural network of a
    fixed size, for all specified sizes, using the given metric, and saving
    to the filename according to the given modifier.
    """
    if not isinstance(hidden_sizes, list):
        hidden_sizes = [hidden_sizes]
    for num_hidden in hidden_sizes:
        print('\nnum_hidden:', num_hidden)
        num_pairs = num_runs * (num_runs - 1) / 2
        print_freq = max(round(num_pairs / 20), 1)
        train_dists = []
        test_dists = []
        count = 0
        for i, j in combinations(range(num_runs), 2):
            train_bitmap1 = load_train_bitmap(num_hidden, i, slurm_id)
            train_bitmap2 = load_train_bitmap(num_hidden, j, slurm_id)
            train_dist = metric(train_bitmap1, train_bitmap2)
            train_dists.append(train_dist)

            test_bitmap1 = load_test_bitmap(num_hidden, i, slurm_id)
            test_bitmap2 = load_test_bitmap(num_hidden, j, slurm_id) 
            test_dist = metric(test_bitmap1, test_bitmap2)
            test_dists.append(test_dist)

            count += 1
            if count % print_freq == 0:    
                print('{}% of the way done'.format(count / num_pairs * 100))
        
        modifier_train, modifier_test = get_train_test_modifiers(modifier)
        save_pairwise_dists(train_dists, num_hidden, num_runs, modifier_train)
        save_pairwise_dists(test_dists, num_hidden, num_runs, modifier_test)            


def eval_model_and_save(data_model_comp, num_hidden, i, slurm_id):
    """Evaluate and save model train and test bitmaps."""
    _, _, train_bitmap = data_model_comp.evaluate_train()
    save_train_bitmap(train_bitmap, num_hidden, i, slurm_id)
    _, _, test_bitmap = data_model_comp.evaluate_test()
    save_test_bitmap(test_bitmap, num_hidden, i, slurm_id)


def train_shallow_nn_and_save(num_hidden, i, slurm_id):
    """
    Train a single shallow nn, evaluate it on train and test set,
    and save everything.
    """
    shallow_net = ShallowNet(num_hidden)
    data_model_comp = DataModelComp(shallow_net, lr=0.1, momentum=0.5, epochs=10)
    data_model_comp.train()
    save_model(shallow_net, num_hidden, i, slurm_id)
    save_weights(shallow_net.get_params(), num_hidden, i, slurm_id)
    eval_model_and_save(data_model_comp, num_hidden, i, slurm_id)


if __name__ == '__main__':
    compute_pairwise_metrics_and_save([5, 10, 15, 25, 50, 100, 250, 500], 20, SLURM_ID, hamming_diff, 'hammdiffp2')
    # eval_saved_models_and_save([10, 100], num_runs=1000, start_i=20, slurm_id=SLURM_ID)
