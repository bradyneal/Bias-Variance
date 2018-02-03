"""
Module for long running scripts such as evaluation or training of many models
"""

from __future__ import print_function, division
from DataModelComp import DataModelComp           
import torch
import os
import re
from itertools import combinations
from fileio import load_model, save_model, save_train_bitmap, save_test_bitmap
from models import ShallowNet

NUM_I = 1000
SLURM_ID = 116568


def eval_model_and_save(data_model_comp, num_hidden, i, slurm_id):
    """Evaluate and save model train and test bitmaps."""
    _, _, train_bitmap = data_model_comp.evaluate_train()
    save_train_bitmap(train_bitmap, num_hidden, i, slurm_id)
    _, _, test_bitmap = data_model_comp.evaluate_test()
    save_test_bitmap(test_bitmap, num_hidden, i, slurm_id)
    

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
    

if __name__ == '__main__':
    eval_saved_models_and_save([10, 100], num_runs=1000, start_i=20, slurm_id=SLURM_ID)
