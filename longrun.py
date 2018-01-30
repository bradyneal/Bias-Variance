from __future__ import print_function, division
from NNTrainer import NNTrainer                  
import torch
import os
import re
from itertools import combinations
from fileio import load_model, save_train_bitmap, save_test_bitmap

NUM_HIDDEN = 25
NUM_I = 1000
SLURM_ID = 116568

def eval_saved_models(hidden_sizes, num_runs, slurm_id):
    """
    Load saved models, evaluate them on the training and test set,
    and save both bitmaps.
    """
    if not isinstance(hidden_sizes, list):
        hidden_sizes = [hidden_sizes]
    for num_hidden in hidden_sizes:
        print('num_hidden:', num_hidden)
        for i in range(num_runs):
            print('%d of %d' % (i + 1, num_runs))
            model = load_model(num_hidden, i, slurm_id)
            evaler = NNTrainer(model)
            train_acc, _, train_bitmap = evaler.evaluate_training()
            save_train_bitmap(train_bitmap, num_hidden, i, slurm_id)
            test_bitmap = evaler.test()
            save_test_bitmap(test_bitmap, num_hidden, i, slurm_id)


if __name__ == '__main__':
    eval_saved_models([15, 100, 250], 20, SLURM_ID)
