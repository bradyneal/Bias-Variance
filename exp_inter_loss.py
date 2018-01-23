from __future__ import division
from NNTrainer import NNTrainer
import os
import torch
from itertools import combinations
from models import get_inter_model
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HIDDEN_SIZES = [10, 15, 25, 50, 100, 250, 500]
# HIDDEN_SIZES = [5, 10, 15, 25, 50, 100, 250, 500]
MODELS_DIR = '/data/milatmp1/nealbray/information-paths/saved/models'
LINEAR_FIG_DIR = '/data/milatmp1/nealbray/information-paths/figures'

def plot_linearization(model1, model2, num_hidden, pair_i):
    num_steps = 10
    losses = []
    for i in range(num_steps):
        theta = i / num_steps
        tester = NNTrainer(get_inter_model(model1, model2, theta))
        inter_loss = tester.evaluate_loss().data[0]
        print(inter_loss)
        losses.append(inter_loss)
        
    plt.figure()
    plt.title('num_hidden: %d' % num_hidden)
    plt.plot(range(num_steps), losses)
    filename = os.path.join(LINEAR_FIG_DIR, 'linear%d_%d.png' % (num_hidden, pair_i))
    print('Saving', filename)
    plt.savefig(filename)
    plt.close()     # Close figure for memory reasons


def get_filename(num_hidden, i, slurm_id):
    return 'shallow%d_run%d_job%s.pt' % (num_hidden, i, slurm_id)


def load_model(num_hidden, run, slurm_id):
    # return torch.load(os.path.join(MODELS_DIR,
    #                                get_filename(num_hidden, run, slurm_id)),
    #                   map_location=lambda storage, loc: storage)
    return torch.load(os.path.join(MODELS_DIR,
                                   get_filename(num_hidden, run, slurm_id)))


def get_models(num_hidden):
    slurm_id = 116568
    models = []
    for i in range(20):
        models.append(load_model(num_hidden, i, slurm_id))
    return models


def plot_linearizations(num_hidden):
    models = get_models(num_hidden)
    all_pairs = list(combinations(models, 2))
    rand_pairs = random.sample(all_pairs, 25)
    for i, (model1, model2) in enumerate(rand_pairs):
        plot_linearization(model1, model2, num_hidden, i)
    plt.close('all')    # Make sure all figures are closed for memory reasons
        
        
if __name__ == '__main__':
    for num_hidden in HIDDEN_SIZES:
        plot_linearizations(num_hidden)
