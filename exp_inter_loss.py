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

NUM_PAIRS = 25
# HIDDEN_SIZES = [10, 50]
HIDDEN_SIZES = [5, 10, 15, 25, 50, 100, 250, 500]
MODELS_DIR = '/data/milatmp1/nealbray/information-paths/saved/models'
LINEAR_FIG_DIR = '/data/milatmp1/nealbray/information-paths/figures'

def hidden_size_to_color(num_hidden):
    return 'C' + str(HIDDEN_SIZES.index(num_hidden))


def plot_linearization(model1, model2, num_hidden, pair_i, same_plot=True):
    num_steps = 10
    losses = []
    for i in range(num_steps):
        theta = i / num_steps
        tester = NNTrainer(get_inter_model(model1, model2, theta))
        inter_loss = tester.evaluate_loss().data[0]
        print(inter_loss)
        losses.append(inter_loss)
    
    if not same_plot:
        plt.figure()
        plt.title('num_hidden: %d\trun: %d' % (num_hidden, pair_i))
        plt.plot(range(num_steps), losses)
        filename = os.path.join(LINEAR_FIG_DIR, 'linear%d_%d.png' % (num_hidden, pair_i))
        print('Saving', filename)
        plt.savefig(filename)
        plt.close()     # Close figure for memory reasons
    else:
        plt.plot(range(num_steps), losses, hidden_size_to_color(num_hidden))
        # filename = os.path.join(LINEAR_FIG_DIR, 'linear%d_%d.png' % (num_hidden, pair_i))


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


def plot_linearizations(num_hidden, same_plot=True):
    models = get_models(num_hidden)
    all_pairs = list(combinations(models, 2))
    rand_pairs = random.sample(all_pairs, NUM_PAIRS)
    for i, (model1, model2) in enumerate(rand_pairs):
        plot_linearization(model1, model2, num_hidden, i, same_plot=same_plot)


def plot_all_linearizations():
    for num_hidden in HIDDEN_SIZES:
        plot_linearizations(num_hidden, same_plot=True)
    filename = os.path.join(LINEAR_FIG_DIR, 'linear_all.png')
    plt.savefig(filename)
    
        
if __name__ == '__main__':
    # for num_hidden in HIDDEN_SIZES:
    #     plot_linearizations(num_hidden, same_plot=False)
    plot_all_linearizations()
