from __future__ import print_function
from NNTrainer import NNTrainer
from models import Linear, ShallowNet, MinDeepNet, ExampleNet
from infmetrics import get_pairwise_hamming_dists, get_pairwise_disagreements, \
                       get_pairwise_weight_dists
import torch
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

slurm_id = os.environ["SLURM_JOB_ID"]

DEFAULT_WIDTH = 6
DEFAULT_HEIGHT = 4

SAVED_MODELS_DIR = 'saved_models'


def run_nn_exp(num_hidden, retrain=False):
    bitmaps, weights = get_nn_information(num_hidden, retrain=retrain)
    all_ham_dists, _ = get_pairwise_hamming_dists(bitmaps)
    all_disagreements, _ = get_pairwise_disagreements(bitmaps)
    all_weight_dists, _ = get_pairwise_weight_dists(weights)
    print_summary(all_ham_dists, '%s RESULT for %d hidden units' % ('hamming', num_hidden))
    print_summary(all_disagreements, '%s RESULT for %d hidden units' % ('disagreement', num_hidden))
    print_summary(all_weight_dists, '%s RESULT for %d hidden units' % ('weights', num_hidden))
    plot_results(num_hidden, all_ham_dists, all_disagreements, all_weight_dists, same_figure=False)
    plot_results(num_hidden, all_ham_dists, all_disagreements, all_weight_dists, same_figure=True)
    # return all_ham_dists, all_disagreements, all_weight_dists


def get_nn_information(num_hidden, retrain=False):
    """
    Get neural network the 'information' stored in the neural networks
    (currently, test set bitmaps and weights), loading the neural networks
    from disk if they already exist and rerun isn't set to true.
    """
    # Decide whether or not to load existing neural networks
    saved_models = os.listdir(SAVED_MODELS_DIR)
    load_models = not retrain
    if len(saved_models) > 0:
        match = re.match(r'shallow\d+_run\d+_job(\d+).pt', saved_models[0])
        if match is None:
            load_models = False
        else:
            saved_slurm_id = match.groups()[0]
    if load_models:
        print('Saved models found... loading them')
    else:
        print('No saved models found... running training')
            
    bitmaps = []
    weights = []
    for i in range(20):
        # Load or train neural network
        if load_models:
            shallow_net = torch.load('saved_models/shallow%d_run%d_job%s.pt' % \
                                     (num_hidden, i, saved_slurm_id))
            trainer = NNTrainer(shallow_net)    # no training necessary
        else:
            shallow_net = ShallowNet(num_hidden)
            trainer = NNTrainer(shallow_net, lr=0.1, momentum=0.5, epochs=10)
            trainer.train(test=True)
            torch.save(shallow_net, 'saved_models/shallow%d_run%d_job%s.pt' % \
                                    (num_hidden, i, slurm_id))
        
        # Test and append bitmaps and weights to output lists    
        bitmap = trainer.test()
        bitmaps.append(bitmap)
        weights.append(shallow_net.get_params())
    
    return bitmaps, weights


def print_summary(l, message):
    print(message)
    print('min:', min(l), 'max:', max(l), 'mean:', sum(l) / len(l))


def plot_results(num_hidden, all_ham_dists, all_disagreements, all_weight_dists,
                 same_figure=False):
    num_plots_per_exp = 2
    if not same_figure:
        plt.figure(figsize=((DEFAULT_WIDTH + 1) * num_plots_per_exp, DEFAULT_HEIGHT))
    
    plt.subplot(1, num_plots_per_exp, 1)
    plt.plot(all_weight_dists, all_ham_dists, 'o')
    plt.title('Hamming distance vs. Weight distance')
    plt.xlabel('weight distance')
    plt.ylabel('hamming distance')
    
    plt.subplot(1, num_plots_per_exp, 2)
    plt.plot(all_weight_dists, all_disagreements, 'o')
    plt.title('Disagreement vs. Weight distance')
    plt.xlabel('weight distance')
    plt.ylabel('disagreement')
    
    if same_figure:
        plt.savefig('figures/shallow_exp_cum%d.png' % num_hidden)
    else:
        plt.savefig('figures/shallow_exp%d.png' % num_hidden)

        
if __name__ == '__main__':
    hidden_sizes = [5, 10, 15, 25, 50, 100, 250, 500]
    for num_hidden in hidden_sizes:
        run_nn_exp(num_hidden)
