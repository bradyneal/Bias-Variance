"""
Module for plotting experimental results
"""

from __future__ import print_function, division
from fileio import load_model, load_train_bitmap, load_test_bitmap, get_train_test_modifiers
from infmetrics import hamming_diff

SLURM_ID = 116568


def load_and_plot_pairwise(hidden_sizes, num_runs, modifier, num_bins):
    """Plot pairwise distances in overlayed histograms"""
    if not isinstance(hidden_sizes, list):
        hidden_sizes = [hidden_sizes]
    modifier_train, modifier_test = get_train_test_modifiers(modifier)
    print(modifier_train)
    
    for num_hidden in hidden_sizes:
        pairwise_train = load_pairwise_dists(num_hidden, num_runs, modifier_train)
        pairwise_test = load_pairwise_dists(num_hidden, num_runs, modifier_test)
        
        plt.figure('train')
        plt.hist(pairwise_train, bins=num_bins)
        
        plt.figure('test')
        plt.hist(pairwise_test, bins=num_bins)


if __name__ == '__main__':
    load_and_plot_pairwise([5, 10, 15, 25, 50, 100, 250, 500], 20, 'hammdiffp2', 20)
