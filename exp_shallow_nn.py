from __future__ import print_function, division
from NNTrainer import NNTrainer
from models import Linear, ShallowNet, MinDeepNet, ExampleNet
from infmetrics import get_pairwise_hamming_diffs, get_pairwise_disagreements, \
                       get_pairwise_weight_dists_normalized
import torch
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

slurm_id = os.environ["SLURM_JOB_ID"]

DEFAULT_WIDTH = 6
DEFAULT_HEIGHT = 4
HIDDEN_SIZES = [5, 10, 15, 25, 50, 100, 250, 500]
LAST_HIDDEN_SIZE = HIDDEN_SIZES[-1]

OUTPUT_DIR = '/data/milatmp1/nealbray/information-paths/'
SAVED_DIR = os.path.join(OUTPUT_DIR, 'saved')
SAVED_MODELS_DIR = os.path.join(SAVED_DIR, 'models')
SAVED_MODEL_INF_DIR =  os.path.join(SAVED_DIR, 'model_inf')
SAVED_BITMAP_DIR =  os.path.join(SAVED_MODEL_INF_DIR, 'bitmaps')
SAVED_WEIGHTS_DIR =  os.path.join(SAVED_MODEL_INF_DIR, 'weights')


def run_nn_exp(num_hidden, retrain=False, retest=False):
    print()
    print('Running experiment with %d hidden units' % num_hidden)
    bitmaps, weights = get_nn_information(num_hidden, retrain=retrain, retest=retest)
    bitmap_accs = map(torch.mean, bitmaps)
    avg_acc = sum(bitmap_accs) / len(bitmaps)
    print('Average accuracy over %d trials:' % len(bitmaps), avg_acc)
    all_ham_dists, _ = get_pairwise_hamming_diffs(bitmaps, avg_acc)
    all_disagreements, _ = get_pairwise_disagreements(bitmaps)
    all_weight_dists, _ = get_pairwise_weight_dists_normalized(weights)
    print_summary(all_ham_dists, '%s RESULT for %d hidden units' % ('hamming', num_hidden))
    print_summary(all_disagreements, '%s RESULT for %d hidden units' % ('disagreement', num_hidden))
    print_summary(all_weight_dists, '%s RESULT for %d hidden units' % ('weights', num_hidden))
    plot_results(num_hidden, all_ham_dists, all_disagreements, all_weight_dists, same_figure=False)
    plot_results(num_hidden, all_ham_dists, all_disagreements, all_weight_dists, same_figure=True)
    # return all_ham_dists, all_disagreements, all_weight_dists


def get_nn_information(num_hidden, retrain=False, retest=False):
    """
    Get neural network the 'information' stored in the neural networks
    (currently, test set bitmaps and weights), loading the neural networks
    from disk if they already exist and rerun isn't set to true.
    """
    # Regular expression used for common file naming
    regexp = r'shallow%d_run\d+_job(\d+).pt' % num_hidden
    
    # Decide whether or not to load existing model information
    saved_bitmaps = os.listdir(SAVED_BITMAP_DIR)
    saved_weights = os.listdir(SAVED_BITMAP_DIR)
    load_model_inf = (not retest) and len(saved_bitmaps) > 0
    if load_model_inf:
        match_gen = (re.match(regexp, bitmap_fn) for bitmap_fn in saved_bitmaps)
        match = next((m for m in match_gen if m), False)
        if match:
            saved_slurm_id = match.groups()[0]
        else:
            load_model_inf = False
    
    # Decide whether or not to load existing models
    saved_models = os.listdir(SAVED_MODELS_DIR)
    load_models = (not load_model_inf) and (not retrain) and len(saved_models) > 0
    if load_models:
        match_gen = (re.match(regexp, model_fn) for model_fn in saved_models)
        match = next((m for m in match_gen if m), False)
        if match:
            saved_slurm_id = match.groups()[0]
        else:
            load_models = False
            
    if load_model_inf:
        print('Saved model information found! Loading it from SLURM id %s...' % saved_slurm_id)
    elif load_models:
        print('Saved models found! Loading them from SLURM id %s...' % saved_slurm_id)
    else:
        print('No saved models found or purposefully retraining. Running training...')
            
    bitmaps = []
    weights = []
    for i in range(20):
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
    
    return bitmaps, weights


def print_summary(l, message):
    print(message)
    print('min:', min(l), 'max:', max(l), 'mean:', sum(l) / len(l))


def plot_results(num_hidden, all_ham_dists, all_disagreements, all_weight_dists,
                 same_figure=False):
    num_plots_per_exp = 2
    if same_figure:
        plt.figure('Cumulative', figsize=((DEFAULT_WIDTH + 1) * num_plots_per_exp, DEFAULT_HEIGHT))
    else:
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
        if num_hidden == LAST_HIDDEN_SIZE:
            plt.savefig('figures/shallow_exp_all.png')
    else:
        plt.savefig('figures/shallow_exp%d.png' % num_hidden)

        
if __name__ == '__main__':
    for num_hidden in HIDDEN_SIZES:
        run_nn_exp(num_hidden)
