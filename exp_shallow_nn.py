from __future__ import print_function
from NNTrainer import NNTrainer
from models import Linear, ShallowNet, MinDeepNet, ExampleNet
from infmetrics import get_pairwise_hamming_dists, get_pairwise_disagreements, \
                       get_pairwise_weight_dists
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_WIDTH = 6
DEFAULT_HEIGHT = 4

def print_summary(l, message):
    print(message)
    print('min:', min(l), 'max:', max(l), 'mean:', sum(l) / len(l))


def run_nn_exp(num_hidden):
    bitmaps_shallow = []
    weights = []
    for _ in range(20):
    # for _ in range(5):
        shallow_net = ShallowNet(num_hidden)
        trainer = NNTrainer(shallow_net, lr=0.1, momentum=0.5, epochs=10)
        # trainer = NNTrainer(shallow_net, lr=0.1, momentum=0.5, epochs=1)
        trainer.train(test=True)
        bitmap_shallow = trainer.test()
        bitmaps_shallow.append(bitmap_shallow)
        weights.append(shallow_net.get_params())
    all_ham_dists, _ = get_pairwise_hamming_dists(bitmaps_shallow)
    all_disagreements, _ = get_pairwise_disagreements(bitmaps_shallow)
    all_weight_dists, _ = get_pairwise_weight_dists(weights)
    print_summary(all_ham_dists, '%s RESULT for %d hidden units' % ('hamming', num_hidden))
    print_summary(all_disagreements, '%s RESULT for %d hidden units' % ('disagreement', num_hidden))
    print_summary(all_weight_dists, '%s RESULT for %d hidden units' % ('weights', num_hidden))
    return all_ham_dists, all_disagreements, all_weight_dists

        
if __name__ == '__main__':
    hidden_sizes = [5, 10, 15, 25, 50, 100, 250, 500]
    # hidden_sizes = [5, 50]
    num_exp = len(hidden_sizes)
    num_plots_per_exp = 2
    plt.figure(figsize=((DEFAULT_WIDTH + 1) * num_plots_per_exp, DEFAULT_HEIGHT))
    for num_hidden in hidden_sizes:
        all_ham_dists, all_disagreements, all_weight_dists = run_nn_exp(num_hidden)
        
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
        
        plt.savefig('figures/shallow_exp%d.png' % num_hidden)
