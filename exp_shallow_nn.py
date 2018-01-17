from __future__ import print_function
from NNTrainer import NNTrainer
from models import Linear, ShallowNet, MinDeepNet, ExampleNet
from infmetrics import get_pairwise_hamming_dists
from scipy.spatial.distance import hamming


def run_nn_exp(num_hidden):
    bitmaps_shallow = []
    for _ in range(20):
        shallow_net = ShallowNet(num_hidden)
        trainer = NNTrainer(shallow_net, lr=0.1, momentum=0.5, epochs=10)
        trainer.train(test=True)
        bitmap_shallow = trainer.test()
        bitmaps_shallow.append(bitmap_shallow)
    all_ham_dists, _ = get_pairwise_hamming_dists(bitmaps_shallow)
    print('RESULT for', num_hidden, 'hidden units')
    print('min:', min(all_ham_dists), 'max:', max(all_ham_dists), 'mean:', sum(all_ham_dists) / len(all_ham_dists))
        
if __name__ == '__main__':
    hidden_sizes = [5, 10, 15, 25, 50, 100, 250, 500]
    for num_hidden in hidden_sizes:
        run_nn_exp(num_hidden)
