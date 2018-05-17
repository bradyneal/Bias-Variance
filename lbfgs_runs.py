from DataModelComp import DataModelComp
from models import ShallowNet
import argparse

parser = argparse.ArgumentParser(description='hyperparam tuning for hidden size')
parser.add_argument('--index_to_use', type=int, help='hidden size to use')

args = parser.parse_args()

HIDDEN_SIZES = [1, 2, 5, 25, 100, 500, 2500, 10000]
LRS = [-2, -2, -2.5, -0.5, -1.25, -0.5, -1, -2]
num_hidden = HIDDEN_SIZES[args.index_to_use]
lr = 10**LRS[args.index_to_use]  # from hyperparam optimization

for seed in range(50):
    print('seed {}'.format(seed))
    val_acc, _ = DataModelComp(ShallowNet(num_hidden), epochs=25, log_interval=None,
                               run_i=seed, train_val_split_seed=seed, seed=seed,
                               bootstrap=True, batch_size=100, num_train_after_split=100,
                               print_only_train_and_val_errors=True, print_all_errors=False, lr=lr, momentum=0.9,
                               plot_curves=False, save_best_model=False, optimizer="lbfgs").train()