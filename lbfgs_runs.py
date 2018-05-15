from DataModelComp import DataModelComp
from models import ShallowNet
import argparse

parser = argparse.ArgumentParser(description='hyperparam tuning for hidden size')
parser.add_argument('--index_to_use', type=int, help='hidden size to use')

args = parser.parse_args()

HIDDEN_SIZES = [1, 2, 5, 25, 100, 500, 2500, 10000, 40000]
num_hidden = HIDDEN_SIZES[args.index_to_use]
lr = 0.1 # from hyperparam optimization

for seed in range(50):
    print('seed {}'.format(seed))
    val_acc, _ = DataModelComp(ShallowNet(num_hidden), epochs=500, log_interval=None,
                               run_i=seed, train_val_split_seed=seed, seed=seed,
                               bootstrap=True, batch_size=10, num_train_after_split=100,
                               print_only_train_and_val_errors=True, print_all_errors=False, lr=lr, momentum=0.9,
                               plot_curves=False, save_best_model=True).train()