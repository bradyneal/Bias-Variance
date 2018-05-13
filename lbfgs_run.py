from DataModelComp import DataModelComp
from models import ShallowNet
import sys
import matplotlib.pyplot as plt
from credentials import SIGOPT_TOKEN  # note this is not pushed
from sigopt import Connection
import argparse

parser = argparse.ArgumentParser(description='hyperparam tuning for hidden size')
parser.add_argument('--index_to_use', type=int, help='hidden size to use')
args = parser.parse_args()

HIDDEN_SIZES = [100, 500, 2500, 10000]
NUM_HIDDEN = HIDDEN_SIZES[args.index_to_use]
SEED = 0
LOGLR_MIN = -5
LOGLR_MAX = 0
NUM_EVALS = 20
BASE = 10


def val_cost(loglr, num_hidden, seed, maximize=True):
    lr = BASE ** loglr
    print('learning rate:', lr)
    val_acc, _ = DataModelComp(ShallowNet(num_hidden), epochs=20, log_interval=None,
                               run_i=seed, train_val_split_seed=seed, seed=seed,
                               bootstrap=True, batch_size=10, num_train_after_split=100,
                               print_only_train_and_val_errors=True, print_all_errors=False, lr=lr, momentum=0.9,
                               plot_curves=False, save_best_model=True,
                               optimizer="lbfgs", max_iter=20, history_size=100).train()
    return val_acc if maximize else 1 - val_acc


def tune_sigopt(num_hidden, seed, experiment_name=None, experiment_id=None):
    conn = Connection(client_token=SIGOPT_TOKEN)

    if experiment_id is None:
        # Create experiment
        experiment = conn.experiments().create(
            name=experiment_name,
            parameters=[
                dict(name='loglr', type='double', bounds=dict(min=LOGLR_MIN, max=LOGLR_MAX)),
            ],
        )
        print("Created experiment: https://sigopt.com/experiment/" + experiment.id)
        experiment_id = experiment.id
    else:
        print("Continuing experiment: https://sigopt.com/experiment/" + experiment_id)

    # Run the Optimization Loop between 10x - 20x the number of parameters
    for _ in range(NUM_EVALS):
        suggestion = conn.experiments(experiment_id).suggestions().create()
        value = val_cost(suggestion.assignments['loglr'], num_hidden, seed, maximize=True)
        print('Suggestion:', suggestion.assignments['loglr'], 'Validation value:', value)
        conn.experiments(experiment_id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

    return experiment_id


if __name__ == '__main__':
    for seed in range(5):
        exp_name = '{} for hidden size {} with seed {}'.format('SGD', NUM_HIDDEN, seed)
        print('Starting exp:', exp_name)
        exp_id = tune_sigopt(NUM_HIDDEN, seed, experiment_name=exp_name)
        print('Experiment {} finished'.format(exp_id))

'''
from DataModelComp import DataModelComp
from models import ShallowNet
import sys


HIDDEN_SIZES = [5, 25, 100, 1E3, 5E3, 10E3] #20E3, 40E3, 80E3]
LR = [1, 0.1, 0.01]


indx = 0
for lr in LR:
    for i in range(len(HIDDEN_SIZES)):
        seed = 2018+indx
        num_hidden = HIDDEN_SIZES[i]
        val_acc, _ = DataModelComp(ShallowNet(num_hidden), epochs=10, log_interval=None,
                                   run_i=seed, train_val_split_seed=seed, seed=seed,
                                   bootstrap=True, batch_size=100, num_train_after_split=100,
                                   print_only_train_and_val_errors=False, print_all_errors=True, lr=lr, momentum=0.9,
                                   plot_curves=False, optimizer="lbfgs", max_iter=20, history_size=100).train()
                                   #plot_curves=False, optimizer="adam", beta=0.9, beta2=0.99).train()
        indx += 1

print('done')
#if __name__ == '__main__':
'''