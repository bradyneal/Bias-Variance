from skopt import gp_minimize
from DataModelComp import DataModelComp
from models import DeepNet
import sys
import matplotlib.pyplot as plt
from credentials import SIGOPT_TOKEN    # note this is not pushed
from sigopt import Connection

i = 8
if len(sys.argv) > 1:
    i = int(sys.argv[1])

NUM_HIDDEN = 100
SEED = 0
LOGLR_MIN = -4.5
LOGLR_MAX = -0.5
NUM_EVALS = 5
BASE = 10
depth = [2,1,5,6]
id = [46230,46227,46256,46759]

def val_cost(loglr, num_hidden, num_layer, seed, maximize=True):
    lr = BASE ** loglr
    print('learning rate:', lr)
    val_acc, _ = DataModelComp(DeepNet(num_hidden,num_layer), epochs=100, log_interval=None,
                               run_i=seed, train_val_split_seed=seed, seed=seed,
                               bootstrap=True, batch_size=100,
                               print_only_train_and_val_errors=True, print_all_errors=False, lr=lr, momentum=0.9,
                               plot_curves=False, save_best_model=True).train()
    if maximize:
        return val_acc
    else:
        return 1 - val_acc

def tune_sigopt(num_hidden, num_layer, seed, experiment_name=None, experiment_id=None):
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
        value = val_cost(suggestion.assignments['loglr'], num_hidden, num_layer, seed, maximize=True)
        print('Suggestion:', suggestion.assignments['loglr'], 'Validation value:', value)
        conn.experiments(experiment_id).observations().create(
            suggestion=suggestion.id,
            value=value.item(),
        )

    return experiment_id

if __name__ == '__main__':
    for seed in range(1):
        for num_layer,e in zip(depth,id):
            exp_name='{} for depth {}, with seed {}'.format('SGD', num_layer, seed)
            print('Starting exp:', exp_name)
            exp_id = tune_sigopt(NUM_HIDDEN, num_layer, seed, experiment_name=exp_name, experiment_id=str(e))
            print('Experiment {} finished'.format(exp_id))

    # print('Hidden size:', num_hidden)
    # res = gp_minimize(val_cost, [loglr_range])
    
# def x2(x):
#     return x[0] ** 2 + 4
# 
# if __name__ == '__main__':
#     print('Hidden size:', num_hidden)
#     res = gp_minimize(x2, [loglr_range], n_calls=50, n_random_starts=5)
#     opt_x, opt_value = res['x'], res['fun']
#     xs, values = res['x_iters'], res['func_vals']
#     xs = [x for [x] in xs]
#     values = values.tolist()
#     print('optimal x and f(x):', (opt_x, opt_value))
#     print('type of xs:', type(xs))
#     print(xs)
#     print('type of vals:', type(values))
#     print(values)
#     plt.plot(xs, values, 'ro')
#     plt.title('Learning rate tuning for network of hidden size {}'.format(num_hidden))
#     plt.xlabel('log learning rate')
#     plt.ylabel('validation error after 100 epochs')
#     plt.show()
