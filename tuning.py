from skopt import gp_minimize
from DataModelComp import DataModelComp
from models import ShallowNet
import sys
import matplotlib.pyplot as plt
from credentials import SIGOPT_TOKEN    # note this is not pushed
from sigopt import Connection

i = 8
if len(sys.argv) > 1:
    i = int(sys.argv[1])
    
HIDDEN_SIZES = [1, 2, 5, 25, 100, 500, 2500, 10000, 40000]
# tuning SGD 40000 (500) hidden for 5 seeds: 175670, 175835

# seed 0: 175028, 175029, 175083, 175033, 175035-175039 (175032 was unfinished for size 5)
# seed 1-5: 175169, 175170, 175171, 175173, 175174, 175175, 175176, 175178, 175184
NUM_HIDDEN = HIDDEN_SIZES[i]
SEED = 0
LOGLR_MIN = -5
LOGLR_MAX = 0
NUM_EVALS = 20
BASE = 10

def val_cost(loglr, num_hidden, seed, maximize=True):
    lr = BASE ** loglr
    print('learning rate:', lr)
    val_acc, _ = DataModelComp(ShallowNet(num_hidden), epochs=500, log_interval=None,
                               run_i=seed, train_val_split_seed=seed, seed=seed,
                               bootstrap=True, batch_size=10, num_train_after_split=100,
                               print_only_train_and_val_errors=True, print_all_errors=False, lr=lr, momentum=0.9,
                               plot_curves=False, save_best_model=True).train()
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
        exp_name='{} for hidden size {} with seed {}'.format('SGD', NUM_HIDDEN, seed)
        print('Starting exp:', exp_name)
        exp_id = tune_sigopt(NUM_HIDDEN, seed, experiment_name=exp_name)
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
