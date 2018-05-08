from skopt import gp_minimize
from DataModelComp import DataModelComp
from models import ShallowNet
import sys
import matplotlib.pyplot as plt

i = 3
if len(sys.argv) > 1:
    i = int(sys.argv[1])
    
hidden_sizes = [5, 25, 100, 1E3, 5E3, 10E3, 20E3, 40E3, 80E3]
num_hidden = hidden_sizes[i]
seed = 0
loglr_range = (-25.0, 10.0)

def val_cost(loglr):
    print('learning rate:', lr)
    lr = 2 ** loglr[0]
    val_acc, _ = DataModelComp(ShallowNet(num_hidden), epochs=200, log_interval=None,
                               run_i=seed, train_val_split_seed=seed, save_all_at_end=True, seed=seed,
                               bootstrap=True, save_model_every_epoch=False, batch_size=10, num_train_after_split=100,
                               print_only_train_and_val_errors=True, print_all_errors=False, lr=lr, momentum=0.9).train()
    return 1 - val_acc

if __name__ == '__main__':
    print('Hidden size:', num_hidden)
    res = gp_minimize(val_cost, [loglr_range])
    
def x2(x):
    return x[0] ** 2 + 4

if __name__ == '__main__':
    print('Hidden size:', num_hidden)
    res = gp_minimize(x2, [loglr_range], n_calls=50, n_random_starts=5)
    opt_x, opt_value = res['x'], res['fun']
    xs, values = res['x_iters'], res['func_vals']
    xs = [x for [x] in xs]
    values = values.tolist()
    print('optimal x and f(x):', (opt_x, opt_value))
    print('type of xs:', type(xs))
    print(xs)
    print('type of vals:', type(values))
    print(values)
    plt.plot(xs, values, 'ro')
    plt.title('Learning rate tuning for network of hidden size {}'.format(num_hidden))
    plt.xlabel('log learning rate')
    plt.ylabel('validation error after 100 epochs')
    plt.show()
