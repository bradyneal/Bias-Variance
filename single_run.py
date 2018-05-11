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