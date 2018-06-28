'''
Module for parsing functions
'''
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from plotting import run_fig_extras


BEST_VAL_MARK = 'Best validation acc:'
LAST_VAL_MARK = 'Last validation acc:'
VAL_LIST_MARK = 'Validation list:'

def parse_validations(filename):
    '''
    Parse and read the best validation accuracy, last validation accuracy,
    and list of validation accuracies (for each epoch) for output file
    '''
    val_list = []
    best_val_acc = -1
    last_val_acc = -1

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(VAL_LIST_MARK):
                val_list_str = line.split(':')[1].strip().lstrip('[').rstrip(']')
                val_list = [float(acc) for acc in val_list_str.split(',')]
            elif line.startswith(BEST_VAL_MARK):
                best_val_acc = float(line.split(':')[1].strip())
            elif line.startswith(LAST_VAL_MARK):
                last_val_acc = float(line.split(':')[1].strip())
    return best_val_acc, last_val_acc, val_list


ADD_PARAMS_MSGS = 'Starting exp'
NEXT_EXPERIMENT_MSG = 'Suggestion:'
LR_MESSAGE = 'learning rate: '

def parse_accuracy(line):
    try:
        val_acc = float(line.split(':')[1].strip())
    except ValueError:
        if line.split(':')[1].strip().startswith('tensor'):
            val_acc = float(line.split(':')[1].strip()[7:-1])
        else:
            raise ValueError
    return val_acc


def parse_validations_table(filename):
    '''
    Parse and read the best validation accuracy, last validation accuracy,
    and list of validation accuracies (for each epoch) for output file
    returns
        - output: accuracies for every run
        - formatted_output: Only the best (early stopped) accuracy per seed
    This only works for one fixed size!
    '''
    output = []
    val_list = []
    best_val_acc = -1
    last_val_acc = -1
    seed = -1
    hidden_size = -1
    learning_rate = -1

    with open(filename, 'r') as f:
        for line in f:
            #if line.startswith(VAL_LIST_MARK):
            #    val_list_str = line.split(':')[1].strip().lstrip('[').rstrip(']')
            #    val_list = [float(acc) for acc in val_list_str.split(',')]
            if line.startswith(BEST_VAL_MARK):
                best_val_acc = parse_accuracy(line)
            elif line.startswith(LAST_VAL_MARK):
                last_val_acc = parse_accuracy(line)
            elif line.startswith(ADD_PARAMS_MSGS):
                hidden_size, seed = line.split('size')[1].strip().split('with seed')
                hidden_size, seed = float(hidden_size.strip()), float(seed.strip())
            elif line.startswith(LR_MESSAGE):
                learning_rate = float(line.split(':')[1].strip())
            elif line.startswith(NEXT_EXPERIMENT_MSG):
                output.append([learning_rate, best_val_acc, last_val_acc, hidden_size, seed])
        output = np.array(output)
        formatted_output = []
        for val in np.unique(output[:, 3]):
            #  restrict to only networks of specific size
            new_out = output[output[:, 3] == val, :]
            #  collect seeds with highest
            sidx = np.lexsort(new_out[:, [2, 4]].T)
            idx = np.append(np.flatnonzero(new_out[1:, 4] > new_out[:-1, 4]), new_out.shape[0] - 1)
            formatted_output.extend(new_out[sidx[idx]])

    return np.array(formatted_output), output, hidden_size


def plot_formatted_output(formatted_output):
    for row in formatted_output:
        lr, best_val_acc, last_val_acc, hidden_size, _ = row
        plt.plot(lr, last_val_acc, '.', label=str(hidden_size))
    run_fig_extras(
        xlabel='Learning rate',
        ylabel='Validation accuracy at end',
        title='Hyperparameter Tuning',
        filename='plots/tuning_cifar.pdf',
        xscale='log'
    )


if __name__ == '__main__':
    slurm_ids = [202408, 200631]
    formatted_outputs = None
    for slurm_id in slurm_ids:
        print('Parsing {}'.format(slurm_id))
        filename = 'slurm-{}.out'.format(slurm_id)
        formatted_output, _, _ = parse_validations_table(filename)
        formatted_outputs = np.append(
            formatted_outputs, formatted_output, axis=0
            ) if formatted_outputs is not None else formatted_output
    plot_formatted_output(formatted_outputs)
