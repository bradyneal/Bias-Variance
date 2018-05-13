'''
Module for parsing functions
'''
import numpy as np

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


ADD_PARAMS_MSG = 'Starting exp: Batch GD for hidden size'
NEXT_EXPERIMENT_MSG = 'Suggestion:'
LR_MESSAGE = 'learning rate: '

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
                best_val_acc = float(line.split(':')[1].strip())
            elif line.startswith(LAST_VAL_MARK):
                last_val_acc = float(line.split(':')[1].strip())
            elif line.startswith(ADD_PARAMS_MSG):
                seed, hidden_size = line.split('size')[1].strip().split('with seed')
                seed, hidden_size = float(seed.strip()), float(hidden_size.strip())
            elif line.startswith(LR_MESSAGE):
                learning_rate = float(line.split(':')[1].strip())
            elif line.startswith(NEXT_EXPERIMENT_MSG):
                output.append([learning_rate, best_val_acc, last_val_acc, seed, hidden_size])
        output = np.array(output)

        sidx = np.lexsort(output[:, [1, 4]].T)
        idx = np.append(np.flatnonzero(output[1:, 4] > output[:-1, 4]), output.shape[0] - 1)
        formatted_output = output[sidx[idx]]

    return formatted_output, output, hidden_size
