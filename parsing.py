'''
Module for parsing functions
'''

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
