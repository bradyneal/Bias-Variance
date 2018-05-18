import copy

'''
Module for parsing functions
'''

EPOCH_MARK = 'After 1000 epochs'

def parse_train_errors(slurm_id, hidden_arr):
    '''
    Parse and read the best validation accuracy, last validation accuracy,
    and list of validation accuracies (for each epoch) for output file
    '''
    filename = 'slurm-{}.out'.format(slurm_id)
    errors = []
    global EPOCH_MARK

    # TODO: update this.
    if slurm_id == 167011:
        EPOCH_MARK = 'After 100 epochs'
        hidden_arr = hidden_arr[:-2]
    num_hidden = len(hidden_arr)
    for _ in range(num_hidden):
        errors.append([])
    for _ in range(2):
        errors.append([0]*50)
    current_hidden_index = 0
    even = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(EPOCH_MARK) and 'Training set' in line:
                if slurm_id == 167011 and even == 1:
                    even = 0
                    continue
                even = 1
                line2 = copy.copy(line)
                first_bracket_index = line.index('(')
                line2 = line2[first_bracket_index+1:]

                second_bracket_index = line2.index('(')
                percent_index = line2.index('%')
                accuracy_str = line2[second_bracket_index+1:percent_index]
                error = 1 - float(accuracy_str)/100

                errors[current_hidden_index].append(error)
                current_hidden_index = (current_hidden_index + 1) % num_hidden

    return errors
