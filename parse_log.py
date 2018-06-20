from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os

from parse_helper import *
from plotting import plot_line, plot_line_with_normal_errbars

class Types(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


def parse_array_dimensions(filename):
    types, seeds, hidden_arr, total_epochs = [Types.TRAIN, Types.VALIDATION], set(), [], None

    for line in open(filename, 'r'):
        if line.startswith('Learning rate'):
            seed = parse_seed(line)
            seeds.add(seed)

            if total_epochs is None:
                total_epochs = parse_total_epochs(line)

        if line.startswith('Training complete!!'):
            hidden_size = parse_hidden_size(line)
            if hidden_size not in hidden_arr:
                hidden_arr.append(hidden_size)

        if len(types) == 2 and line.startswith('After') and 'Test' in line:
            types.append(Types.TEST)

    seeds = list(seeds)
    return types, seeds, hidden_arr, total_epochs


def parse_from_file(slurm_id, quantity_required='error'):
    '''Inputs:
        slurm_id : int
        quantity_required : string. 'loss' or 'error' depending on which quantity is required to be parsed
       Returns:
        parsed_array : a numpy array of dimension (types, total_seeds, total_hidden_layers, total_epochs)
            types can be 2 or 3 depending on whether test errors have been printed in the log file
        seeds : list of seeds used in the experiment
        hidden_arr : list of hidden sizes used in the experiment
    '''

    filename = 'slurm-{}.out'.format(slurm_id)
    # Find fields required to calculate dimensions in final numpy array
    types, seeds, hidden_arr, total_epochs = parse_array_dimensions(filename)

    parsed_array = np.empty([len(types), len(seeds), len(hidden_arr), total_epochs])

    # Will parse line by line and extract any useful information it finds (3 cases defined in if conditions below)
    hidden_size_index = 0
    for line in open(filename, 'r'):
        # Updates the seed number for next lines to be parsed
        if line.startswith('Learning rate'):
            seed = parse_seed(line)
            seed_index = seeds.index(seed)

        # Updates the hidden size for next lines to be parsed
        if line.startswith('Training complete!!'):
            # hidden_size, hidden_size_index = parse_hidden_size(line) Can insert a check here but not needed
            hidden_size_index = (hidden_size_index + 1) % len(hidden_arr)

        # Parses the actual line and stores its contents
        if line.startswith('After'):
            quantity_value, current_epoch, type = parse_from_error_line(line, quantity_required, types)
            if type in types:
                # print(type.value, seed_index, hidden_size_index, current_epoch-1)
                parsed_array[type.value, seed_index, hidden_size_index, current_epoch-1] = quantity_value

    # This condition checks for incomplete training and returns upto the last seed for which training is complete for all hidden sizes
    if current_epoch is not total_epochs or hidden_size_index is not 0:
        parsed_array = parsed_array[:,:-1]
    return parsed_array, types, seeds, hidden_arr


def parse_and_make_plots(slurm_id):
    '''Parses from the slurm output file and creates plots for loss and error vs number of epochs for all hidden layer sizes
    '''
    plots_dir = os.path.join('parsed_plots', str(slurm_id))
    os.makedirs(plots_dir, exist_ok=True)

    for quantity_required in ['loss', 'error']:
        parsed_array, types, seeds, hidden_arr = parse_from_file(slurm_id, quantity_required)
        total_epochs = parsed_array.shape[3]
        for type in types:
            for i, hidden_size in enumerate(hidden_arr):
                yscale = 'linear'
                if quantity_required == 'loss':
                    yscale = 'log'
                plot_line(
                    x=range(1, total_epochs+1),
                    y=np.mean(parsed_array[type.value, :, i, :], axis=0),
                    # y_std=np.std(parsed_array[type.value, :, i, :], axis=0),
                    label='Hidden size = {}'.format(hidden_size),
                    xlabel='Number of epochs',
                    ylabel='{} {}'.format(type.name, quantity_required),
                    yscale=yscale,
                    title=quantity_required + ' vs number of epochs',
                    filename=os.path.join(plots_dir, '{}_{}.pdf'.format(type.name, quantity_required))
                )
            plt.close()

    if len(types) == 3:
        for i, hidden_size in enumerate(hidden_arr):
            plot_line(
                x=range(1, total_epochs+1),
                y=np.mean(parsed_array[2, :, i, :]-parsed_array[0, :, i, :], axis=0),
                # y_std=np.std(parsed_array[type.value, :, i, :], axis=0),
                label='Hidden size = {}'.format(hidden_size),
                xlabel='Number of epochs',
                ylabel='Generalization Gap',
                title='Generalization Gap vs number of epochs',
                filename=os.path.join(plots_dir, 'generalization_gap.pdf')
            )
