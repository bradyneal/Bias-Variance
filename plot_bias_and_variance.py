from fileio import load_probabilities, save_variance_diffs, load_variance_diffs, save_bias_diffs, load_bias_diffs, load_train_errors
import math
import numpy as np
from plotting import plot_line_with_errbars, plot_line_with_normal_errbars

from test_y_onehot import get_test_y_onehot


def calculate_variance(bitmaps):
    mean = np.mean(bitmaps, 0)
    return np.mean((bitmaps - np.expand_dims(mean, axis=0)) ** 2)


def get_variance(slurm_id, num_hidden):
    '''
    Returns the variance for a slurm id (corresponding to an experiment) and a hidden size.
    '''
    probabilities = load_probabilities(slurm_id, num_hidden)
    return calculate_variance(probabilities)


def calculate_bias(probabilities, test_y_onehot):
    mean = np.mean(probabilities, 0)
    return np.mean((mean - test_y_onehot) ** 2)


def calculate_losses(probabilities, test_y_onehot):
    print(probabilities.shape, test_y_onehot.shape)
    pred = np.argmax(probabilities, axis=2)
    target = np.argmax(test_y_onehot, axis=1)
    losses = np.mean(pred != target, axis=1)
    return losses


def get_bias(slurm_id, num_hidden):
    '''
    Returns the variance for a slurm id (corresponding to an experiment) and a hidden size.
    '''
    test_y_onehot = get_test_y_onehot()
    probabilities = load_probabilities(slurm_id, num_hidden)
    return calculate_bias(probabilities, test_y_onehot)


def load_probabilities_and_get_variances(slurm_id, hidden_arr, num_bootstrap=10000):
    '''
    Loads saved probabilities, calculates differences using bootstrapping from
    the value of variance computed using all the seeds and saves those diffs.
    Prerequisite: Probabilities should be saved earlier using the
    save_probabilities function with dimension (num_seeds, num_test_examples,
    probabilities_for_each_example), eg. (50, 10000, 10) for 50 seeds for MNIST.
    '''
    for num_hidden in hidden_arr:
        probabilities = load_probabilities(slurm_id, num_hidden)
        original_variance = calculate_variance(probabilities)

        diffs = []
        for i in range(num_bootstrap):
            indices = np.random.choice(50, 50, replace=True)
            bootstrap_probabilities = probabilities[indices]
            bootstrap_variance = calculate_variance(bootstrap_probabilities)
            diff_variance = (bootstrap_variance - original_variance)
            diffs.append(diff_variance)

        save_variance_diffs(slurm_id, num_hidden, diffs)


def load_probabilities_and_get_biases(slurm_id, hidden_arr, num_bootstrap=10000):
    test_y_onehot = get_test_y_onehot()
    for num_hidden in hidden_arr:
        probabilities = load_probabilities(slurm_id, num_hidden)
        original_variance = calculate_bias(probabilities, test_y_onehot)

        diffs = []
        for i in range(num_bootstrap):
            indices = np.random.choice(50, 50, replace=True)
            bootstrap_probabilities = probabilities[indices]
            bootstrap_variance = calculate_bias(bootstrap_probabilities, test_y_onehot)
            diff_variance = (bootstrap_variance - original_variance)
            diffs.append(diff_variance)

        save_bias_diffs(slurm_id, num_hidden, diffs)


def load_probabilities_and_get_losses_and_std(slurm_id, hidden_arr):
    test_y_onehot = get_test_y_onehot()
    average_losses, stds = [], []
    for num_hidden in hidden_arr:
        probabilities = load_probabilities(slurm_id, num_hidden)
        losses = calculate_losses(probabilities, test_y_onehot)

        average_loss = np.mean(losses)
        std = np.std(losses)/math.sqrt(len(losses))

        average_losses.append(average_loss)
        stds.append(std)

    return average_losses, stds


def plot_losses_and_std(slurm_id, hidden_arr, label=None, marker=None):
    average_losses, stds = load_probabilities_and_get_losses_and_std(slurm_id, hidden_arr)
    plot_line_with_normal_errbars(hidden_arr, average_losses, stds,
        xlabel='Number of hidden units', ylabel='Average loss on test data',
        grid=True, xscale='log', label=label,
        filename='plots/{}_test_error.pdf'.format(slurm_id), marker=marker)


def load_train_losses_and_get_average_and_std(slurm_id, hidden_arr):
    average_losses, stds = [], []
    for num_hidden in hidden_arr:
        probabilities = load_probabilities(slurm_id, num_hidden)
        losses = load_train_errors(slurm_id, num_hidden)

        average_loss = np.mean(losses)
        std = np.std(losses)/math.sqrt(len(losses))

        average_losses.append(average_loss)
        stds.append(std)

    return average_losses, stds


def plot_train_losses_and_std(slurm_id, hidden_arr, label=None, marker=None):
    average_losses, stds = load_train_losses_and_get_average_and_std(slurm_id, hidden_arr)
    plot_line_with_normal_errbars(hidden_arr, average_losses, stds,
        xlabel='Number of hidden units', ylabel='Average loss on train data',
        grid=True, xscale='log', label=label,
        filename='plots/{}_train_error.pdf'.format(slurm_id), marker=marker)


def get_percentile(diffs, percentile):
    print(int(round(percentile * len(diffs))))
    return diffs[int(round(percentile * len(diffs)))]


def find_variances_and_diffs(slurm_id, hidden_arr):
    variances, lower_diffs, upper_diffs = [], [], []
    for num_hidden in hidden_arr:
        original_variance = get_variance(slurm_id, num_hidden)
        variances.append(original_variance)

        diffs = load_variance_diffs(slurm_id, num_hidden)
        print(np.mean(np.array(diffs)))
        diffs = list(sorted(diffs))

        upper_diff = get_percentile(diffs, 0.995)
        lower_diff = -get_percentile(diffs, 0.005)

        upper_diffs.append(upper_diff)
        lower_diffs.append(lower_diff)

    return variances, lower_diffs, upper_diffs


def find_biases_and_diffs(slurm_id, hidden_arr, upper_percentile=0.995,
    lower_percentile=0.005):
    '''
    Loads the differences calculated using bootstrapping, finds error bars using
    those (calculated using the percentile values) and returns the error bars.
    Prerequisite: diffs should be saved earlier using the
    load_probabilities_and_get_variances function.
    '''
    variances, lower_diffs, upper_diffs = [], [], []
    for num_hidden in hidden_arr:
        original_variance = get_bias(slurm_id, num_hidden)
        variances.append(original_variance)

        diffs = load_bias_diffs(slurm_id, num_hidden)
        print(np.mean(np.array(diffs)))
        diffs = list(sorted(diffs))

        upper_diff = get_percentile(diffs, upper_percentile)
        lower_diff = -get_percentile(diffs, lower_percentile)

        upper_diffs.append(upper_diff)
        lower_diffs.append(lower_diff)

    return variances, lower_diffs, upper_diffs


def plot_variances_with_diffs(slurm_id, hidden_arr, title=None, label=None, marker=None):
    '''
    Loads the differences calculated using bootstrapping, finds error bars using
    those and plots the graph for variance and error bars.
    Prerequisite: diffs should be saved earlier using the
    load_probabilities_and_get_variances function.
    '''
    variances, lower_diffs, upper_diffs = find_variances_and_diffs(slurm_id, hidden_arr)
    plot_line_with_errbars(hidden_arr, variances, lower_diffs, upper_diffs,
        grid=True, xscale='log', ylabel='Variance', xlabel='Number of hidden units',
        filename='plots/{}_variance.pdf'.format(slurm_id), title=title, label=label, marker=marker)


def plot_biases_with_diffs(slurm_id, hidden_arr, title=None, label=None, marker=None):
    variances, lower_diffs, upper_diffs = find_biases_and_diffs(slurm_id, hidden_arr)
    plot_line_with_errbars(hidden_arr, variances, lower_diffs, upper_diffs,
        grid=True, xscale='log', ylabel='Bias and Variance', xlabel='Number of hidden units',
        filename='plots/{}_bias.pdf'.format(slurm_id), title=title, label=label, marker=marker)


def plot_variances_with_diffs_together(slurm_ids, hidden_arr, labels, title=None):
    '''
    Same as plot_variances_with_diffs but does the plotting for multiple slurm_ids.
    '''
    for i, slurm_id in enumerate(slurm_ids):
        variances, lower_diffs, upper_diffs = find_variances_and_diffs(slurm_id, hidden_arr)
        plot_line_with_errbars(hidden_arr, variances, lower_diffs, upper_diffs,
            grid=True, xscale='log', label=labels[i], xlabel='Number of hidden units',
            ylabel='Variance', filename='plots/{}_variance.pdf'.format(slurm_id),
            elinewidth=0.5*(2-i), title=title)


def plot_biases_with_diffs_together(slurm_ids, hidden_arr, labels, title=None):
    for i, slurm_id in enumerate(slurm_ids):
        variances, lower_diffs, upper_diffs = find_biases_and_diffs(slurm_id, hidden_arr)
        plot_line_with_errbars(hidden_arr, variances, lower_diffs, upper_diffs,
            grid=True, xscale='log', label=labels[i], xlabel='Number of hidden units',
            ylabel='Bias and Variance', filename='plots/{}_bias.pdf'.format(slurm_id),
            elinewidth=0.5*(2-i), title=title)
