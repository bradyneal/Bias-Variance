from fileio import load_probabilities, save_variance_diffs, load_variance_diffs, save_bias_diffs, load_bias_diffs, load_train_errors
import math
import numpy as np
from plotting import plot_line_with_errbars, plot_line_with_normal_errbars, plot_line
from parsing_train_error import parse_train_errors

from test_y_onehot import get_test_y_onehot, get_train_y_onehot


def calculate_variance(bitmaps):
    mean = np.mean(bitmaps, 0)
    return np.mean((bitmaps - np.expand_dims(mean, axis=0)) ** 2)


def get_variance(slurm_id, num_hidden,inter):
    '''
    Returns the variance for a slurm id (corresponding to an experiment) and a hidden size.
    '''
    probabilities = load_probabilities(slurm_id, num_hidden,inter)
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


def get_bias(slurm_id, num_hidden, inter):
    '''
    Returns the variance for a slurm id (corresponding to an experiment) and a hidden size.
    '''
    test_y_onehot = get_test_y_onehot()
    probabilities = load_probabilities(slurm_id, num_hidden,inter)
    return calculate_bias(probabilities, test_y_onehot)


def load_probabilities_and_get_variances(slurm_id, hidden_arr,inter=0, num_bootstrap=10000):
    '''
    Loads saved probabilities, calculates differences using bootstrapping from
    the value of variance computed using all the seeds and saves those diffs.
    Prerequisite: Probabilities should be saved earlier using the
    save_probabilities function with dimension (num_seeds, num_test_examples,
    probabilities_for_each_example), eg. (50, 10000, 10) for 50 seeds for MNIST.
    '''
    for num_hidden in hidden_arr:
        probabilities = load_probabilities(slurm_id, num_hidden, inter)
        original_variance = calculate_variance(probabilities)

        diffs = []
        for i in range(num_bootstrap):
            indices = np.random.choice(50, 50, replace=True)
            if slurm_id == 195683:
                indices = np.random.choice(28, 28, replace=True)
            elif slurm_id == 195684:
                indices = np.random.choice(43, 43, replace=True)
            bootstrap_probabilities = probabilities[indices]
            bootstrap_variance = calculate_variance(bootstrap_probabilities)
            diff_variance = (bootstrap_variance - original_variance)
            diffs.append(diff_variance)

        save_variance_diffs(slurm_id, num_hidden, diffs)


def find_probabilities_for_sampling(probabilities, sampling_no, num_initializations_per_split):
    return probabilities[sampling_no * num_initializations_per_split:
        (sampling_no+1) * num_initializations_per_split]

def find_probabilities_for_optimization(probabilities, sampling_no, num_initializations_per_split):
    return probabilities[sampling_no::num_initializations_per_split]

def load_probabilities_and_get_first_term(slurm_id, hidden_arr, num_initializations_per_split, inter=0, reverse=False):
    first_terms = []
    for num_hidden in hidden_arr:
        probabilities = load_probabilities(slurm_id, num_hidden, inter)

        num_samplings = probabilities.shape[0] // num_initializations_per_split
        individual_variances = []
        for sampling_no in range(num_samplings):
            if not reverse:
                probabilities_for_this_sampling = find_probabilities_for_sampling(probabilities, sampling_no, num_initializations_per_split)
            else:
                probabilities_for_this_sampling = find_probabilities_for_optimization(probabilities, sampling_no, num_initializations_per_split)
            individual_variance = calculate_variance(probabilities_for_this_sampling)
            individual_variances.append(individual_variance)

        first_terms.append(np.mean(np.array(individual_variances)))
    return first_terms


def load_probabilities_and_get_second_term(slurm_id, hidden_arr, num_initializations_per_split, inter=0, reverse=False):
    second_terms = []
    for num_hidden in hidden_arr:
        probabilities = load_probabilities(slurm_id, num_hidden, inter)

        num_samplings = probabilities.shape[0] // num_initializations_per_split

        expected_probabilities_shape = list(probabilities.shape)
        expected_probabilities_shape[0] = num_samplings

        expected_probabilities = np.zeros(expected_probabilities_shape)

        for sampling_no in range(num_samplings):
            if not reverse:
                probabilities_for_this_sampling = find_probabilities_for_sampling(probabilities, sampling_no, num_initializations_per_split)
            else:
                probabilities_for_this_sampling = find_probabilities_for_optimization(probabilities, sampling_no, num_initializations_per_split)
            expected_probabilities[sampling_no] = np.mean(probabilities_for_this_sampling, 0)

        second_terms.append(calculate_variance(expected_probabilities))

    return second_terms


def load_probabilities_and_get_biases(slurm_id, hidden_arr, inter =0, num_bootstrap=10000):
    test_y_onehot = get_test_y_onehot()
    for num_hidden in hidden_arr:
        probabilities = load_probabilities(slurm_id, num_hidden, inter)
        original_variance = calculate_bias(probabilities, test_y_onehot)

        diffs = []
        for i in range(num_bootstrap):
            indices = np.random.choice(50, 50, replace=True)
            if slurm_id == 195683:
                indices = np.random.choice(28, 28, replace=True)
            elif slurm_id == 195684:
                indices = np.random.choice(43, 43, replace=True)
            bootstrap_probabilities = probabilities[indices]
            bootstrap_variance = calculate_bias(bootstrap_probabilities, test_y_onehot)
            diff_variance = (bootstrap_variance - original_variance)
            diffs.append(diff_variance)

        save_bias_diffs(slurm_id, num_hidden, diffs)


def load_probabilities_and_get_losses_and_std(slurm_id, hidden_arr,inter =0):
    test_y_onehot = get_test_y_onehot()
    average_losses, stds = [], []
    for num_hidden in hidden_arr:
        probabilities = load_probabilities(slurm_id, num_hidden, inter)
        losses = calculate_losses(probabilities, test_y_onehot)

        average_loss = np.mean(losses)
        std = np.std(losses)/math.sqrt(len(losses))

        average_losses.append(average_loss)
        stds.append(std)

    return average_losses, stds


def load_train(slurm_id, hidden_arr):
    average_losses, std = [], []
    data = np.load("./cifar_train.npy")
    average_losses = np.mean(data,1)
    stds = np.std(data,1)/math.sqrt(50)
    
    return average_losses, stds

def plot_train(slurm_id, hidden_arr, label=None, marker=None):
    average_losses, stds = load_train(slurm_id, hidden_arr)
    # average_losses[-1] = 0
    # stds[-1] = stds[-2]
    average_losses = average_losses[:5]
    stds = stds[:5]
    plot_line_with_normal_errbars(hidden_arr, average_losses, stds, xlabel="Number of hidden units", ylabel="Average error", grid=True, xscale="log", label=label, filename="plots/{}_train_error.pdf".format(slurm_id), marker=marker)

def plot_losses_and_std(slurm_id, hidden_arr, label=None, marker=None, inter=0):
    average_losses, stds = load_probabilities_and_get_losses_and_std(slurm_id, hidden_arr,inter)
    plot_line_with_normal_errbars(hidden_arr, average_losses, stds,
        xlabel='Number of hidden units', ylabel='Average error',
        grid=True, xscale='log', label=label,
        filename='plots/{}_test_error.pdf'.format(slurm_id), marker=marker)


def load_train_losses_and_get_average_and_std(slurm_id, hidden_arr):
    average_losses, stds = [], []
    losses_all = parse_train_errors(slurm_id, hidden_arr)
    for i, num_hidden in enumerate(hidden_arr):
        # losses = load_train_errors(slurm_id, num_hidden)
        losses = np.array(losses_all[i])

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


def find_variances(slurm_id, hidden_arr, inter=0):
    variances, lower_diffs, upper_diffs = [], [], []
    for num_hidden in hidden_arr:
        original_variance = get_variance(slurm_id, num_hidden, inter)
        variances.append(original_variance)
    return variances


def find_variances_and_diffs(slurm_id, hidden_arr, inter):
    variances, lower_diffs, upper_diffs = [], [], []
    for num_hidden in hidden_arr:
        original_variance = get_variance(slurm_id, num_hidden,inter)
        variances.append(original_variance)

        diffs = load_variance_diffs(slurm_id, num_hidden)
        print(np.mean(np.array(diffs)))
        diffs = list(sorted(diffs))

        upper_diff = get_percentile(diffs, 0.995)
        lower_diff = -get_percentile(diffs, 0.005)

        upper_diffs.append(upper_diff)
        lower_diffs.append(lower_diff)

    return variances, lower_diffs, upper_diffs


def find_biases_and_diffs(slurm_id, hidden_arr, inter, upper_percentile=0.995,
    lower_percentile=0.005):
    '''
    Loads the differences calculated using bootstrapping, finds error bars using
    those (calculated using the percentile values) and returns the error bars.
    Prerequisite: diffs should be saved earlier using the
    load_probabilities_and_get_variances function.
    '''
    variances, lower_diffs, upper_diffs = [], [], []
    for num_hidden in hidden_arr:
        original_variance = get_bias(slurm_id, num_hidden, inter)
        variances.append(original_variance)

        diffs = load_bias_diffs(slurm_id, num_hidden)
        print(np.mean(np.array(diffs)))
        diffs = list(sorted(diffs))

        upper_diff = get_percentile(diffs, upper_percentile)
        lower_diff = -get_percentile(diffs, lower_percentile)

        upper_diffs.append(upper_diff)
        lower_diffs.append(lower_diff)

    return variances, lower_diffs, upper_diffs


def plot_three_variances(slurm_id, hidden_arr, num_initializations_per_split=10, inter=0, reverse=False):
    variances = find_variances(slurm_id, hidden_arr, inter)
    first_terms = load_probabilities_and_get_first_term(slurm_id, hidden_arr, num_initializations_per_split, inter, reverse)
    second_terms = load_probabilities_and_get_second_term(slurm_id, hidden_arr, num_initializations_per_split, inter, reverse)

    plot_line(hidden_arr, variances, label='Joint Variance', xscale='log', xlabel='Number of hidden units',
    ylabel='Variance', title='Law of Total Variance for MNIST full data', filename='plots/{}_all_variances.pdf'.format(slurm_id))
    
    if not reverse:
        plot_line(hidden_arr, first_terms, label='Variance due to Initialization', xscale='log', xlabel='Number of hidden units',
    ylabel='Variance', title='Law of Total Variance for MNIST full data', filename='plots/{}_all_variances.pdf'.format(slurm_id))
        plot_line(hidden_arr, second_terms, label='Variance due to Sampling', xscale='log', xlabel='Number of hidden units',
    ylabel='Variance', title='Law of Total Variance for MNIST full data', filename='plots/{}_all_variances.pdf'.format(slurm_id))
    else:
        plot_line(hidden_arr, first_terms, label='Variance due to Sampling (Reversed)', xscale='log', xlabel='Number of hidden units',
    ylabel='Variance', title='Law of Total Variance for MNIST full data', filename='plots/{}_all_variances.pdf'.format(slurm_id))
        plot_line(hidden_arr, second_terms, label='Variance due to Initialization (Reversed)', xscale='log', xlabel='Number of hidden units',
    ylabel='Variance', title='Law of Total Variance for MNIST full data', filename='plots/{}_all_variances.pdf'.format(slurm_id))



def plot_variances_with_diffs(slurm_id, hidden_arr, title=None, label=None, marker=None, inter=0):
    '''
    Loads the differences calculated using bootstrapping, finds error bars using
    those and plots the graph for variance and error bars.
    Prerequisite: diffs should be saved earlier using the
    load_probabilities_and_get_variances function.
    '''
    variances, lower_diffs, upper_diffs = find_variances_and_diffs(slurm_id, hidden_arr,inter)
    print(variances)
    plot_line_with_errbars(hidden_arr, variances, lower_diffs, upper_diffs,
        grid=True, xscale='log', ylabel='Bias and Variance', xlabel='Number of hidden units',
        filename='plots/{}_variance.pdf'.format(slurm_id), title=title, label=label, marker=marker)


def plot_biases_with_diffs(slurm_id, hidden_arr, title=None, label=None, marker=None, inter=0):
    variances, lower_diffs, upper_diffs = find_biases_and_diffs(slurm_id, hidden_arr, inter)
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

def plot_train_errors():
    data = np.load("./cifar_train.npy")

