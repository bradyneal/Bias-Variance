import matplotlib.pyplot as plt
import numpy as np
import os

from fileio import MODEL_DIR, get_hyperparam_indi_plot_path, get_hyperparam_main_plot_path
from parsing import parse_validations
from plotting import plot_line_with_normal_errbars, run_fig_extras


def get_best_learning_rates(job_ids, hidden_arr, learning_rates, num_seeds, option='best'):
    first_job_id = job_ids[0]
    job_index = 0
    best_learning_rates = []

    for num_hidden in hidden_arr:
        mean_accuracies, std_accuracies = [], []

        for learning_rate in learning_rates:
            accuracies = []

            for seed in range(num_seeds):
                validation_path = os.path.join(MODEL_DIR, str(job_ids[job_index]), 'joblog.log')
                best_accuracy, last_accuracy, _ = parse_validations(validation_path)
                if option == 'last':
                    accuracies.append(last_accuracy)
                elif option == 'best':
                    accuracies.append(best_accuracy)
                else:
                    raise Exception('option should be either best or last')
                job_index += 1

            accuracies = np.array(accuracies)
            mean_accuracy, std_accuracy = np.mean(accuracies), np.std(accuracies)

            mean_accuracies.append(mean_accuracy)
            std_accuracies.append(std_accuracy)

        plot_line_with_normal_errbars(learning_rates, mean_accuracies,
            std_accuracies, xlabel='Learning rate', ylabel='Mean of {} accuracies'.format(option),
            title='Plot for hidden size = {}'.format(num_hidden), filename=get_hyperparam_indi_plot_path(first_job_id, num_hidden, option))
        best_lr_index = np.argmax(mean_accuracies)
        best_learning_rate = learning_rates[best_lr_index]
        best_learning_rates.append(best_learning_rate)

    plt.plot(hidden_arr, best_learning_rates)
    run_fig_extras(xlabel='Number of hidden units', ylabel='Best learning rate',
                   xscale='log', filename=get_hyperparam_main_plot_path(first_job_id))
    return best_learning_rates
