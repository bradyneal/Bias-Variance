from parsing import parse_validations_table
import os
import numpy as np
import matplotlib.pyplot as plt

#RES_PATH = '/media/mattscicluna/Backups/Projects/CourseWork-MILA/IFT6085/inf-paths-results/hyperparam_out'
RES_PATH = '/media/mattscicluna/Backups/Projects/CourseWork-MILA/IFT6085/inf-paths-results/sgd_opt'
files_dir = os.listdir(RES_PATH)

show_top = False
optimizer = 'sgd'
early_stopped = False

if show_top:
    to_plot = np.zeros([len(files_dir), 5, 5])  # size of table
else:
    if optimizer == 'adam':
        to_plot = np.zeros([len(files_dir), 5*60*3, 5])  # size of table
    elif optimizer == 'lbfgs':
        to_plot = np.zeros([len(files_dir), 5*20, 5])  # size of table
    elif optimizer == 'batch_gd':
        to_plot = np.zeros([len(files_dir), 50, 5])  # size of table
    elif optimizer == 'sgd':
        to_plot = np.zeros([len(files_dir), 50, 5])  # size of table

hidden_size_list = []

i = 0
for file in files_dir:
    table, full_table, hidden_size = parse_validations_table(os.path.join(RES_PATH, file))
    if show_top:
        to_plot[i] = table
    else:
        to_plot[i] = full_table
    i += 1
    hidden_size_list.append(hidden_size)

#np.save('hyperparam_output', to_plot)
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'pink', 'cyan', 'purple', 'violet', 'black']
# For best valid acc
for i in range(len(hidden_size_list)):
    if early_stopped:
        plt.plot(to_plot[i, :, 0], to_plot[i, :, 1], color=colors[i], marker='o', linestyle='None',
                 label='{}'.format(int(hidden_size_list[i])))
    else:
        plt.plot(to_plot[i, :, 0], to_plot[i, :, 2], color=colors[i], marker='o', linestyle='None',
                 label='{}'.format(int(hidden_size_list[i])))
    plt.legend(loc=4)
    plt.xlabel('Best learning rate')
    #plt.xlim(xmax=10e-1)  # restrict x axis for readability
    #plt.xlim(xmin=10e-5)
    plt.xscale("log")  # log scale
    plt.ylabel('Early stopped validation accuracy')
    plt.title('Best learning rates per network size for {}'.format(optimizer))
    if early_stopped:
        plt.savefig('Hyperparam_optimization-{}-early_stopped'.format(optimizer))
    else:
        plt.savefig('Hyperparam_optimization-{}-last_epoch'.format(optimizer))
