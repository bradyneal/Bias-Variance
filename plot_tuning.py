from parsing import parse_validations_table
import os
import numpy as np
import matplotlib.pyplot as plt

RES_PATH = '/media/mattscicluna/Backups/Projects/CourseWork-MILA/IFT6085/inf-paths-results/hyperparam_out'
files_dir = os.listdir(RES_PATH)

to_plot = np.zeros([len(files_dir), 5, 5])  # size of table
hidden_size_list = []

i = 0
for file in files_dir:
    table, _, hidden_size = parse_validations_table(os.path.join(RES_PATH, file))
    to_plot[i] = table
    i += 1
    hidden_size_list.append(hidden_size)

colors = ['red', 'blue', 'green', 'yellow', 'orange']
# For best valid acc
for i in range(len(hidden_size_list)):
    plt.plot(to_plot[i, :, 0], to_plot[i, :, 1], color=colors[i], marker='o', linestyle='None',
             label='{}'.format(int(hidden_size_list[i])))
    plt.legend()
    plt.xlabel('Best learning rate')
    plt.ylabel('Early stopped validation accuracy')
    plt.title('Best learning rates per network size')
    plt.savefig('Hyperparam_optimization')
