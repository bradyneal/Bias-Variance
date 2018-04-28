# To run.
# from variance import Variance
# Variance(fill arguments here).calculate_plot_and_return_variances()

import torch
import os.path
import numpy as np
import matplotlib.pyplot as plt
from DataModelComp import DataModelComp
from fileio import load_fine_path_bitmaps
from models import ShallowNet
plt.switch_backend('agg')


class Variance:
    def __init__(self, hidden_arr, num_seeds, slurm_id, data_type, types=[2],
                 inter=0, data_model_comp_obj=None):
        self.hidden_arr = hidden_arr
        self.num_seeds = num_seeds
        self.slurm_id = slurm_id
        self.data_type = data_type
        self.types = types
        self.inter = inter
        self.data_model_comp_obj = data_model_comp_obj

        print('''Calculating variance for num_hidden_arr: {}, num_seeds: {},
            slurm_id: {}, data_type: {}'''.format(hidden_arr, num_seeds,
            slurm_id, data_type))

        if types != [2] and self.data_model_comp_obj is None:
            raise Exception('Implement data model comp for other types')

    def load_data(self, num_hidden, run_i, type):
        if self.data_type == 'bitmap':
            data = load_fine_path_bitmaps(num_hidden, run_i, self.inter, self.slurm_id, type)
            print('Line 35:', data)
        elif self.data_type.startswith('evaluate'):
            if self.data_model_comp_obj is None:
                self.data_model_comp_obj = DataModelComp(ShallowNet(num_hidden))
            self.data_model_comp_obj.load_saved_shallow_net(num_hidden, run_i, self.slurm_id, self.inter)  # Definitely do to load saved model

            if self.data_type == 'evaluate_test_error':  # DO NOT USE
                raise Exception('Check if this is implemented correctly')
                data = self.data_model_comp_obj.evaluate(0, type)[0]
            elif self.data_type == 'evaluate_probabilities':
                log_prob = self.data_model_comp_obj.evaluate(0, type, probs_required=True)[3]
                data = log_prob.exp()
            elif self.data_type == 'evaluate_bitmaps':
                data = self.data_model_comp_obj.evaluate(0, type)[2]

        else:
            raise Exception('load_data does not handle %s as data_type. Please implement it' % (data_type))

        return data.unsqueeze(0)

    def calculate_variance(self, bitmaps, mean):
        return torch.mean((bitmaps - mean.unsqueeze(0)) ** 2)

    def get_variances(self):
        if not isinstance(self.types, list):
            raise Exception('In get_variances, parameter types should be a list')

        variances_by_train_val_test = []
        for type in self.types:
            variances_by_hidden_layer = []
            for num_hidden in self.hidden_arr:
                data_combined = None
                print('Running for num_hidden: {}'.format(num_hidden))
                for seed in range(self.num_seeds):
                    print('Running for seed: {}'.format(seed))

                    data = self.load_data(num_hidden, seed, type)
                    print('Line 69:', data, 'size:', data.size())
                    if data_combined is None:
                        data_combined = data
                    else:
                        print('Line 73', data_combined.shape, data.shape)
                        data_combined = torch.cat((data_combined, data), 0)
                        print('Line 75', data_combined.shape, data.shape)

                    print('Line 76 data_combined:', data_combined, 'size:', data_combined.size())

                if data_combined is None:
                    print('%s is none for num_hidden=%d, type=%d' %
                          (self.data_type, num_hidden, type))
                    continue
                print('Calculating mean')

                print('Data combined:', data_combined, data_combined.size())
                mean = torch.mean(data_combined, 0)
                print('Mean:', mean, mean.size())
                print('Mean unsqueezed:', mean.unsqueeze(0), mean.unsqueeze(0).size())
                print('Calculating variance')
                variance = self.calculate_variance(data_combined, mean)
                print('Appending variance')
                variances_by_hidden_layer.append(variance)

            variances_by_train_val_test.append(variances_by_hidden_layer)

        return np.array(variances_by_train_val_test)

    def plot_variances(self, variances):
        fig, ax = plt.subplots()
        plt.grid(True)

        labels = ["Training", "Validation", "Test"]
        for i, type in enumerate(self.types):
            plt.semilogx(self.hidden_arr, variances[i], label=labels[type])

        plt.xlabel("Hidden layer size")
        plt.ylabel("Variance accross %ss" % (self.data_type))

        ax.legend(loc='lower left', shadow=True, fontsize='x-large')
        hidden_arr_str = ','.join([str(i) for i in self.hidden_arr])
        plt.title("Plot of variance across %ss for hidden_arr=%s" %
                  (self.data_type, hidden_arr_str))

        dir_name = 'plots/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = '{}_{}.jpg'.format(self.slurm_id, self.data_type)
        plt.savefig(dir_name + file_name)
        plt.close()

    # TODO: Use mean and variances to plot this. Can do later
    # def plot_loss_with_error_bars(hidden_arr, ):
    #     means, variances
    #     make plot over here

    def calculate_plot_and_return_variances(self):
        variances = self.get_variances()
        print(variances)
        self.plot_variances(variances)
        return variances
