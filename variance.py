# This will make a plot of variances, test losses and weight norms. To run:
# from variance import Variance
# Variance(fill arguments here).calculate_plot_and_return_variances()

import torch
import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from DataModelComp import DataModelComp
from fileio import load_fine_path_bitmaps
from models import ShallowNet
from scipy.stats import chi2

import torch
from torchvision import datasets, transforms
plt.switch_backend('agg')

MNIST_TEST_SIZE = 10000
NUM_MNIST_CLASSES = 10


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

        elif self.data_type.startswith('evaluate'):
            if self.data_model_comp_obj is None:
                self.data_model_comp_obj = DataModelComp(ShallowNet(num_hidden))
            self.data_model_comp_obj.load_saved_shallow_net(num_hidden, run_i, self.slurm_id, self.inter)  # Definitely do to load saved model
            weight_norms1, weight_norms2 = self.data_model_comp_obj.model.get_weight_norms()

            if self.data_type == 'evaluate_probabilities':
                values_returned_by_evaluate = self.data_model_comp_obj.evaluate(0, type, probs_required=True)
                log_prob = values_returned_by_evaluate[3]
                data = log_prob.exp()
            elif self.data_type == 'evaluate_bitmaps':
                values_returned_by_evaluate = self.data_model_comp_obj.evaluate(0, type)
                data = values_returned_by_evaluate[2]

            loss = 1 - values_returned_by_evaluate[0]
            loss = torch.Tensor([loss])

        else:
            raise Exception('load_data does not handle %s as data_type. Please implement it' % (self.data_type))

        return_values = [data, loss, weight_norms1, weight_norms2]
        return [return_value.unsqueeze(0) for return_value in return_values]

    def calculate_variance(self, bitmaps, mean):
        return torch.mean((bitmaps - mean.unsqueeze(0)) ** 2)

    def get_variances(self):
        if not isinstance(self.types, list):
            raise Exception('In get_variances, parameter types should be a list')

        variances_by_train_val_test, losses_by_train_val_test, weight_norms1_by_train_val_test, weight_norms2_by_train_val_test = [], [], [], []
        for type in self.types:
            variances_by_hidden_layer, losses_by_hidden_layer, weight_norms1_by_hidden_layer, weight_norms2_by_hidden_layer = [], [], [], []
            for num_hidden in self.hidden_arr:
                data_combined, losses_combined, weight_norms1_combined, weight_norms2_combined = None, None, None, None
                print('Running for num_hidden: {}'.format(num_hidden))
                for seed in range(0, self.num_seeds):
                    print('Running for seed: {}'.format(seed))

                    data, loss, weight_norms1, weight_norms2 = self.load_data(num_hidden, seed, type)
                    print('data size:', data.size())
                    print('loss size:', loss.size())
                    print('weight_norms1 size:', weight_norms1.size())
                    print('weight_norms2 size:', weight_norms2.size())

                    data_combined = torch.cat((data_combined, data), 0) if data_combined is not None else data
                    losses_combined = torch.cat((losses_combined, loss), 0) if losses_combined is not None else loss
                    weight_norms1_combined = torch.cat((weight_norms1_combined, weight_norms1), 0) if weight_norms1_combined is not None else weight_norms1
                    weight_norms2_combined = torch.cat((weight_norms2_combined, weight_norms2), 0) if weight_norms2_combined is not None else weight_norms1

                print('variance_combined size:', data_combined.size())
                print('loss_combined size:', losses_combined.size())
                print('weight_norms1_combined size:', weight_norms1_combined.size())
                print('weight_norms2_combined size:', weight_norms2_combined.size())

                if data_combined is None:
                    print('%s is none for num_hidden=%d, type=%d' %
                          (self.data_type, num_hidden, type))
                    continue
                print('Calculating mean')

                mean = torch.mean(data_combined, 0)

                # ### Calculate bias ###
                # test = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
                # test_loader = torch.utils.data.DataLoader(test, batch_size=MNIST_TEST_SIZE)
                # _, y = next(iter(test_loader))
                #
                # # get one-hot encoding (should be a separate function)
                # y_onehot = torch.FloatTensor(MNIST_TEST_SIZE, NUM_MNIST_CLASSES)
                # y_onehot.zero_()
                # y_onehot.scatter_(1, y.unsqueeze(1), 1)
                #
                # bias = torch.mean(mean - y_onehot)
                # ######################

                variance = self.calculate_variance(data_combined, mean)
                variance = torch.Tensor([variance]).unsqueeze(0)
                relative_error = sqrt(2/self.num_seeds)
                std_variance = variance * relative_error
                variances_by_hidden_layer.append(torch.cat([variance, std_variance, std_variance]).numpy())
                print('variance mean size:', variance.size())
                print('variance std size:', std_variance.size())

                mean_loss = losses_combined.mean(0, keepdim=True)
                std_loss = losses_combined.std(0, keepdim=True)
                losses_by_hidden_layer.append(torch.cat([mean_loss, std_loss, std_loss]).numpy())
                print('loss mean size:', mean_loss.size())
                print('loss std size:', std_loss.size())

                mean_weight_norm1 = weight_norms1_combined.mean(0, keepdim=True)
                std_weight_norm1 = weight_norms1_combined.std(0, keepdim=True)
                weight_norms1_by_hidden_layer.append(torch.cat([mean_weight_norm1, std_weight_norm1, std_weight_norm1]).data.cpu().numpy())
                print('weight_norm1 mean size:', mean_weight_norm1.size())
                print('weight_norm1 std size:', std_weight_norm1.size())

                mean_weight_norm2 = weight_norms2_combined.mean(0, keepdim=True)
                std_weight_norm2 = weight_norms2_combined.std(0, keepdim=True)
                weight_norms2_by_hidden_layer.append(torch.cat([mean_weight_norm2, std_weight_norm2, std_weight_norm2]).data.cpu().numpy())
                print('weight_norm2 mean size:', mean_weight_norm2.size())
                print('weight_norm2 std size:', std_weight_norm2.size())

            variances_by_train_val_test.append(variances_by_hidden_layer)
            losses_by_train_val_test.append(losses_by_hidden_layer)
            weight_norms1_by_train_val_test.append(weight_norms1_by_hidden_layer)
            weight_norms2_by_train_val_test.append(weight_norms2_by_hidden_layer)

        return np.array([variances_by_train_val_test, losses_by_train_val_test, weight_norms1_by_train_val_test, weight_norms2_by_train_val_test])

    def plot_variances(self, variances):
        fig, ax = plt.subplots()
        plt.grid(True)

        hidden_arr_str = ','.join([str(i) for i in self.hidden_arr])

        quantities = ["Variance", "Loss", "Weight Norm 1", "Weight Norm 2"]
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
        for i, quantity in enumerate(quantities):
            print("QUANTITY:", quantity)
            data = variances[i]
            for j, type in enumerate(self.types):
                ax = axs[i // 2, i % 2]
                ax.grid(True)
                ax.set_xscale('log')

                # To print log scale x axis for weight norm graphs
                if i>=2:
                    ax.set_yscale('log')

                ax.set_xlabel("Hidden layer size")
                ax.set_ylabel("%s" % (quantity))
                ax.set_title("Plot of %s for hidden_arr=%s" % (quantity, hidden_arr_str))

                print(quantity)
                print(data[j])
                print(len(data[j]))
                print(data[j][0].size)
                ax.errorbar(self.hidden_arr, data[j][:, 0], yerr=[data[j][:, 1], data[j][:, 2]])

        dir_name = 'plots/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = '{}.jpg'.format(self.slurm_id)
        print(dir_name + file_name)
        fig.savefig(dir_name + file_name)
        fig.clf()

    def calculate_plot_and_return_variances(self):
        sns.set()
        variances = self.get_variances()
        print("VARIANCES SIZE", variances.shape)
        print(variances)
        self.plot_variances(variances)
        print(variances)

        dir_name = 'plots/'
        file_name = '{}.npy'.format(self.slurm_id)
        np.save(dir_name+file_name, variances)
        return variances

    def get_conf_int(self, variance, conf_perc=.95):
        '''
        Return tuple (lower, upper) of confidence interval for variance.

        Assumption: random variables are independent
        (which is, of course, not true in the learning setting)
        '''
        return chi2.interval(alpha=conf_perc, df=self.num_seeds-1,
                             scale=variance/self.num_seeds)
