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
from fileio import load_fine_path_bitmaps, save_probabilities, save_variance_data, save_train_errors, load_train_loader
from models import ShallowNet
from scipy.stats import chi2
import torch
from torchvision import datasets, transforms
plt.switch_backend('agg')

MNIST_TEST_SIZE = 10000
NUM_MNIST_CLASSES = 10

train_errors = np.zeros([6,100])

class Variance:
    def __init__(self, hidden_arr, num_seeds, slurm_id, dataset='MNIST', data_type='evaluate_probabilities', types=[2],
                 inter=0, data_model_comp_obj=None):
        self.hidden_arr = hidden_arr
        self.num_seeds = num_seeds
        self.slurm_id = slurm_id
        self.data_type = data_type
        self.types = types
        self.inter = inter
        self.data_model_comp_obj = data_model_comp_obj
        self.dataset = dataset

        print('''Calculating variance for num_hidden_arr: {}, num_seeds: {},
            slurm_id: {}, data_type: {}'''.format(hidden_arr, num_seeds,
            slurm_id, data_type))

        if types != [2] and self.data_model_comp_obj is None and slurm_id != 166436:
            raise Exception('Implement data model comp for other types')

    def load_data(self, num_hidden, run_i, type):
        x = -1
        if num_hidden is 10:
            x = 0
        elif num_hidden is 100:
            x = 1
        elif num_hidden is 1000:
            x = 2
        elif num_hidden is 10000:
            x = 3
        elif num_hidden is 40000:
            x = 4
        else:
            x = 5

        seed = run_i
        if self.data_type == 'bitmap':
            data = load_fine_path_bitmaps(num_hidden, run_i, self.inter, self.slurm_id, type)

        elif self.data_type.startswith('evaluate'):
            if self.data_model_comp_obj is None:
                # TODO: bootstrap True or False. Fix the errors Matthew got so that you don't get those again
                if self.slurm_id == 166436:
                    self.data_model_comp_obj = DataModelComp(ShallowNet(num_hidden), train_val_split_seed=0, bootstrap=True, dataset=self.dataset)
                else:
                    print(self.dataset)
                    self.data_model_comp_obj = DataModelComp(ShallowNet(num_hidden, self.dataset), train_val_split_seed=seed, num_train_after_split=None, dataset=self.dataset)
            self.data_model_comp_obj.load_saved_shallow_net(num_hidden, run_i, self.slurm_id, self.inter)  # Definitely do to load saved model
            weight_norms1, weight_norms2 = self.data_model_comp_obj.model.get_weight_norms()

            if self.data_type == 'evaluate_probabilities':
                print(self.data_model_comp_obj.dataset)
                values_returned_by_evaluate = self.data_model_comp_obj.evaluate(0, type, probs_required=True)
                log_prob = values_returned_by_evaluate[3]
                data = log_prob.exp()
                self.validation_loss = 1 - self.data_model_comp_obj.evaluate(0, 1)[0]
                self.train_loss = 1 - self.data_model_comp_obj.evaluate(0, 0)[0]
                train_errors[x][run_i] = self.train_loss
            elif self.data_type == 'evaluate_bitmaps':
                values_returned_by_evaluate = self.data_model_comp_obj.evaluate(0, type)
                data = values_returned_by_evaluate[2]

            loss = 1 - values_returned_by_evaluate[0]
            loss = torch.Tensor([loss])

        else:
            raise Exception('load_data does not handle %s as data_type. Please implement it' % (self.data_type))

        return_values = [data, loss, weight_norms1, weight_norms2]
        return [return_value.unsqueeze(0) for return_value in return_values]

    # def calculate_variance(self, bitmaps):
    #     bitmaps = bitmaps.cpu().numpy()
    #     mean = np.mean(bitmaps, 0)
    #     return np.mean((bitmaps - np.expand_dims(mean, axis=0)) ** 2)

    def calculate_variance(self, bitmaps, mean):
        return torch.mean((bitmaps - mean.unsqueeze(0)) ** 2)

    def calculate_and_save_individual_variances(self, bitmaps):
        bitmaps = bitmaps.cpu().numpy()
        mean = np.mean(bitmaps, 0)
        mean = np.expand_dims(mean, axis=1)
        bitmaps = np.swapaxes(bitmaps, 0, 1)
        individual_variances = np.mean(bitmaps - mean, axis=(1, 2))

        # Save variances
        # dir_name = 'saved/'
        # file_name = '{}_individual_variances.npy'.format(self.slurm_id)
        # np.save(dir_name+file_name, individual_variances)
        save_variance_data(self.slurm_id, individual_variances, 'individual_variances')

    def get_variances(self):
        if not isinstance(self.types, list):
            raise Exception('In get_variances, parameter types should be a list')

        variances_by_train_val_test, losses_by_train_val_test, weight_norms_by_train_val_test = [], [], []
        biases = []
        for type in self.types:
            variances_by_hidden_layer, losses_by_hidden_layer, weight_norms_by_hidden_layer = [], [], []
            self.val_combined_by_hidden = []
            for num_hidden in self.hidden_arr:
                data_combined, losses_combined, weight_norms_combined = None, None, None
                train_combined, val_combined = [], []
                print('Running for num_hidden: {}'.format(num_hidden))
                for seed in range(0, self.num_seeds):
                    print('Running for seed: {}'.format(seed))

                    data, loss, _, _ = self.load_data(num_hidden, seed, type)
                    print('data size:', data.size())
                    print('loss size:', loss.size())

                    data_combined = torch.cat((data_combined, data), 0) if data_combined is not None else data
                    losses_combined = torch.cat((losses_combined, loss), 0) if losses_combined is not None else loss

                    val_combined.append(self.validation_loss)
                    train_combined.append(self.train_loss)

                # quantize variance combined
                data_combined_quantized = data_combined.cpu().numpy().flatten()
                #plt.hist(data_combined_quantized, bins=100, density=True)
                #plt.savefig('plots/2density_plot_for_hidden_{}_inter_{}.jpg'.format(num_hidden, self.inter))
                #plt.close()

                save_probabilities(self.slurm_id, num_hidden, self.inter, data_combined.cpu().numpy())

                mean = torch.mean(data_combined, 0)

                self.calculate_and_save_individual_variances(data_combined)
                variance = self.calculate_variance(data_combined, mean)
                variance = torch.Tensor([variance]).unsqueeze(0)
                relative_error = sqrt(2/self.num_seeds)
                std_variance = variance * relative_error
                variances_by_hidden_layer.append([variance.item(), std_variance.item(), std_variance.item()])
                print('variance mean size:', variance.size())
                print('variance std size:', std_variance.size())

                mean_loss = losses_combined.mean(0, keepdim=True)
                std_loss = losses_combined.std(0, keepdim=True)
                losses_by_hidden_layer.append([mean_loss.item(), std_loss.item(), std_loss.item()])
                print('loss mean size:', mean_loss.size())
                print('loss std size:', std_loss.size())

                self.val_combined_by_hidden.append(val_combined)

            variances_by_train_val_test.append(variances_by_hidden_layer)
            losses_by_train_val_test.append(losses_by_hidden_layer)

        variances = [variances_by_train_val_test, losses_by_train_val_test, weight_norms_by_train_val_test]

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
        np.save("./cifar_train.npy", train_errors)
        # print("VARIANCES SIZE", variances.shape)
        # print(variances)
        # self.plot_variances(variances)
        # print(variances)

        # # dir_name = 'saved/'
        # # file_name = '{}_variances.npy'.format(self.slurm_id)
        # # np.save(dir_name+file_name, variances)
        # print('Validation values combined:', self.val_combined_by_hidden)
        # return variances

    def get_conf_int(self, variance, conf_perc=.95):
         '''
         Return tuple (lower, upper) of confidence interval for variance.
         Assumption: random variables are independent
         (which is, of course, not true in the learning setting)
         '''
         return chi2.interval(alpha=conf_perc, df=self.num_seeds-1,
                              scale=variance/self.num_seeds)
