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
from models import ShallowNet, DeepNet
from scipy.stats import chi2
import torch
from torchvision import datasets, transforms
plt.switch_backend('agg')

MNIST_TEST_SIZE = 10000
NUM_MNIST_CLASSES = 10

class Variance:
    def __init__(self, hidden_arr, num_layers, num_seeds, slurm_id, data_type, types=[2],
                 inter=0, data_model_comp_obj=None):
        self.hidden_arr = hidden_arr
        self.num_layers = num_layers
        self.num_seeds = num_seeds
        self.slurm_id = slurm_id
        self.data_type = data_type
        self.types = types
        self.inter = inter
        self.data_model_comp_obj = data_model_comp_obj

        print('''Calculating variance for numL: {}, num_seeds: {},
            slurm_id: {}, data_type: {}'''.format(hidden_arr, num_seeds,
            slurm_id, data_type))

        if types != [2] and self.data_model_comp_obj is None:
            raise Exception('Implement data model comp for other types')

    def load_data(self, num_hidden, run_i, type):
        seed = run_i
        if self.data_type == 'bitmap':
            data = load_fine_path_bitmaps(num_hidden, run_i, self.inter, self.slurm_id, type)

        elif self.data_type.startswith('evaluate'):
            if self.data_model_comp_obj is None:
                self.data_model_comp_obj = DataModelComp(DeepNet(100, num_hidden), train_val_split_seed=seed, num_train_after_split=100)
            self.data_model_comp_obj.load_saved_deep_net(100, num_hidden, run_i, self.slurm_id, self.inter)  # Definitely do to load saved model
            weight_norms = self.data_model_comp_obj.model.get_weight_norms()

            print("Model loaded")

            if self.data_type == 'evaluate_probabilities':
                values_returned_by_evaluate = self.data_model_comp_obj.evaluate(0, type, probs_required=True)
                log_prob = values_returned_by_evaluate[3]
                data = log_prob.exp()
                self.validation_loss = 1 - self.data_model_comp_obj.evaluate(0, 1)[0]
                self.train_loss =  1 - self.data_model_comp_obj.evaluate(0, 0)[0]
            elif self.data_type == 'evaluate_bitmaps':
                values_returned_by_evaluate = self.data_model_comp_obj.evaluate(0, type)
                data = values_returned_by_evaluate[2]

            loss = 1 - values_returned_by_evaluate[0]
            loss = torch.Tensor([loss])

        else:
            raise Exception('load_data does not handle %s as data_type. Please implement it' % (self.data_type))

        return_values = [data, loss]
        theta = [return_value.unsqueeze(0) for return_value in return_values]
        theta.append([weight_norm.unsqueeze(0) for weight_norm in weight_norms])
        return theta

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
            for num_hidden in self.num_layers:
                data_combined, losses_combined, weight_norms_combined = None, None, None
                train_combined, val_combined = [], []
                print('Running for num_hidden: {}'.format(num_hidden))
                for seed in range(0, self.num_seeds):
                    print('Running for seed: {}'.format(seed))

                    data, loss, weight_norms = self.load_data(num_hidden, seed, type)
                    print('data size:', data.size())
                    print('loss size:', loss.size())
                    for i,norm in enumerate(weight_norms):
                        print('weight_norms{} size: {}'.format(i,weight_norms[i].size()))

                    data_combined = torch.cat((data_combined, data), 0) if data_combined is not None else data
                    losses_combined = torch.cat((losses_combined, loss), 0) if losses_combined is not None else loss

                    if weight_norms_combined is None:
                        weight_norms_combined = [weight_norm for weight_norm in weight_norms]
                    else:
                        for i,norm in enumerate(weight_norms):
                            weight_norms_combined[i] = torch.cat((weight_norms_combined[i],weight_norms[i]),0)

                    val_combined.append(self.validation_loss)
                    train_combined.append(self.train_loss)

                # quantize variance combined
                data_combined_quantized = data_combined.cpu().numpy().flatten()
                plt.hist(data_combined_quantized, bins=100, density=True)
                plt.savefig('plots/2density_plot_for_hidden_{}_inter_{}.jpg'.format(num_hidden, self.inter))
                plt.close()

                save_probabilities(self.slurm_id, num_hidden, data_combined.cpu().numpy())
                save_train_errors(self.slurm_id, num_hidden, np.array(train_combined))
                # plot this
                print('variance_combined size:', data_combined.size())
                print('loss_combined size:', losses_combined.size())
                for i,norm in enumerate(weight_norms_combined):
                    print('weight_norms{}_combined size: {}'.format(i,norm.size()))

                if data_combined is None:
                    print('%s is none for num_hidden=%d, type=%d' %
                          (self.data_type, num_hidden, type))
                    continue
                print('Calculating mean')

                mean = torch.mean(data_combined, 0)
                #
                # # Calculate bias
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
                # biases.append(bias)
                ######################

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

                for i in weight_norms_combined:
                    weight_norms_by_hidden_layer.append([])

                for i,norm_combined in enumerate(weight_norms_combined):
                    mean_weight_norm = norm_combined.mean(0,keepdim=True)
                    std_weight_norm = norm_combined.std(0,keepdim=True)
                    weight_norms_by_hidden_layer[i].append([mean_weight_norm.item(),std_weight_norm.item(),std_weight_norm.item()])
                    print('weight_norm{} mean size: {}'.format(i,mean_weight_norm.size()))
                    print('weight_norm{} std size: {}'.format(i,std_weight_norm.size()))

                self.val_combined_by_hidden.append(val_combined)

            variances_by_train_val_test.append(variances_by_hidden_layer)
            losses_by_train_val_test.append(losses_by_hidden_layer)
            weight_norms_by_train_val_test.append(weight_norms_by_hidden_layer)

        variances = [variances_by_train_val_test, losses_by_train_val_test, weight_norms_by_train_val_test]
        print(variances)

        # Save variances
        save_variance_data(self.slurm_id, variances, 'all')
        # file_name = 'saved/{}_all.npy'.format(self.slurm_id)
        # np.save(file_name, variances)

        return variances  # , biases

    def plot_variances(self, variances):
        fig, ax = plt.subplots()
        plt.grid(True)

        hidden_arr_str = ','.join([str(i) for i in self.num_layers])

        quantities = ["Variance"] #, "Loss", "Weight Norm 1", "Weight Norm 2"]
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)
        for i, quantity in enumerate(quantities):
            print("QUANTITY: ", quantity)
            data = variances[i]
            for j, type in enumerate(self.types):
                # ax = axs[i // 2, i % 2]
                ax = axs[i]
                ax.grid(True)
                # ax.set_xscale('log')

                # To print log scale x axis for weight norm graphs
                # if i>=2:
                #     ax.set_yscale('log')

                ax.set_xlabel("Hidden layer size")
                ax.set_ylabel("%s" % (quantity))
                ax.set_title("Plot of %s for hidden_arr=%s" % (quantity, hidden_arr_str))

                print(quantity)
                # print(data[j])
                # print(len(data[j]))
                # print(data[j][0].size)
                theta = np.array(data[j])
                theta = np.reshape(theta,[len(self.num_layers),3])
                print(theta)
                input()
                ax.errorbar(self.num_layers, theta[:, 0], yerr=[theta[:, 1], theta[:, 2]])

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
        # print("VARIANCES SIZE", variances.shape)
        print(variances)
        self.plot_variances(variances)

        # dir_name = 'saved/'
        # file_name = '{}_variances.npy'.format(self.slurm_id)
        # np.save(dir_name+file_name, variances)
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