from DataModelComp import DataModelComp
from models import ShallowNetCIFAR10, ThreeLayerNetCIFAR10, AlexNetCIFAR10, InceptionCIFAR10
from variance import calculate_variance
from matplotlib import pyplot as plt
import numpy as np
import torch
import os

run_exp_a = False
run_exp_b_c = False
run_exp_d = False
run_exp_e = True

#import getpass

#USERNAME = getpass.getuser()
#OUTPUT_DIR = os.path.join('/data/milatmp1', USERNAME, 'information-paths')

#  Local changes only
OUTPUT_DIR = os.path.join(os.getcwd(), '/matt_folder')

#  Learning Curves
if run_exp_a:
    fig_1 = []
    for k in [0, 1]:
        deepish_net = ThreeLayerNetCIFAR10(num_hidden=512)
        data_model_comp = DataModelComp(deepish_net, batch_size=128, test_batch_size=128, epochs=60,
                                        lr=0.01, decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                        no_cuda=False, seed=False, log_interval=100,
                                        run_i=0, save_interval=None, data='CIFAR10', corruption=k)
        _, _, steps, train_loss_to_return = data_model_comp.train(eval_path=False, early_stopping=False,
                                                                  eval_train_every=True)

        fig_1.append(train_loss_to_return)

    with open(OUTPUT_DIR+'/fig_a_series_test', 'wb') as f:
        np.save(file=f, arr=np.array(fig_1))

    plt.figure()
    plt.title('Learning Curves')
    plt.plot(fig_1[0], color='blue', label='true labels')
    plt.plot(fig_1[1], color='red', label='random labels')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(OUTPUT_DIR+'/result_fig_1')

    print('done?')


if run_exp_b_c:
    lr_list = [0.01, 0.01, 0.1]
    corruption_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    label_corruption_threshold = [39998.0 / 40000, 0.9982, 39998.0 / 40000]  # paper uses [1, 0.9982, 1]
    network_names = ['3 Layer MLP', 'AlexNet', 'Inception']
    colors = ['blue', 'green', 'black']

    #  Label Corruption
    try:
        fig_2_3 = np.load(OUTPUT_DIR+'/fig_bc_series')
        print('loaded previous results')
    except:
        fig_2_3 = np.zeros(
            shape=(3, 11, 2))  # 3 diff networks x 11 levels of corruption x time to overfit

    for i, network in enumerate(network_names):
        print('computing for: {} ...'.format(network))
        bitmaps = torch.FloatTensor(0, 1)
        for j, corr in enumerate(corruption_list):
            if not (fig_2_3[i, j] == [0, 0]).all():
                print('alread computed! skipping ...')
            else:
                if network == '3 Layer MLP':
                    net = ThreeLayerNetCIFAR10(num_hidden=512)
                if network == 'AlexNet':
                    net = AlexNetCIFAR10()
                if network == 'Inception':
                    net = InceptionCIFAR10(use_batch_norm=True)

                data_model_comp = DataModelComp(net, batch_size=128, test_batch_size=128, epochs=200,
                                                lr=lr_list[i], decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                                no_cuda=False, seed=False, log_interval=1000,
                                                run_i=0, save_interval=None, data='CIFAR10', corruption=corr)
                _, _, steps = data_model_comp.train(eval_path=False, early_stopping=False,
                                                    train_to_overfit=label_corruption_threshold[i],
                                                    eval_train_every=False)
                test_error, _, _ = data_model_comp.evaluate_test(cur_iter=1)

                fig_2_3[i, j, 0] = steps
                fig_2_3[i, j, 1] = test_error

                with open(OUTPUT_DIR+'/fig_bc_series', 'wb') as f:
                    np.save(file=f, arr=fig_2_3)

    with open(OUTPUT_DIR+'/fig_bc_series', 'wb') as f:
        np.save(file=f, arr=fig_2_3)
    with open(OUTPUT_DIR+'/bitmaps', 'wb') as f:
        np.save(file=f, arr=bitmaps)

    # plot final results
    mean = torch.mean(bitmaps, 1)
    variance = calculate_variance(bitmaps, mean)

    # to load run: results = np.load(file='matt_folder/fig_bc_series')

    plt.figure()
    plt.title('Convergence Slowdown')
    fig_2_3[:, :, 0] /= fig_2_3[:, 0, 0].reshape((3, 1))
    for i, network in enumerate(network_names):
        plt.plot(np.array(corruption_list), fig_2_3[i, :, 0], color=colors[i], label=network)
    plt.xlabel('Label Corruption')
    plt.ylabel('Time to Overfit')
    plt.legend(loc='lower left')
    plt.savefig(OUTPUT_DIR+'/result_fig_2')

    plt.figure()
    plt.title('Generalization Error Growth')
    for i, network in enumerate(network_names):
        plt.plot(np.array(corruption_list), 1-fig_2_3[i, :, 1], color=colors[i], label=network)
    plt.axhline(y=0.9, linestyle='--', color='red')
    plt.xlabel('Label Corruption')
    plt.ylabel('Test Error')
    plt.legend(loc='upper left')
    plt.savefig(OUTPUT_DIR+'/result_fig_3')

    print('done?')

if run_exp_d:
    #  Label Corruption
    lr_list = [0.01, 0.01, 0.1]
    size_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    label_size_threshold = [39998.0 / 40000, 0.9982, 39998.0 / 40000]  # paper uses [1, 0.9982, 1]
    network_names = ['3 Layer MLP', 'AlexNet', 'Inception']
    colors = ['blue', 'green', 'black']

    try:
        fig_2_3 = np.load(OUTPUT_DIR+'/fig_bc_series')
        print('loaded previous results')
    except:
        fig_2_3 = np.zeros(
            shape=(3, 10, 2))  # 3 diff networks x 10 dataset sizes x time to overfit OR test error

    for i, network in enumerate(network_names):
        print('computing for: {} ...'.format(network))
        for j, corr in enumerate(size_list):
            if not (fig_2_3[i, j] == [0, 0]).all():
                print('alread computed! skipping ...')
            else:
                if network == '3 Layer MLP':
                    net = ThreeLayerNetCIFAR10(num_hidden=512)
                if network == 'AlexNet':
                    net = AlexNetCIFAR10()
                if network == 'Inception':
                    net = InceptionCIFAR10(use_batch_norm=True)

                data_model_comp = DataModelComp(net, batch_size=128, test_batch_size=128, epochs=200,
                                                lr=lr_list[i], decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                                no_cuda=False, seed=False, log_interval=1000,
                                                run_i=0, save_interval=None, data='CIFAR10', corruption=0,
                                                to_exclude=corr)
                _, _, steps = data_model_comp.train(eval_path=False, early_stopping=False,
                                                    train_to_overfit=label_size_threshold[i],
                                                    eval_train_every=False)
                test_error = data_model_comp.evaluate_test(cur_iter=1)

                fig_2_3[i, j, 0] = steps
                fig_2_3[i, j, 1] = test_error[0]
                with open(OUTPUT_DIR+'fig_bc_series', 'wb') as f:
                    np.save(file=f, arr=fig_2_3)
                    print('saved up to {} of {}'.format(corr, network))

    with open(OUTPUT_DIR+'/fig_bc_series', 'wb') as f:
        np.save(file=f, arr=fig_2_3)

    plt.figure()
    plt.title('Convergence Slowdown')
    fig_2_3[:, :, 0] /= fig_2_3[:, 0, 0].reshape((3, 1))
    for i, network in enumerate(network_names):
        plt.plot(np.array(size_list), fig_2_3[i, :, 0], color=colors[i], label=network)
    plt.xlabel('Percent of Dataset Removed')
    plt.ylabel('Time to Overfit')
    plt.legend(loc='lower left')
    plt.savefig(OUTPUT_DIR+'/result_fig_2')

    plt.figure()
    plt.title('Generalization Error Growth')
    for i, network in enumerate(network_names):
        plt.plot(np.array(size_list), 1-fig_2_3[i, :, 1], color=colors[i], label=network)
    plt.axhline(y=0.9, linestyle='--', color='red')
    plt.xlabel('Percent of Dataset Removed')
    plt.ylabel('Test Error')
    plt.legend(loc='upper left')
    plt.savefig(OUTPUT_DIR+'/result_fig_3')


# Check params
def get_weights(net):
    total_weights = 0
    for k in net.state_dict().keys():
        print(k)
        print(net.state_dict()[k].numel())
        total_weights += net.state_dict()[k].numel()
    print('total weights: {}'.format(total_weights))


"""
# check how values match up
Zhang = np.load('matt_folder/ZhangRep/fig_bc_series')
Baseline = np.load('matt_folder/baseline/fig_bc_series')

Zhang = Zhang[:, :10, 1]
size_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
colors = ['blue', 'green', 'black']
network_names = ['3 Layer MLP', 'AlexNet', 'Inception']
Baseline = Baseline[:, :, 1]
# (1 - 1/k)c + (1 - p_c)(1 - c)
Error = (1-np.array(Baseline))*(1-np.array(size_list)) + (1-1/10)*np.array(size_list)

plt.figure()
plt.title('Generalization Error Growth')
for i, network in enumerate(network_names):
    plt.plot(np.array(size_list), Error[i], color=colors[i], label='Estimated {}'.format(network), linestyle='--')
    plt.plot(np.array(size_list), 1-Zhang[i], color=colors[i], label='Random Labels {}'.format(network))
plt.axhline(y=0.9, linestyle='--', color='red')
plt.xlabel('Percent of Dataset Removed')
plt.ylabel('Test Error')
plt.legend(loc='lower right')
plt.savefig('matt_folder/baseline/compare_vals')
"""

if run_exp_e:
    lr_list = [0.01, 0.01, 0.1]
    corruption_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    label_corruption_threshold = [39998.0 / 40000, 0.9982, 39998.0 / 40000]  # paper uses [1, 0.9982, 1]
    network_names = ['3 Layer MLP']
    colors = ['blue']
    num_runs = 10

    #  Label Corruption
    try:
        fig_e = np.load(OUTPUT_DIR+'/fig_e_series')
        print('loaded previous results')
    except:
        fig_e = np.zeros(
            shape=(num_runs, len(corruption_list), 2))  # 3 diff networks x 11 levels of corruption x time to overfit

    try:
        all_bitmaps = np.load(OUTPUT_DIR+'/bitmaps')
        print('loaded previous results')
    except:
        all_bitmaps = np.zeros(
            shape=(num_runs, len(corruption_list), 10000))  # 3 diff networks x 11 levels of corruption x bitmap len
        all_bitmaps += 11                 # 11 = no sample!

    for i in range(num_runs):
        print('computing for run {} ...'.format(i))
        #bitmaps = torch.FloatTensor(0, 1)
        for j, corr in enumerate(corruption_list):
            if not (fig_e[i, j] == [0, 0]).all():
                print('alread computed! skipping ...')
            else:
                net = ThreeLayerNetCIFAR10(num_hidden=512)

                data_model_comp = DataModelComp(net, batch_size=128, test_batch_size=128, epochs=1,
                                                lr=lr_list[0], decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                                no_cuda=False, seed=i+2018, log_interval=1000,
                                                run_i=i, save_interval=None, data='CIFAR10', corruption=corr)
                _, _, steps = data_model_comp.train(eval_path=False, early_stopping=False,
                                                    train_to_overfit=label_corruption_threshold[0],
                                                    eval_train_every=False)
                #test_error, _, bitmap = data_model_comp.evaluate_test(cur_iter=1)

                #bitmaps = torch.cat((bitmaps, bitmap), 1)

                #fig_e[i, j, 0] = steps
                #fig_e[i, j, 1] = test_error
                #all_bitmaps[i, j, :] = bitmap.squeeze()

                #with open(OUTPUT_DIR+'/fig_e_series', 'wb') as f:
                #    np.save(file=f, arr=fig_e)
                #with open(OUTPUT_DIR+'/bitmaps', 'wb') as f:
                #    np.save(file=f, arr=all_bitmaps)
                state = {'state': net.state_dict()}
                torch.save(state, f='matt_folder/model-corr-{}run-{}.pt'.format(corr, j))
                print('saved up to {} of run {}'.format(corr, i))

    #with open(OUTPUT_DIR+'/fig_e_series', 'wb') as f:
    #    np.save(file=f, arr=fig_e)
    #with open(OUTPUT_DIR+'/bitmaps', 'wb') as f:
    #    np.save(file=f, arr=all_bitmaps)
