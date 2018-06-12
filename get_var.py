from plot_bias_and_variance import load_probabilities_and_get_variances, plot_variances_with_diffs, \
    load_probabilities_and_get_biases, plot_biases_with_diffs
import os, torch, argparse
from models import ThreeLayerNetCIFAR10
from DataModelComp import DataModelComp
import numpy as np

parser = argparse.ArgumentParser(description='hyperparam tuning for hidden size')
parser.add_argument('--step', type=int, help='which step to perform (0 = eval probs, '
                                             '1 = compute bias and vars, 2 = plot bias and vars)')

args = parser.parse_args()

HIDDEN_ARRS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # [1, 2, 5, 25, 100, 10000, 2500, 500]
HIDDEN_ARRS2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_seeds = 50

print('performing step {}'.format(args.step))
if args.step == 0:
    #  compute probabilities
    OUTPUT_DIR = '/media/mattscicluna/Backups/Projects/CourseWork-MILA/IFT6085/inf-paths-results/zhang_model_corr'
    corr_list = os.listdir(OUTPUT_DIR)
    variances_by_corr_corrupt = []
    variances_by_corr_normal = []
    biases_by_corr_corrupt = []
    biases_by_corr_normal = []
    j = 0
    for corr in corr_list:
        dir = os.path.join(OUTPUT_DIR, corr)
        num_seeds = len(os.listdir(dir))
        outputs = torch.zeros(num_seeds, 10000, 10)
        ind = 0
        for model in os.listdir(dir):
            print('running for corr: {} model: {}'.format(corr, model))
            mod = torch.load(f=os.path.join(dir, model))
            deepish_net = ThreeLayerNetCIFAR10(512)
            deepish_net.load_state_dict(mod['state'])
            seed = int(model.split('-')[-1].split('.')[0]) + int(model.split('-')[3]) * 11
            # data_model_comp = DataModelComp(deepish_net, batch_size=128, test_batch_size=128, epochs=200,
            #                                lr=1000, decay=True, step_size=1, gamma=0.95, momentum=0.9,
            #                                no_cuda=False, seed=seed, log_interval=1000,
            #                                run_i=1, save_interval=None, data='CIFAR10', corruption=float(corr))
            # _, _, _, output = data_model_comp.evaluate_test(cur_iter=1)
            # outputs_corrupt[ind] = output.exp().detach()
            # corrupted_labels = np.array(data_model_comp.test_loader.dataset.test_labels)

            data_model_comp = DataModelComp(deepish_net, batch_size=128, test_batch_size=128, epochs=200,
                                            lr=1000, decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                            no_cuda=False, seed=seed, log_interval=1000,
                                            run_i=1, save_interval=None, data='CIFAR10', corruption=0)
            _, _, _, output = data_model_comp.evaluate_test(cur_iter=1)
            outputs[ind] = output.exp().detach()
            ind += 1

        np.save('saved/probabilities/shallow{}_job0000000.npy'.format(HIDDEN_ARRS2[j]), outputs)
        j += 1

if args.step == 1:
    #  compute bias and variance
    load_probabilities_and_get_variances('0000000', HIDDEN_ARRS2)
    load_probabilities_and_get_biases('0000000', HIDDEN_ARRS2)

if args.step == 2:
    import matplotlib.pyplot as plt

    from plot_bias_and_variance import *
    from plotting import run_fig_extras

    base_dir = 'plots/'
    base_final_dir = 'plots/final/'

    labelsize = 15

    # Figure 11
    hidden_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plot_biases_with_diffs('0000000', hidden_arr, label='Bias')
    plot_variances_with_diffs('0000000', hidden_arr, label='Variance')
    plt.ylabel('Bias and Variance', fontsize=labelsize)
    plt.xlabel('Proportion Corrupted', fontsize=labelsize)
    plt.xscale('linear')
    plt.savefig(base_dir + 'final_11.pdf')
    plt.savefig(base_final_dir + 'zhang_rep.pdf')
    plt.close()

    # Figure 12
    plot_losses_and_std('0000000', hidden_arr, label='Test Error')
    #plot_train_losses_and_std('0000000', hidden_arr, label='Train Error')
    plt.ylabel('Average Error', fontsize=labelsize)
    plt.xlabel('Proportion Corrupted', fontsize=labelsize)
    plt.xscale('linear')
    plt.legend().set_visible(False)
    plt.savefig(base_dir + 'final_12.pdf')
    plt.savefig(base_final_dir + 'error_zhang_rep.pdf')
    plt.close()
