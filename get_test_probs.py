import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from DataModelComp import DataModelComp
from models import ThreeLayerNetCIFAR10


def calculate_variance(bitmaps, mean):
    return torch.mean((bitmaps - mean.unsqueeze(0)) ** 2)


def calculate_bias(bitmaps, mean):
    # get one-hot encoding (should be a separate function)
    y_onehot = torch.FloatTensor(10000, 10)
    y_onehot.zero_()
    y_onehot.scatter_(1, bitmaps.unsqueeze(1), 1)

    bias = torch.mean(mean - y_onehot)
    return bias

OUTPUT_DIR = '/media/mattscicluna/Backups/Projects/CourseWork-MILA/IFT6085/inf-paths-results/matt_folder'
corr_list = os.listdir(OUTPUT_DIR)
variances_by_corr_corrupt = []
variances_by_corr_normal = []
biases_by_corr_corrupt = []
biases_by_corr_normal = []

for corr in corr_list:
    dir = os.path.join(OUTPUT_DIR, corr)
    num_seeds = len(os.listdir(dir))
    outputs_normal = torch.zeros(num_seeds, 10000, 10)
    outputs_corrupt = torch.zeros(num_seeds, 10000, 10)
    ind = 0
    for model in os.listdir(dir):
        mod = torch.load(f=os.path.join(dir, model))
        deepish_net = ThreeLayerNetCIFAR10(512)
        deepish_net.load_state_dict(mod['state'])
        seed = 0 #int(model.split('-')[-1].split('.')[0])+int(model.split('-')[3])*11
        data_model_comp = DataModelComp(deepish_net, batch_size=128, test_batch_size=128, epochs=200,
                                        lr=1000, decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                        no_cuda=False, seed=seed, log_interval=1000,
                                        run_i=1, save_interval=None, data='CIFAR10', corruption=float(corr))
        _, _, _, output = data_model_comp.evaluate_test(cur_iter=1)
        outputs_corrupt[ind] = output.exp().detach()
        true_labels = data_model_comp.test_loader.dataset.test_labels

        data_model_comp = DataModelComp(deepish_net, batch_size=128, test_batch_size=128, epochs=200,
                                        lr=1000, decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                        no_cuda=False, seed=seed, log_interval=1000,
                                        run_i=1, save_interval=None, data='CIFAR10', corruption=0)
        _, _, _, output = data_model_comp.evaluate_test(cur_iter=1)
        outputs_normal[ind] = output.exp().detach()
        corrupted_labels = data_model_comp.test_loader.dataset.test_labels
        print(corrupted_labels)
        ind += 1

    mean_corrupt = torch.mean(outputs_corrupt, 0)
    mean_normal = torch.mean(outputs_normal, 0)

    #test = torch.utils.data.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    #test_loader = torch.utils.data.DataLoader(test, batch_size=10000)
    #_, y = next(iter(test_loader))

    biases_by_corr_corrupt.append(calculate_bias(corrupted_labels, mean_corrupt))
    biases_by_corr_normal.append(calculate_bias(true_labels, mean_normal))

    variance = calculate_variance(outputs_corrupt, mean_corrupt)
    variance = torch.Tensor([variance]).unsqueeze(0)
    relative_error = np.sqrt(2 / num_seeds)
    std_variance = variance * relative_error
    variances_by_corr_corrupt.append(torch.cat([variance, std_variance, std_variance]).numpy())
    print('variance mean size:', variance.size())
    print('variance std size:', std_variance.size())

    variance = calculate_variance(outputs_normal, mean_normal)
    variance = torch.Tensor([variance]).unsqueeze(0)
    relative_error = np.sqrt(2 / num_seeds)
    std_variance = variance * relative_error
    variances_by_corr_normal.append(torch.cat([variance, std_variance, std_variance]).numpy())
    print('variance mean size:', variance.size())
    print('variance std size:', std_variance.size())


variances_by_corr_corrupt = np.array(variances_by_corr_corrupt)
variances_by_corr_normal = np.array(variances_by_corr_normal)
np.save('vn', variances_by_corr_normal)
np.save('vc', variances_by_corr_corrupt)
np.save('bc', biases_by_corr_corrupt)
np.save('bn', biases_by_corr_normal)
#plt.plot(variances_by_corr[:, 0, 0])

fig, ax = plt.subplots()
plt.grid(True)
plt.title('Variance per Level of Corruption')

quantities = ["Variance"]
for i, quantity in enumerate(quantities):
    print("QUANTITY:", quantity)
    #data = variances_by_corr_normal

    ax.grid(True)
    #ax.set_xscale('log')

    ax.set_xlabel("Level of Corruption")
    ax.set_ylabel("%s" % (quantity))
    #ax.errorbar([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    #            variances_by_corr[:, 0, 0], yerr=np.array([variances_by_corr[:, 1, 0], variances_by_corr[:, 2, 0]]))
    ax.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            variances_by_corr_normal[:, 0, 0],
            label='normal targets')
    ax.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            variances_by_corr_corrupt[:, 0, 0],
            label='corrupted targets')
    ax.legend()

plt.savefig('variances-per-corruption')

fig, ax = plt.subplots()
plt.grid(True)
plt.title('Bias per Level of Corruption')

quantities = ["Bias"]
for i, quantity in enumerate(quantities):
    print("QUANTITY:", quantity)
    #data = variances_by_corr_normal

    ax.grid(True)
    #ax.set_xscale('log')

    ax.set_xlabel("Level of Corruption")
    ax.set_ylabel("%s" % (quantity))
    #ax.errorbar([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    #            variances_by_corr[:, 0, 0], yerr=np.array([variances_by_corr[:, 1, 0], variances_by_corr[:, 2, 0]]))
    ax.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            biases_by_corr_normal,
            label='normal targets')
    ax.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            biases_by_corr_corrupt,
            label='corrupted targets')
    ax.legend()

plt.savefig('biases-per-corruption')
