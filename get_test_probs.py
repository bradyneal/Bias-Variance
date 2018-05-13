import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from DataModelComp import DataModelComp
from models import ThreeLayerNetCIFAR10

def calculate_variance(bitmaps, mean):
    return torch.mean((bitmaps - mean.unsqueeze(0)) ** 2)

OUTPUT_DIR = '/media/mattscicluna/Backups/Projects/CourseWork-MILA/IFT6085/inf-paths-results/matt_folder'
corr_list = os.listdir(OUTPUT_DIR)
variances_by_corr = []

for corr in corr_list:
    dir = os.path.join(OUTPUT_DIR, corr)
    num_seeds = len(os.listdir(dir))
    outputs = torch.zeros(num_seeds, 10000, 10)
    ind = 0
    for model in os.listdir(dir):
        mod = torch.load(f=os.path.join(dir, model))
        deepish_net = ThreeLayerNetCIFAR10(512)
        deepish_net.load_state_dict(mod['state'])
        seed = int(model.split('-')[-1].split('.')[0])+int(model.split('-')[3])*11
        data_model_comp = DataModelComp(deepish_net, batch_size=128, test_batch_size=128, epochs=200,
                                        lr=1000, decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                        no_cuda=False, seed=seed, log_interval=1000,
                                        run_i=1, save_interval=None, data='CIFAR10')
        _, _, _, output = data_model_comp.evaluate_test(cur_iter=1)
        outputs[ind] = output.exp().detach()
        ind += 1

    mean = torch.mean(outputs, 0)
    variance = calculate_variance(outputs, mean)
    variance = torch.Tensor([variance]).unsqueeze(0)
    relative_error = np.sqrt(2 / num_seeds)
    std_variance = variance * relative_error
    variances_by_corr.append(torch.cat([variance, std_variance, std_variance]).numpy())
    print('variance mean size:', variance.size())
    print('variance std size:', std_variance.size())



variances_by_corr = np.array(variances_by_corr)
np.save('to_plot', variances_by_corr)
#plt.plot(variances_by_corr[:, 0, 0])

fig, ax = plt.subplots()
plt.grid(True)
plt.title('Variance per Level of Corruption')

quantities = ["Variance"]
for i, quantity in enumerate(quantities):
    print("QUANTITY:", quantity)
    data = variances_by_corr

    ax.grid(True)
    #ax.set_xscale('log')

    ax.set_xlabel("Level of Corruption")
    ax.set_ylabel("%s" % (quantity))
    ax.errorbar([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                variances_by_corr[:, 0, 0], yerr=np.array([variances_by_corr[:, 1, 0], variances_by_corr[:, 2, 0]]))
plt.savefig('variances-per-model-seed')

