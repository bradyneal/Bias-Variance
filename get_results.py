import numpy as np
import torch
from matplotlib import pyplot as plt

fig_e = np.load('results/fig_e_series')
bitmaps = np.load('results/bitmaps')

print('done?')

def calculate_variance(bitmaps, mean):
    return torch.mean((bitmaps - mean.unsqueeze(0)) ** 2)

var_obj = np.zeros(shape=[11])

for i in range(var_obj.shape[0]):
    next_var = ((bitmaps[:, i, :] - np.expand_dims(bitmaps[:, i, :].mean(0), 0)) ** 2).mean()
    var_obj[i] = next_var

plt.figure()
plt.title('Variance Vs Corruption')
plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], var_obj)
plt.xlabel('Corruption Level')
plt.ylabel('Bitmap Variance')
plt.savefig('var_vs_corruption_result')


#bitmaps[:, 0, :] - bitmaps[:, 0, :].mean(0)
#((bitmaps[:, 0, :] - np.expand_dims(bitmaps[:, 0, :].mean(0), 0))**2).mean(0) # variance per individual bitmap