# To get the variances,
# from variance import get_variances
# variances = get_variances([5, 25, 100, 1000, 10000], 20, 159282)
# Returns variances in a 2d array - dimensions are hidden size and ty

import torch
import os.path

from fileio import get_fine_path_bitmaps_path, load_fine_path_bitmaps


def find_last_epoch_when_saving_every_epoch(num_hidden, seed, slurm_id, type, max_epochs=10000):
    for i in range(1, max_epochs):
        if not os.path.isfile(get_fine_path_bitmaps_path(num_hidden, seed, i, slurm_id, type)):
            break

    if i == 0:
        return None
    return i-1


def calculate_variance(bitmaps, mean):
    return torch.mean((bitmaps - mean) ** 2)


def get_variances(num_hidden_arr, num_seeds, slurm_id):
    variances = []
    for num_hidden in num_hidden_arr:
        train_val_test_variances = []
        for type in range(3):
            bitmaps = None
            for seed in range(num_seeds):
                last_epoch = find_last_epoch_when_saving_every_epoch(num_hidden, seed, slurm_id, type)
                if last_epoch is None:
                    continue
                bitmap = load_fine_path_bitmaps(num_hidden, seed, last_epoch, slurm_id, type)
                if bitmaps is None:
                    bitmaps = bitmap
                else:
                    print(bitmaps.shape, bitmap.shape)
                    torch.cat((bitmaps, bitmap), 1)

            if bitmaps is None:
                print('Bitmaps is none for num_hidden=%d, seed=%d, type=%d' % (num_hidden, seed, type))
                continue
            mean = torch.mean(bitmaps, 1)
            variance = calculate_variance(bitmaps, mean)
            train_val_test_variances.append(variance)

        variances.append(train_val_test_variances)

    return variances
