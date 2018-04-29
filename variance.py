# To run this file,
# from variance import get_variances
# variances = get_variances([5, 25, 100, 1000, 10000])
# Returns

import torch
import os.path

from fileio import get_fine_path_bitmaps_path, load_fine_path_bitmaps

NUM_SEEDS_COMPLETE = 15

def find_last_epoch_when_saving_every_epoch(num_hidden, seed, type):  # TODO: add 80 and 100 as parameters
    for i in range(100):
        if not os.path.isfile(get_fine_path_bitmaps_path(num_hidden, seed, i, type)):
            break

    if i == 0 or i > 80:
        return None
    return i

def calculate_variance(bitmaps, mean):
    return torch.mean((bitmaps - mean.unsqueeze(0)) ** 2)

def get_variances(num_hidden_arr, slurm_id):
    variances = []
    for num_hidden in num_hidden_arr:
        train_val_test_variances = []
        for type in range(3):
            bitmaps = None
            for seed in range(NUM_SEEDS_COMPLETE):
                last_epoch = find_last_epoch(num_hidden, seed, type)
                if last_epoch is None:
                    continue
                bitmap = load_fine_path_bitmaps(num_hidden, seed, last_epoch, type)
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
