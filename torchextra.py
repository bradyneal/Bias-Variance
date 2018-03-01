from __future__ import print_function, division

from torch.utils.data.sampler import Sampler

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):

        return len(self.indices)
