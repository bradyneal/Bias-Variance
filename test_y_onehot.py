'''Returns the one hot vectors for the y label'''

import torch
from torchvision import datasets, transforms

MNIST_TEST_SIZE = 10000
NUM_MNIST_CLASSES = 10


def get_test_y_onehot():
    # Return onehot matrix of test y labels

    test = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test, batch_size=MNIST_TEST_SIZE)
    _, test_y = next(iter(test_loader))

    # get one-hot encoding (should be a separate function)
    test_y_onehot = torch.FloatTensor(MNIST_TEST_SIZE, NUM_MNIST_CLASSES)
    test_y_onehot.zero_()
    test_y_onehot.scatter_(1, test_y.unsqueeze(1), 1)
    test_y_onehot = test_y_onehot.cpu().numpy()
    return test_y_onehot