from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class NNTrainer:
    """
    Class for training neural networks and outputing various measures of the
    information the weights contain.
    """

    def __init__(self, batch_size=64, test_batch_size=1000, epochs=10, lr=0.01, 
               momentum=0.5, no_cuda=False, seed=False, log_interval=100):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.seed = seed
        self.log_interval = log_interval

        if self.cuda:
            print('Using CUDA')
        else:
            print('Using CPU')
        if seed:
            torch.manual_seed(seed)
            if self.cuda:
                torch.cuda.manual_seed(seed)

        # Create network and optimizer
        self.model = Net()
        if self.cuda:
            self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Load data
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    def train_step(self, epoch=1):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data[0]))

    def train(self, epochs=None, test=False):
        if test:
            test_seq = []
        if epochs is None:
            epochs = self.epochs
        for epoch in range(1, epochs + 1):
            self.train_step(epoch)
            if test:
                test_seq.append(self.test())
        if test:
            return test_seq
                    
    def test(self, return_correct=True):
        self.model.eval()
        test_loss = 0
        num_correct = 0
        correct = torch.ByteTensor(0, 1)
        for data, target in self.test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            batch_correct = pred.eq(target.data.view_as(pred))
            correct = torch.cat([correct, batch_correct], 0)
            num_correct += batch_correct.cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, num_correct, len(self.test_loader.dataset),
            100. * num_correct / len(self.test_loader.dataset)))
        if return_correct:
            return correct
        

class Net(nn.Module):
    """Neural network used in NNTrainer"""
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
