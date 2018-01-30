from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class DataModelComp:
    """
    Class for training neural networks and outputing various measures of the
    information the weights contain.
    """

    def __init__(self, model, batch_size=64, test_batch_size=10000, epochs=10,
                 lr=0.01, decay=False, step_size=10, gamma=0.1, momentum=0.5,
                 no_cuda=False, seed=False, log_interval=100):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.decay = decay
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
        self.model = model
        if self.cuda:
            self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        if decay:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma)

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
            batch_size=test_batch_size, shuffle=False, **kwargs)

    def train_step(self, epoch=1):
        if self.decay:
            self.scheduler.step()
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
 
    def train(self, epochs=None, eval_path=False):
        if eval_path:
            train_seq = []
            test_seq = []
        if epochs is None:
            epochs = self.epochs
        for epoch in range(1, epochs + 1):
            self.train_step(epoch)
            if eval_path:
                 _, _, train_bitmap = self.evaluate_train()
                 _, _, test_bitmap = self.evaluate_test()
                 train_seq.append(train_bitmap)
                 test_seq.append(test_bitmap)
        if eval_path:
            return train_seq, test_seq
             
    def evaluate_train(self):
        """Evaluate the training loss at the current setting of weights"""
        num_train = len(self.train_loader.dataset)
        total_loss = 0
        num_correct = 0
        correct = torch.FloatTensor(0, 1)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = self.model(data)
            
            # Loss
            batch_loss = F.nll_loss(output, target)
            total_loss += batch_loss
            
            # Predictions and accuracy accumulation
            pred = output.data.max(1, keepdim=True)[1]
            batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
            num_correct += batch_correct.sum()
            correct = torch.cat([correct, batch_correct], 0)
        
        acc = num_correct / num_train
        avg_loss = total_loss / num_train
        return acc, avg_loss, correct
               
    def evaluate_test(self):
        self.model.eval()
        num_test = len(self.test_loader.dataset)
        total_loss = 0
        num_correct = 0
        correct = torch.FloatTensor(0, 1)
        for data, target in self.test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            total_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
            correct = torch.cat([correct, batch_correct], 0)
            num_correct += batch_correct.sum()

        avg_loss = total_loss / num_test
        acc = num_correct / num_test
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, num_correct, num_test, 100. * acc))
        
        return acc, avg_loss, correct
