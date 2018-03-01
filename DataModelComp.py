from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchextra import SubsetSequentialSampler
from fileio import save_fine_path_train_bitmaps, save_fine_path_test_bitmaps


class DataModelComp:
    """
    Class that is the abstraction of a dataset and model. It holds the methods
    that would require both of these, such as training of the model, evaluation
    of the model on the dataset, etc.
    """

    def __init__(self, model, batch_size=100, test_batch_size=10000, epochs=10,
                 lr=0.01, decay=False, step_size=10, gamma=0.1, momentum=0.5,
                 no_cuda=False, seed=False, log_interval=100,
                 run_i=0, save_interval=None):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.seed = seed
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.num_saved_iters = 1
        self.run_i = run_i

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
        self.train_loader, self.test_loader = self.get_data_loaders()
        
        # Save initial bitmaps
        if self.save_interval is not None:
            train_bitmap, test_bitmap = self.get_train_test_bitmaps()
            save_fine_path_train_bitmaps(train_bitmap, self.model.num_hidden, self.run_i, 0)
            save_fine_path_test_bitmaps(test_bitmap, self.model.num_hidden, self.run_i, 0)
            
    def get_data_loaders(self, same_dist=True, split_random_seed=0):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])
        train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test = datasets.MNIST('./data', train=False, download=True, transform=transform)
        if not same_dist:
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                                       shuffle=False, **kwargs)
            test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size,
                                        shuffle=False, **kwargs)
        else:
            combined = torch.utils.data.ConcatDataset([train, test])
            num_train = len(train)
            n = len(combined)
            indices = list(range(n))
            
            np.random.seed(split_random_seed)
            np.random.shuffle(indices)
            
            train_idx, test_idx = indices[:num_train], indices[num_train:]
            train_sampler = SubsetSequentialSampler(train_idx)
            test_sampler = SubsetSequentialSampler(test_idx)
            
            train_loader = torch.utils.data.DataLoader(combined, batch_size=self.batch_size,
                                                       sampler=train_sampler, **kwargs)
            test_loader = torch.utils.data.DataLoader(combined, batch_size=self.test_batch_size,
                                                      sampler=test_sampler, **kwargs)
        return train_loader, test_loader

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
            if self.save_interval is not None and batch_idx % self.save_interval == 0:
                train_bitmap, test_bitmap = self.get_train_test_bitmaps()
                save_fine_path_train_bitmaps(train_bitmap, self.model.num_hidden,
                                             self.run_i, self.num_saved_iters)
                save_fine_path_test_bitmaps(test_bitmap, self.model.num_hidden,
                                            self.run_i, self.num_saved_iters)
                self.num_saved_iters += 1
                
 
    def train(self, epochs=None, eval_path=False):
        if eval_path:
            _, _, train_bitmap = self.evaluate_train()
            _, _, test_bitmap = self.evaluate_test()
            train_seq = [train_bitmap]
            test_seq = [test_bitmap]
        if epochs is None:
            epochs = self.epochs
        for epoch in range(1, epochs + 1):
            self.train_step(epoch)
            if eval_path:
                 train_bitmap, test_bitmap = self.get_train_test_bitmaps()
                 train_seq.append(train_bitmap)
                 test_seq.append(test_bitmap)
        if eval_path:
            return train_seq, test_seq
  
    def get_train_test_bitmaps(self):
        _, _, train_bitmap = self.evaluate_train()
        _, _, test_bitmap = self.evaluate_test()
        return train_bitmap, test_bitmap
             
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
