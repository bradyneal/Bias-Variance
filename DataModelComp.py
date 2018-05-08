from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from collections import deque
from torchextra import SubsetSequentialSampler
import matplotlib.pyplot as plt

from fileio import save_fine_path_bitmaps, save_shallow_net, load_shallow_net, save_data_model_comp
from models import ShallowNet


class DataModelComp:
    """
    Class that is the abstraction of a dataset and model. It holds the methods
    that would require both of these, such as training of the model, evaluation
    of the model on the dataset, etc.
    """

    def __init__(self, model, batch_size=100, test_batch_size=100, epochs=10,
                 lr=0.1, decay=False, step_size=10, gamma=0.1, momentum=0.9,
                 no_cuda=False, seed=False, log_interval=100, run_i=0,
                 num_train_after_split=None, save_interval=None,
                 train_val_split_seed=0, bootstrap=False, save_obj=False,
                 print_all_errors=False, print_only_train_and_val_errors=False,
                 size_of_one_pass=None, save_model="all"):
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
        self.num_train_after_split = num_train_after_split
        self.train_val_split_seed = train_val_split_seed
        self.bootstrap = bootstrap
        self.save_model = save_model
        self.print_all_errors = print_all_errors
        self.print_only_train_and_val_errors = print_only_train_and_val_errors
        self.accuracies = [[], [], []]
        self.save_obj = save_obj
        self.size_of_one_pass = size_of_one_pass

        if self.cuda:
            print('Using CUDA')
        else:
            print('Using CPU')
        if seed:
            torch.manual_seed(seed)
            if self.cuda:
                torch.cuda.manual_seed(seed)

        # Handle when we want to make multiple passes for a batch size to use
        # less memory
        if not self.size_of_one_pass:
            self.size_of_one_pass = self.batch_size
        if self.batch_size % self.size_of_one_pass != 0:
            raise Exception('batch_size should e a mutliple of size_of_one_pass')
        self.num_passes = self.batch_size // self.size_of_one_pass
        self.lr /= self.num_passes

        # Create network and optimizer
        self.model = model
        if self.cuda:
            self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                   momentum=momentum)
        if decay:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma)

        # Load data
        self.train_loader, self.val_loader, self.test_loader = self.get_data_loaders()

        # Save initial bitmaps
        if self.save_interval is not None:
            bitmaps = self.get_bitmaps(0)
            for i, bitmap in enumerate(bitmaps):
                save_fine_path_bitmaps(bitmap, self.model.num_hidden,
                                       self.run_i, 0, i)

        if self.save_obj:
            save_data_model_comp(self)

    def get_data_loaders(self, same_dist=False):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])
        train = datasets.MNIST('./data', train=True, download=True,
                               transform=transform)
        test = datasets.MNIST('./data', train=False, download=True,
                              transform=transform)

        np.random.seed(self.train_val_split_seed)

        num_train_before_split = len(train)

        self.num_train_before_split = num_train_before_split

        if self.num_train_after_split is None:
            self.num_val = self.num_train_before_split // 5
            self.num_train_after_split = num_train_before_split - self.num_val

        else:
            self.num_val = self.num_train_after_split

        if self.num_val + self.num_train_after_split > num_train_before_split:
            print("k must be less than %d (number of training examples = %d"
                  " - number of validation examples = %d)" %
                  (num_train_before_split-self.num_val, num_train_before_split, self.num_val))
            raise Exception

        train_and_val_idxs = np.random.choice(num_train_before_split, self.num_val+self.num_train_after_split,
                                              replace=False)
        train_idxs = train_and_val_idxs[:self.num_train_after_split]
        if self.bootstrap:
            train_idxs = np.random.choice(train_idxs, self.num_train_after_split, replace=True)
        val_idxs = train_and_val_idxs[self.num_train_after_split:]

        train_sampler = SubsetSequentialSampler(train_idxs)
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size,
                                                   sampler=train_sampler, **kwargs)

        val_sampler = SubsetSequentialSampler(val_idxs)
        val_loader = torch.utils.data.DataLoader(train, batch_size=self.test_batch_size,
                                                 sampler=val_sampler, **kwargs)

        test_loader = torch.utils.data.DataLoader(test, batch_size=self.test_batch_size,
                                                  shuffle=False, **kwargs)

        # TODO: fix this
        # else:
        #     combined = torch.utils.data.ConcatDataset([train, test])
        #     num_train = len(train)
        #     n = len(combined)
        #     indices = list(range(n))
        #
        #     np.random.seed(split_random_seed)
        #     np.random.shuffle(indices)
        #
        #     train_idx, test_idx = indices[:num_train], indices[num_train:]
        #     train_sampler = SubsetSequentialSampler(train_idx)jjk
        #     test_sampler = SubsetSequentialSampler(test_idx)
        #
        #     train_loader = torch.utils.data.DataLoader(combined, batch_size=self.batch_size,
        #                                                sampler=train_sampler, **kwargs)
        #     test_loader = torch.utils.data.DataLoader(combined, batch_size=self.test_batch_size,
        #                                               sampler=test_sampler, **kwargs)
        return train_loader, val_loader, test_loader

    def train_step(self, epoch=1):
        if self.decay:
            self.scheduler.step()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            self.optimizer.zero_grad()
            for i in range(0, self.num_passes):
                data_partial = data[i*self.size_of_one_pass: (i+1)*self.size_of_one_pass]
                target_partial = target[i*self.size_of_one_pass: (i+1)*self.size_of_one_pass]
                output = self.model(data_partial)
                loss = F.nll_loss(output, target_partial)
                loss.backward()
            self.optimizer.step()

            if self.log_interval is not None and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data[0]))
            if self.save_interval is not None and batch_idx % self.save_interval == 0:
                bitmaps = self.get_bitmaps(self.save_interval * self.num_saved_iters)
                for i, bitmap in enumerate(bitmaps):
                    save_fine_path_bitmaps(bitmap, self.model.num_hidden,
                                           self.run_i, self.num_saved_iters, i)
                self.num_saved_iters += 1

    # def plot_training_curves(self):
    #     # print val list, best, and last
    #     x = list(range(self.epochs))
    #     plt.plot(x, self.accuracies[0])
    #     plt.plot(x, self.accuracies[1])
    #     plt.plot(x, self.accuracies[2])
    #     plt.title('Learning curves')
    #     plt.xlabel('epochs')
    #     plt.ylabel('error')
    #     plt.legend(handles=['train', 'val', 'test'], loc='upper right')
    #     plt.show()
    #     plt.savefig('train_curves.jpg')
        
    def print_validation_accs(self):
        print('Validation list:', self.accuracies[1])
        print('Best and last validation: {}, {}'.format(max(self.accuracies[1]), self.accuracies[1][-1]))

    def train(self, epochs=None, eval_path=False):
        print("Learning rate: {}, momentum: {}, number of training examples: {}, epochs: {}"
              .format(self.lr, self.momentum, self.num_train_after_split, epochs if epochs else self.epochs))
        if eval_path:
            train_bitmap = self.evaluate(0, type=0)[2]
            test_bitmap = self.evaluate(0, type=2)[2]
            train_seq = [train_bitmap]
            test_seq = [test_bitmap]
        if epochs is None:
            epochs = self.epochs

        best_val_accuracy = 0
        reached_zero_training_error = False

        for epoch in range(1, epochs + 1):
            self.train_step(epoch)

            if self.print_only_train_and_val_errors or self.print_all_errors:
                for i in range(2):
                    self.accuracies[i].append(self.evaluate(epoch, type=i)[0])

                # Saves best model so far
                if self.accuracies[1][-1] > best_val_accuracy:
                    best_val_accuracy = self.accuracies[1][-1]
                    save_shallow_net(self.model, self.model.num_hidden, self.run_i, inter=-2)

            if self.print_all_errors:
                self.accuracies[2].append(self.evaluate(epoch, type=2)[0])

            if self.print_all_errors or self.print_only_train_and_val_errors:
                if self.accuracies[0][-1] == 1 and not reached_zero_training_error:
                    reached_zero_training_error = True
                    save_shallow_net(self.model, self.model.num_hidden, self.run_i, inter=-1)

            if self.save_model == "every_epoch":
                save_shallow_net(self.model, self.model.num_hidden, self.run_i, inter=epoch)

        if self.save_model == "only_end":
            save_shallow_net(self.model, self.model.num_hidden, self.run_i)

        if eval_path:
            return train_seq, test_seq
        val_acc = self.evaluate(epoch, type=1)[0]

        if isinstance(self.model, ShallowNet):
            print('Training complete!! For hidden size = {}'.format(self.model.num_hidden))
        else:
            print('Training complete!!')
            
        self.print_validation_accs()
        # self.plot_training_curves()

        return val_acc, self.num_train_after_split * epoch

        # Return no of iterations - epoch * k / batch_size

    def load_saved_shallow_net(self, num_hidden, run_i, slurm_id, inter=0):
        r"""To be used instead of train when loading a trained model"""
        try:
            self.model = load_shallow_net(num_hidden, run_i, slurm_id, inter)
        except OSError as e:
            if isinstance(e, FileNotFoundError):
                print("Using inter 0 instead for num_hidden:", num_hidden)
                self.model = load_shallow_net(num_hidden, run_i, slurm_id, 0)
            else:
                raise e

    # Returns bitmaps on training, validation and test data
    def get_bitmaps(self, cur_epochs):
        return [self.evaluate(cur_epochs, type=i)[2] for i in range(3)]

    def evaluate(self, cur_epochs, type, probs_required=False):
        self.model.eval()
        total_loss = 0
        num_correct = 0
        correct = torch.FloatTensor(0, 1)
        probs = None

        num_to_evaluate_on = [self.num_train_after_split, self.num_val, len(self.test_loader.dataset)][type]
        data_loader = [self.train_loader, self.val_loader, self.test_loader][type]

        for data, target in data_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            prob = self.model(data)
            total_loss += F.nll_loss(prob, target, size_average=False).data[0]  # sum up batch loss
            pred = prob.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
            correct = torch.cat([correct, batch_correct], 0)
            if probs_required:
                if probs is None:
                    probs = prob.data
                else:
                    probs = torch.cat([probs, prob.data], 0)
            num_correct += batch_correct.sum()

        avg_loss = total_loss / num_to_evaluate_on
        acc = num_correct / num_to_evaluate_on
        print('\nAfter {} epochs ({} iterations), {} set: Average loss: {:.4f},Accuracy: {}/{} ({:.2f}%)\n'.format(cur_epochs,
              self.num_train_after_split * cur_epochs, ['Training', 'Validation', 'Test'][type],
              avg_loss, num_correct, num_to_evaluate_on, 100. * acc))

        if probs_required:
            return acc, avg_loss, correct, probs
        else:
            return acc, avg_loss, correct
