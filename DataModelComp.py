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
from fileio import save_fine_path_bitmaps

class DataModelComp:
    """
    Class that is the abstraction of a dataset and model. It holds the methods
    that would require both of these, such as training of the model, evaluation
    of the model on the dataset, etc.
    """

    def __init__(self, model, batch_size=100, test_batch_size=10000, epochs=10,
                 lr=0.1, decay=False, step_size=10, gamma=0.1, momentum=0.9,
                 no_cuda=False, seed=False, log_interval=100, run_i=0,
                 num_train_after_split=None, save_interval=None, save_every_epoch=False,
                 train_val_split_seed=0, bootstrap=False, data='MNIST', corruption=0, to_exclude=0):
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
        self.save_every_epoch = save_every_epoch
        self.data = data
        self.corruption = corruption
        self.to_exclude = to_exclude

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
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                   momentum=momentum)
        if decay:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma)

        # Load data
        self.train_loader, self.val_loader, self.test_loader = \
            self.get_data_loaders(data=self.data)

        # Save initial bitmaps
        if self.save_interval is not None:
            bitmaps = self.get_bitmaps(0)
            for i, bitmap in enumerate(bitmaps):
                save_fine_path_bitmaps(bitmap, self.model.num_hidden,
                                       self.run_i, 0, i)

    def get_data_loaders(self, same_dist=False, data='MNIST'):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        if data == 'MNIST':
            transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])
            train = datasets.MNIST('./data', train=True, download=True,
                                   transform=transform)
            test = datasets.MNIST('./data', train=False, download=True,
                                  transform=transform)
        if data == 'CIFAR10':
            transform = transforms.Compose(
                #  See appendix A of Zhang et al for tranformations used -- these are the same
                [transforms.CenterCrop((28, 28)),  # crop input to 28x28
                 transforms.ToTensor(),  # divide by 255
                 transforms.Lambda(lambda x: (x - x.mean()) / x.std())
                 # per_image_whitening function in TENSORFLOW (Abadi et al., 2015)
                 ])

            train = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
            test = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

        np.random.seed(self.train_val_split_seed)

        num_train = len(train)
        num_val = len(train) // 5

        # TODO: fix num_train and num_val
        self.num_train = num_train
        self.num_val = num_val

        if self.num_train_after_split is None:
            self.num_train_after_split = num_train - num_val

        if num_val + self.num_train_after_split > num_train:
            print("k must be less than %d (number of training examples = %d"
                  " - number of validation examples = %d)" %
                  (num_train-num_val, num_train, num_val))
            raise Exception

        train_and_val_idxs = np.random.choice(num_train, num_val+self.num_train_after_split,
                                              replace=False)
        train_idxs = train_and_val_idxs[:self.num_train_after_split]
        #  further reduce by factor of to_exclude (does same if bootstrapping)
        train_idxs = train_idxs[int(self.to_exclude * self.num_train_after_split):]
        if self.bootstrap:
            train_idxs = np.random.choice(train_idxs, self.num_train_after_split, replace=True)
            train_idxs = train_idxs[int(self.to_exclude * self.num_train_after_split):]
        val_idxs = train_and_val_idxs[self.num_train_after_split:]

        train_sampler = SubsetSequentialSampler(train_idxs)
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size,
                                                   sampler=train_sampler, **kwargs)
        # add randomization to labels
        train_loader.dataset.train_labels = [train_loader.dataset.train_labels[k]
                                             if np.random.uniform() > self.corruption
                                             else np.random.randint(10)
                                             for k in range(num_train)]

        val_sampler = SubsetSequentialSampler(val_idxs)
        val_loader = torch.utils.data.DataLoader(train, batch_size=self.test_batch_size,
                                                 sampler=val_sampler, **kwargs)

        test_loader = torch.utils.data.DataLoader(test, batch_size=self.test_batch_size,
                                                  shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader

    def train_step(self, epoch=1):
        if self.decay:
            self.scheduler.step()
            #print(self.scheduler.get_lr())

        self.model.train()
        steps = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)

            #print('norm of weights: {}'.format(self.model.get_weight_norm()))

            loss = F.nll_loss(output, target)
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
            steps += 1
        return steps

    def train(self, epochs=None, eval_path=False, early_stopping=True, train_to_overfit=False, eval_train_every=False):
        steps = 0
        train_loss_to_return = []
        print("Learning rate: {}, momentum: {}, number of training examples: {}"
              .format(self.lr, self.momentum, self.num_train_after_split))
        if eval_path:
            _, _, train_bitmap = self.evaluate_train(0)
            _, _, test_bitmap = self.evaluate_test(0)
            train_seq = [train_bitmap]
            test_seq = [test_bitmap]
        if epochs is None:
            epochs = self.epochs * self.num_train // self.num_train_after_split
        epochs_per_val = (self.num_train - self.num_val) // self.num_train_after_split  # Number of epochs adjusted for the training size TODO: adjust for the batch size
        last_val_accs = deque(maxlen=10)
        for epoch in range(1, epochs + 1):
            step = self.train_step(epoch)
            steps += step

            if eval_train_every:
                acc, avg_loss, correct = self.evaluate_train(cur_iter=steps)
                train_loss_to_return.append(avg_loss)

            if early_stopping:
                # Implements early stoppping and logs accuracies on train and val sets (early stopping done when
                # validation accuracy has not improved in 10 consecutive epochs)
                if epoch % epochs_per_val == 0:
                    val_acc, _, _ = self.evaluate_val(self.num_train_after_split * epoch)
                    if len(last_val_accs) == 10 and val_acc < last_val_accs[0]:  # TODO: make 10 a parameter
                        break
                    last_val_accs.append(val_acc)
                    self.evaluate_train(self.num_train_after_split * epoch)

            if train_to_overfit:
                # check if likely to be over by evaluating first batch only
                for data, target in self.train_loader:
                    if self.cuda:
                        data, target = data.cuda(), target.cuda()
                    data, target = Variable(data, volatile=True), Variable(target)
                    output = self.model(data)
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
                    num_correct = batch_correct.sum()
                    break

                if num_correct/data.shape[0] >= train_to_overfit:
                    _, _, train_bitmap = self.evaluate_train(epoch)
                    print('current train accuracy: {}'.format(train_bitmap.mean()))
                    if train_bitmap.mean() >= train_to_overfit:
                        break
                else:
                    print('current train accuracy (roughly): {}'.format(num_correct/data.shape[0]))

            if eval_path:
                train_bitmap, _, test_bitmap = self.get_bitmaps(self.num_train_after_split * epoch)
                train_seq.append(train_bitmap)
                test_seq.append(test_bitmap)
                                           
        if eval_path:
            return train_seq, test_seq

        if self.save_every_epoch:
            bitmaps = self.get_bitmaps(self.num_train_after_split * epoch)
            for i, bitmap in enumerate(bitmaps):
                save_fine_path_bitmaps(bitmap, self.model.num_hidden,
                                       self.run_i, epoch, i)

        val_acc, _, _ = self.evaluate_val(self.num_train_after_split * epoch)
        print("Training complete!!")
        if eval_train_every:
            return val_acc, self.num_train_after_split * epoch, steps, train_loss_to_return
        else:
            return val_acc, self.num_train_after_split * epoch, steps

        # Return no of iterations - epoch * k / batch_size

    def get_bitmaps(self, cur_iter):
        _, _, train_bitmap = self.evaluate_train(cur_iter)
        _, _, val_bitmap = self.evaluate_val(cur_iter)
        _, _, test_bitmap = self.evaluate_test(cur_iter)
        return train_bitmap, val_bitmap, test_bitmap

    # TODO: combine next 3 functions into 1
    def evaluate_train(self, cur_iter):
        self.model.eval()
        num_train = self.num_train_after_split
        total_loss = 0
        num_correct = 0
        correct = torch.FloatTensor(0, 1)
        for data, target in self.train_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            total_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
            correct = torch.cat([correct, batch_correct], 0)
            num_correct += batch_correct.sum()

        avg_loss = total_loss / num_train
        acc = num_correct / num_train
        print('After {} iterations, Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
              .format(cur_iter, avg_loss, num_correct, num_train, 100. * acc))

        return acc, avg_loss, correct

    def evaluate_val(self, cur_iter):
        self.model.eval()
        num_val = self.num_val
        total_loss = 0
        num_correct = 0
        correct = torch.FloatTensor(0, 1)
        for data, target in self.val_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            total_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
            correct = torch.cat([correct, batch_correct], 0)
            num_correct += batch_correct.sum()

        avg_loss = total_loss / num_val
        acc = num_correct / num_val
        print('After {} iterations, Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            cur_iter, avg_loss, num_correct, num_val, 100. * acc))
        #print(correct)

        return acc, avg_loss, correct

    def evaluate_test(self, cur_iter):
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
            total_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
            correct = torch.cat([correct, batch_correct], 0)
            num_correct += batch_correct.sum()

        avg_loss = total_loss / num_test
        acc = num_correct / num_test
        print('After {} iterations, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            cur_iter, avg_loss, num_correct, num_test, 100. * acc))

        return acc, avg_loss, correct
