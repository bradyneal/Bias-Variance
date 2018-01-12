from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_D = 784   # 28 x 28
MNIST_K = 10


class Linear(nn.Module):
    """Linear classifier"""
    
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Linear(MNIST_D, MNIST_K)
        
    def forward(self, x):
        x = x.view(-1, MNIST_D)
        out = self.linear(x)
        return F.log_softmax(out)
    
    
class ShallowNet(nn.Module):
    """Shallow neural network"""
    
    def __init__(self, num_hidden):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(MNIST_D, num_hidden)
        self.fc2 = nn.Linear(num_hidden, MNIST_K)
        
    def forward(self, x):
        x = x.view(-1, MNIST_D)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return F.log_softmax(out)
    

class MinDeepNet(nn.Module):
    """Neural network with 2 layers"""
    
    def __init__(self, num_hidden1, num_hidden2):
        super(MinDeepNet, self).__init__()
        self.fc1 = nn.Linear(MNIST_D, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, MNIST_K)
        
    def forward(self, x):
        x = x.view(-1, MNIST_D)
        h1 = F.relu(self.fc1(x))
        h2= F.relu(self.fc2(h1))
        out = self.fc3(h2)
        return F.log_softmax(out)
    

class ExampleNet(nn.Module):
    """Neural network from the copied PyTorch example"""
    
    def __init__(self):
        super(ExampleNet, self).__init__()
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
