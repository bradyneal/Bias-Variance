from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

MNIST_D = 784   # 28 x 28
MNIST_K = 10

def get_inter_model(model1, model2, theta):
    inter_model = deepcopy(model1)
    for p1, p2, inter_p in zip(model1.parameters(), model2.parameters(), inter_model.parameters()):
        inter_p.data = theta * p1.data + (1 - theta) * p2.data
    return inter_model

class nn_custom_super(nn.Module):
    
    def get_params(self):
        """Return parameters of the neural network as a vector"""
        return torch.cat([p.data.view(-1) for p in self.parameters()], dim=0)
    

class Linear(nn_custom_super):
    """Linear classifier"""
    
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Linear(MNIST_D, MNIST_K)
        
    def forward(self, x):
        x = x.view(-1, MNIST_D)
        out = self.linear(x)
        return F.log_softmax(out)
    
    
class ShallowNet(nn_custom_super):
    """Shallow neural network"""
    
    def __init__(self, num_hidden):
        super(ShallowNet, self).__init__()
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(MNIST_D, num_hidden)
        self.fc2 = nn.Linear(num_hidden, MNIST_K)
        
    def forward(self, x):
        x = x.view(-1, MNIST_D)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return F.log_softmax(out)
    
    

class MinDeepNet(nn_custom_super):
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
    

class ExampleNet(nn_custom_super):
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


CIFAR10_D = 3*784
CIFAR10_K = 10
class ShallowNetCIFAR10(nn_custom_super):
    """Shallow neural network for CIFAR10"""
    def __init__(self, num_hidden):
        super(ShallowNetCIFAR10, self).__init__()
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(CIFAR10_D, num_hidden)
        self.fc2 = nn.Linear(num_hidden, CIFAR10_K)

    def forward(self, x):
        x = x.view(-1, CIFAR10_D)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return F.log_softmax(out)


class ThreeLayerNetCIFAR10(nn_custom_super):
    """Shallow neural network for CIFAR10"""
    def __init__(self, num_hidden):
        super(ThreeLayerNetCIFAR10, self).__init__()
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(CIFAR10_D, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, CIFAR10_K)

    def forward(self, x):
        x = x.view(-1, CIFAR10_D)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        out = self.fc4(h3)
        return F.log_softmax(out)


class AlexNetCIFAR10(nn_custom_super):
    #  from torchvision: http://pytorch.org/docs/0.3.0/_modules/torchvision/models/alexnet.html
    #  96, 256 are guesses -> and dims don't match up with paper :(
    def __init__(self):
        super(AlexNetCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 3 * 3, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        h1 = F.relu(self.conv1(x), inplace=True)
        h1 = self.pool1(h1)
        h2 = F.relu(self.conv2(h1), inplace=True)
        h2 = self.pool2(h2)
        h2 = h2.view(h2.size(0), 256 * 3 * 3)
        h3 = F.relu(self.fc1(h2), inplace=True)
        h4 = F.relu(self.fc2(h3), inplace=True)
        out = self.fc3(h4)
        return F.log_softmax(out)


class InceptionCIFAR10(nn_custom_super):
    def __init__(self):
        super(InceptionCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(96)

        self.incp1 = InceptionModule(96, 32, 32)
        self.incp2 = InceptionModule(64, 32, 48)
        self.downsample1 = DownSampleModule(80, 80)

        self.incp3 = InceptionModule(160, 112, 48)
        self.incp4 = InceptionModule(160, 96, 64)
        self.incp5 = InceptionModule(160, 80, 80)
        self.incp6 = InceptionModule(160, 48, 96)
        self.downsample2 = DownSampleModule(144, 96)

        self.incp7 = InceptionModule(240, 176, 160)
        self.incp8 = InceptionModule(336, 176, 160)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(336, 10)

    def forward(self, x):
        h1 = self.bn1(self.conv1(x))
        h1 = F.relu(h1, inplace=True)

        h2 = self.incp1.forward(h1)
        h3 = self.incp2.forward(h2)
        h4 = self.downsample1.forward(h3)

        h5 = self.incp3.forward(h4)
        h6 = self.incp4.forward(h5)
        h7 = self.incp5.forward(h6)
        h8 = self.incp6.forward(h7)
        h9 = self.downsample2.forward(h8)

        h10 = self.incp7.forward(h9)
        h11 = self.incp8.forward(h10)
        h12 = self.avgpool(h11)
        out = self.fc(h12.squeeze())
        return F.log_softmax(out)


class InceptionModule(nn.Module):
    #  Inception model from fig 3
    def __init__(self, in_channels, ch1, ch3):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, ch1, kernel_size=1, padding=0)
        self.branch3x3 = nn.Conv2d(in_channels, ch3, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        outputs = [branch1x1, branch3x3]
        return torch.cat(outputs, 1)


class DownSampleModule(nn.Module):
    #  Downsample model from fig 3
    def __init__(self, in_channels, ch3):
        super(DownSampleModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, ch3, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branchconv = self.conv1(x)
        branchpool = self.maxpool(x)
        outputs = [branchconv, branchpool]
        return torch.cat(outputs, 1)
