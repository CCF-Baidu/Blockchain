#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from resnet import ResNet18
from cifar_cnn_3conv_layer import cifar_cnn_3conv

def build_model(args):
    if args.model == 'smallcnn' and args.dataset == 'mnist':
        net_glob = SmallCNNMnist(args=args)
    elif args.model == 'smallcnn' and args.dataset == 'fmnist':
        net_glob = SmallCNNMnist(args=args)
    elif args.dataset == 'mnist'  and args.model == 'resnet':
        net_glob = CNNCifar10Relu()#ResNet()#CNNCifar(args=args)#cifar_cnn_3conv(input_channels=3, output_channels=10)#ResNet18(args)###CNNCifar10Relu()#
    elif args.dataset == 'cifar':
        net_glob = CNNCifar10Relu()#ResNet18(args)##CNNCifar(args=args)#cifar_cnn_3conv(input_channels=3, output_channels=10)#ResNet18(args)###CNNCifar10Relu()#
    elif args.model == 'loannet' and args.dataset == 'loan':
        net_glob = LoanNet()
    else:
        exit('Error: unrecognized model')

    if args.gpu != -1:
        net_glob = net_glob.cuda()
    return net_glob


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class CNNCifar10Relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class SmallCNNMnist(nn.Module):
    def __init__(self, args):
        super(SmallCNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 4, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4 * 4 * 8, 16)
        self.fc2 = nn.Linear(16, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNFashion_MnistTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = torch.tanh(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = torch.tanh(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = torch.tanh(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return F.log_softmax(x, dim=1)

class LoanNet(nn.Module):
    def __init__(self, in_dim=92, n_hidden_1=46, n_hidden_2=23, out_dim=9):
        super(LoanNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    nn.Dropout(0.5), # drop 50% of the neuron to avoid over-fitting
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.Dropout(0.5),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)

        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size, -1)  # torch.view: 可以改变张量的维度和大小,与Numpy的reshape类似

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, s=1):
        nn.Module.__init__(self)
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=s, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if s != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=s),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, residualBlock=BasicBlock, n_class=10):
        nn.Module.__init__(self)
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pooling = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self.maker_layer(residualBlock, 64, 2, s=1)
        self.layer2 = self.maker_layer(residualBlock, 128, 2, s=2)
        self.layer3 = self.maker_layer(residualBlock, 256, 2, s=2)
        self.layer4 = self.maker_layer(residualBlock, 512, 2, s=2)
        self.fc = nn.Linear(512, n_class)

    def maker_layer(self, block, channels, n_blocks, s):
        strides = [s] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pooling(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return F.log_softmax(out, dim=1)