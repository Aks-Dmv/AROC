# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn as nn
from torch.nn import functional as F

from utils import norm_col_init, weights_init


class OCPGlstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space,num_options,nnWidth):
        super(OCPGlstm, self).__init__()
        self.module_list = nn.ModuleList()
        # self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        # self.module_list += [self.conv1]
        # self.maxp1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        # self.module_list += [self.conv2]
        # self.maxp2 = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        # self.module_list += [self.conv3]
        # self.maxp3 = nn.MaxPool2d(2, 2)
        # self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.module_list += [self.conv4]
        # self.maxp4 = nn.MaxPool2d(2, 2)
        self.lin1 = nn.Linear(num_inputs, nnWidth)
        self.module_list += [self.lin1]
        self.lin2 = nn.Linear(nnWidth, nnWidth)
        self.module_list += [self.lin2]
        self.lin3 = nn.Linear(nnWidth, 2*nnWidth)
        self.module_list += [self.lin3]


        self.lstm = nn.LSTMCell(2*nnWidth, nnWidth)
        self.module_list += [self.lstm]
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(nnWidth, num_options)
        self.module_list += [self.critic_linear]
        self.optionpolicy = nn.Linear(nnWidth, num_options)
        self.module_list += [self.optionpolicy]
        self.optionpolicy.weight.data = norm_col_init(self.optionpolicy.weight.data, 0.01)
        self.optionpolicy.bias.data.fill_(0)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.lin1.weight.data = norm_col_init(
            self.lin1.weight.data, 1.0)
        self.lin1.bias.data.fill_(0)

        self.lin2.weight.data = norm_col_init(
            self.lin2.weight.data, 1.0)
        self.lin2.bias.data.fill_(0)

        self.lin3.weight.data = norm_col_init(
            self.lin3.weight.data, 1.0)
        self.lin3.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.policylayer = {}
        self.termlayer = {}
        for i in range(0, num_options):
            self.policylayer[i] = nn.Linear(nnWidth, num_outputs)
            self.module_list += [self.policylayer[i]]
            self.termlayer[i] = nn.Linear(nnWidth, 1)
            self.module_list += [self.termlayer[i]]
            self.policylayer[i].weight.data = norm_col_init(self.policylayer[i].weight.data, 0.01)
            self.policylayer[i].bias.data.fill_(0)
            self.termlayer[i].weight.data = norm_col_init(self.termlayer[i].weight.data, 0.01)
            self.termlayer[i].bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.lin1(inputs))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = x.view(-1, 2*nnWidth)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.optionpolicy(x), (hx, cx)

    def getTermination(self,hidden,o):
        term = torch.sigmoid(self.termlayer[o](hidden))
        return term


    def getAction(self,hidden,o):
        action = self.policylayer[o](hidden)
        return action
