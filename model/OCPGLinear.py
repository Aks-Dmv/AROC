# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn as nn
from torch.nn import functional as F

from utils import norm_col_init, weights_init


class OCPGModel(torch.nn.Module):
    def __init__(self, num_inputs, action_space,num_options,nnWidth):
        super(OCPGModel, self).__init__()
        self.numbInputs=num_inputs
        self.module_list = nn.ModuleList()
        # self.lin1 = nn.Linear(num_inputs, nnWidth)
        # self.module_list += [self.lin1]
        # self.lin2 = nn.Linear(nnWidth, nnWidth)
        # self.module_list += [self.lin2]
        # self.lin3 = nn.Linear(nnWidth, nnWidth)
        # self.module_list += [self.lin3]
        try:
            num_outputs = action_space.n
        except AttributeError:
            num_outputs = len(action_space.sample())
        self.critic_linear = nn.Linear(num_inputs, num_options)
        self.module_list += [self.critic_linear]
        self.optionpolicy = nn.Linear(num_inputs, num_options)
        self.module_list += [self.optionpolicy]
        self.optionpolicy.weight.data = norm_col_init(self.optionpolicy.weight.data, 0.01)
        self.optionpolicy.bias.data.fill_(0)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        # self.lin1.weight.data = norm_col_init(
        #     self.lin1.weight.data, 1.0)
        # self.lin1.bias.data.fill_(0)

        # self.lin2.weight.data = norm_col_init(
        #     self.lin2.weight.data, 1.0)
        # self.lin2.bias.data.fill_(0)
        #
        # self.lin3.weight.data = norm_col_init(
        #     self.lin3.weight.data, 1.0)
        # self.lin3.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.policylayer = {}
        self.termlayer = {}
        for i in range(0, num_options):
            self.policylayer[i] = nn.Linear(num_inputs, num_outputs)
            self.module_list += [self.policylayer[i]]
            self.termlayer[i] = nn.Linear(num_inputs, 1)
            self.module_list += [self.termlayer[i]]
            self.policylayer[i].weight.data = norm_col_init(self.policylayer[i].weight.data, 0.01)
            self.policylayer[i].bias.data.fill_(0)
            self.termlayer[i].weight.data = norm_col_init(self.termlayer[i].weight.data, 0.01)
            self.termlayer[i].bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        # x = F.relu(self.lin1(inputs))
        # x = F.relu(self.lin2(x))
        # x = F.relu(self.lin3(x))
        x = inputs.view(-1, self.numbInputs)

        return self.critic_linear(x), self.optionpolicy(x)

    def getTermination(self,TheState,o):
        x = TheState.view(-1, self.numbInputs)
        term = torch.sigmoid(self.termlayer[o](x))
        return term


    def getAction(self,TheState,o):
        x = TheState.view(-1, self.numbInputs)
        action = self.policylayer[o](x)
        return action
