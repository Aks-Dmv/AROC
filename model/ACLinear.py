# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn as nn
from torch.nn import functional as F

from utils import norm_col_init, weights_init


class ACModel(torch.nn.Module):
    def __init__(self, num_inputs, action_space,num_options,nnWidth):
        super(ACModel, self).__init__()
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
        self.critic_linear = nn.Linear(num_inputs, num_outputs)
        self.module_list += [self.critic_linear]
        self.actionpolicy = nn.Linear(num_inputs, num_outputs)
        self.module_list += [self.actionpolicy]
        self.actionpolicy.weight.data = norm_col_init(self.actionpolicy.weight.data, 0.01)
        self.actionpolicy.bias.data.fill_(0)

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


        self.train()

    def forward(self, inputs):
        # x = F.relu(self.lin1(inputs))
        # x = F.relu(self.lin2(x))
        # x = F.relu(self.lin3(x))
        x = inputs.view(-1, self.numbInputs)

        return self.critic_linear(x), self.actionpolicy(x)
