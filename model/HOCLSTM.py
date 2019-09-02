import torch
from torch import nn as nn
from torch.nn import functional as F

from utils import weights_init, norm_col_init


class HOClstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space,num_options=4):
        super(HOClstm, self).__init__()
        self.module_list = nn.ModuleList()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.module_list += [self.conv1]
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.module_list += [self.maxp1]
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.module_list += [self.conv2]
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.module_list += [self.maxp2]
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.module_list += [self.conv3]
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.module_list += [self.maxp3]
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.module_list += [self.conv4]
        self.maxp4 = nn.MaxPool2d(2, 2)
        self.module_list += [self.maxp4]

        self.lstm = nn.LSTMCell(1024, 512)
        self.module_list += [self.lstm]
        num_outputs = action_space.n
        self.critic_linearO1 = nn.Linear(512, num_options)
        self.module_list += [self.critic_linearO1]

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        self.critic_linearO1.weight.data = norm_col_init(
            self.critic_linearO1.weight.data, 1.0)
        self.critic_linearO1.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        ###new

        self.termlayer1 = {}
        self.option2policy = {}
        self.termlayer2 = {}
        self.actionlayer = {}
        self.h2qo2 = {}

        ### Init of P(o1 | s)
        self.option1policy = nn.Linear(512, num_options)
        self.module_list += [self.option1policy]
        self.option1policy.weight.data = norm_col_init(self.option1policy.weight.data, 0.01)
        self.option1policy.bias.data.fill_(0)


        for i in range(0, num_options):
            self.option2policy[i] = nn.Linear(512, num_options)
            self.module_list += [self.option2policy[i]]
            self.termlayer1[i] = nn.Linear(512, 1)
            self.module_list += [self.termlayer1[i]]

            ###init
            self.option2policy[i].weight.data = norm_col_init(self.option2policy[i].weight.data, 0.01)
            self.option2policy[i].bias.data.fill_(0)
            self.termlayer1[i].weight.data = norm_col_init(self.termlayer1[i].weight.data, 0.01)
            self.termlayer1[i].bias.data.fill_(0)

            self.h2qo2[i] = nn.Linear(512, num_options)
            self.module_list += [self.h2qo2[i]]

            ###init
            self.h2qo2[i].weight.data = norm_col_init(self.h2qo2[i].weight.data, 1.0)
            self.h2qo2[i].bias.data.fill_(0)


            self.termlayer2[i] = {}
            self.actionlayer[i] = {}
            for k in range(0,num_options):
                self.termlayer2[i][k] = nn.Linear(512, 1)
                self.module_list += [self.termlayer2[i][k]]
                self.actionlayer[i][k] = nn.Linear(512,num_outputs)
                self.module_list += [self.actionlayer[i][k]]

                ###init
                self.termlayer2[i][k].weight.data = norm_col_init(self.termlayer2[i][k].weight.data, 0.01)
                self.termlayer2[i][k].bias.data.fill_(0)
                self.actionlayer[i][k].weight.data = norm_col_init(self.actionlayer[i][k].weight.data, 0.01)
                self.actionlayer[i][k].bias.data.fill_(0)


        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linearO1(x), (hx, cx)




    def getTermination1(self,hidden,o):
        term = torch.sigmoid(self.termlayer1[o](hidden))
        return term

    def getTermination2(self,hidden,o1,o2):
        term = torch.sigmoid(self.termlayer2[o1][o2](hidden))
        return term


    def getPolicyO1(self, hidden):
        linearOutput=self.option1policy(hidden)
        ### We do the below for making the gradients
        ### more reliable to large and small numbers
        ### Though doing log and then exp isn't really needed
        logpo1 = F.log_softmax(linearOutput, dim=1)
        probo1 = F.softmax(linearOutput, dim=1)
        o1 = torch.exp(logpo1).multinomial(1).data.numpy()[0]
        return probo1,logpo1,o1[0]



    def getPolicyO2(self,hidden,o):
        linearOutput=self.option2policy[o](hidden)
        ### We do the below for making the gradients
        ### more reliable to large and small numbers
        ### Though doing log and then exp isn't really needed
        logpo2 = F.log_softmax(linearOutput, dim=1)
        probo2 = F.softmax(linearOutput, dim=1)
        o2 = torch.exp(logpo2).multinomial(1).data.numpy()[0]
        return probo2,logpo2,o2[0]

    def getPolicyA(self,hidden,o1,o2):
        linearOutput = self.actionlayer[o1][o2](hidden)
        ### We do the below for making the gradients
        ### more reliable to large and small numbers
        ### Though doing log and then exp isn't really needed
        logpA = F.log_softmax(linearOutput, dim=1)
        probA = F.softmax(linearOutput, dim=1)
        Ac = torch.exp(logpA).multinomial(1).data
        return probA,logpA,Ac

    def getQo2(self,hidden,o):
        qo2 = self.h2qo2[o](hidden)
        return qo2
