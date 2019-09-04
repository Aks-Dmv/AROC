import torch
from torch import nn as nn
from torch.nn import functional as F

from utils import weights_init, norm_col_init


class HOCModel(torch.nn.Module):
    def __init__(self, num_inputs, action_space,num_options,nnWidth):
        super(HOCModel, self).__init__()
        self.numbInputs=num_inputs
        self.module_list = nn.ModuleList()

        try:
            num_outputs = action_space.n
        except AttributeError:
            num_outputs = len(action_space.sample())


        self.critic_linearO1 = nn.Linear(num_inputs, num_options)
        self.module_list += [self.critic_linearO1]

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')

        self.critic_linearO1.weight.data = norm_col_init(
            self.critic_linearO1.weight.data, 1.0)
        self.critic_linearO1.bias.data.fill_(0)


        ###new

        self.termlayer1 = {}
        self.option2policy = {}
        self.termlayer2 = {}
        self.actionlayer = {}
        self.h2qo2 = {}

        ### Init of P(o1 | s)
        self.option1policy = nn.Linear(num_inputs, num_options)
        self.module_list += [self.option1policy]
        self.option1policy.weight.data = norm_col_init(self.option1policy.weight.data, 0.01)
        self.option1policy.bias.data.fill_(0)


        for i in range(0, num_options):
            self.option2policy[i] = nn.Linear(num_inputs, num_options)
            self.module_list += [self.option2policy[i]]
            self.termlayer1[i] = nn.Linear(num_inputs, 1)
            self.module_list += [self.termlayer1[i]]

            ###init
            self.option2policy[i].weight.data = norm_col_init(self.option2policy[i].weight.data, 0.01)
            self.option2policy[i].bias.data.fill_(0)
            self.termlayer1[i].weight.data = norm_col_init(self.termlayer1[i].weight.data, 0.01)
            self.termlayer1[i].bias.data.fill_(0)

            self.h2qo2[i] = nn.Linear(num_inputs, num_options)
            self.module_list += [self.h2qo2[i]]

            ###init
            self.h2qo2[i].weight.data = norm_col_init(self.h2qo2[i].weight.data, 1.0)
            self.h2qo2[i].bias.data.fill_(0)


            self.termlayer2[i] = {}
            self.actionlayer[i] = {}
            for k in range(0,num_options):
                self.termlayer2[i][k] = nn.Linear(num_inputs, 1)
                self.module_list += [self.termlayer2[i][k]]
                self.actionlayer[i][k] = nn.Linear(num_inputs, num_outputs)
                self.module_list += [self.actionlayer[i][k]]

                ###init
                self.termlayer2[i][k].weight.data = norm_col_init(self.termlayer2[i][k].weight.data, 0.01)
                self.termlayer2[i][k].bias.data.fill_(0)
                self.actionlayer[i][k].weight.data = norm_col_init(self.actionlayer[i][k].weight.data, 0.01)
                self.actionlayer[i][k].bias.data.fill_(0)


        self.train()

    def forward(self, inputs):
        x = inputs.view(-1, self.numbInputs)

        return self.critic_linearO1(x)




    def getTermination1(self,TheState,o):
        x = TheState.view(-1, self.numbInputs)
        term = torch.sigmoid(self.termlayer1[o](x))
        return term

    def getTermination2(self,TheState,o1,o2):
        x = TheState.view(-1, self.numbInputs)
        term = torch.sigmoid(self.termlayer2[o1][o2](x))
        return term


    def getPolicyO1(self,TheState):
        x = TheState.view(-1, self.numbInputs)
        linearOutput=self.option1policy(x)
        ### We do the below for making the gradients
        ### more reliable to large and small numbers
        ### Though doing log and then exp isn't really needed
        logpo1 = F.log_softmax(linearOutput, dim=1)
        probo1 = F.softmax(linearOutput, dim=1)
        o1 = torch.exp(logpo1).multinomial(1).data.numpy()[0]
        return probo1,logpo1,o1[0]



    def getPolicyO2(self,TheState,o):
        x = TheState.view(-1, self.numbInputs)
        linearOutput=self.option2policy[o](x)
        ### We do the below for making the gradients
        ### more reliable to large and small numbers
        ### Though doing log and then exp isn't really needed
        logpo2 = F.log_softmax(linearOutput, dim=1)
        probo2 = F.softmax(linearOutput, dim=1)
        o2 = torch.exp(logpo2).multinomial(1).data.numpy()[0]
        return probo2,logpo2,o2[0]

    def getPolicyA(self,TheState,o1,o2):
        x = TheState.view(-1, self.numbInputs)
        linearOutput = self.actionlayer[o1][o2](x)
        ### We do the below for making the gradients
        ### more reliable to large and small numbers
        ### Though doing log and then exp isn't really needed
        logpA = F.log_softmax(linearOutput, dim=1)
        probA = F.softmax(linearOutput, dim=1)
        Ac = torch.exp(logpA).multinomial(1).data
        return probA,logpA,Ac

    def getQo2(self,TheState,o):
        x = TheState.view(-1, self.numbInputs)
        qo2 = self.h2qo2[o](x)
        return qo2
