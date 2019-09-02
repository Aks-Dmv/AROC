# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch
from torch.autograd import Variable
from torch.nn import functional as F


class OCPGAgent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.o = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.qs = []
        self.termprobs = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.log_probso = []
        self.entropieso = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.eps = args.eps
        self.num_options = args.options
        self.terms = 0
        self.switches = 0


    def esoft(self,q,eta):
        if random.random() >  eta:
            ### greedy
            return q.data.numpy()[0].argmax()
        else:
            ### random
            #print "rand"
            return random.randint(0,self.num_options-1)

    def action_train(self):
        ###new
        q, logito = self.model(Variable(self.state))
        probo = F.softmax(logito, dim=1)
        log_probo = F.log_softmax(logito, dim=1)

        yt = self.model.getTermination(Variable(self.state),self.o)
        term = yt.bernoulli()
        t = term.data[0][0]
        oldo = self.o
        if t==1.0:
            ### select new option
            self.otensor = probo.multinomial(1).data
            self.o = self.otensor.numpy()[0][0]
            self.terms += 1
            if self.o != oldo:
                self.switches += 1


        logit = self.model.getAction(Variable(self.state),self.o)

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)

        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        entropyo = -(log_probo * probo).sum(1)
        self.entropieso.append(entropyo)
        action = prob.multinomial(1).data
        log_probo = log_probo.gather(1, Variable(self.otensor))
        log_prob = log_prob.gather(1, Variable(action))
        try:
            state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy())
        except:
            state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy()[0][0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        # self.reward = max(min(self.reward, 1), -1)
        value = q.max(-1)[0]
        self.qs.append(q[0][self.o])
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.log_probso.append(log_probo)
        self.rewards.append(self.reward)
        self.termprobs.append(yt)
        return self



    def action_test(self):
        with torch.no_grad():
            if self.done:
                # setting self.o
                q, logito = self.model(Variable(self.state))
                probo = F.softmax(logito, dim=1)
                self.o = probo.multinomial(1).data.numpy()[0][0]
            ### go through step
            q, logito = self.model(Variable(self.state))

            yt = self.model.getTermination(Variable(self.state),self.o)
            term = yt.bernoulli()
            t = term.data[0][0]
            oldo = self.o
            if t==1.0:
                ### select new option
                probo = F.softmax(logito, dim=1)
                self.otensor = probo.multinomial(1).data
                self.o = self.otensor.numpy()[0][0]
                self.terms += 1
                if self.o != oldo:
                    self.switches += 1

            logit = self.model.getAction(Variable(self.state),self.o)

        ### collect basics
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()


        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()

        self.eps_len += 1
        return self

    def clearTermsAndSwitches(self):
        self.terms = 0
        self.switches = 0

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.log_probso = []
        self.rewards = []
        self.entropies = []
        self.entropieso = []
        self.qs = []
        self.termprobs = []

        return self
