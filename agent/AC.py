# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch
from torch.autograd import Variable
from torch.nn import functional as F


class ACAgent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.eps_len = 0
        self.args = args
        self.values = []
        self.qs = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.eps = args.eps



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
        q, logit = self.model(Variable(self.state))

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)

        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)

        action = prob.multinomial(1).data
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
        try:
            self.qs.append(q[0][action.cpu().numpy()])
        except:
            self.qs.append(q[0][action.cpu().numpy()[0][0]])
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self



    def action_test(self):
        with torch.no_grad():
            ### go through step
            q, logit = self.model(Variable(self.state))
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

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.qs = []
        return self
