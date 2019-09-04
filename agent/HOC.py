import random

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F


class HOCAgent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.o1 = None
        self.o2 = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.qs1 = []
        self.termprobs1 = []
        self.qs2 = []
        self.termprobs2 = []

        self.log_probsa = []
        self.log_probso2 = []
        self.log_probso1 = []
        self.rewards = []
        self.entropiesA = []
        self.entropieso1 = []
        self.entropieso2 = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.eps = args.eps
        self.num_options = args.options
        self.terms1 = 0
        self.switches1 = 0
        self.terms2 = 0
        self.switches2 = 0

    #
    # def esoft(self,q,eta):
    #     if random.random() >  eta:
    #         ### greedy
    #         return q.data.numpy()[0].argmax()
    #     else:
    #         ### random
    #         #print "rand"
    #         return random.randint(0,self.num_options-1)

    def action_train(self):

        qo1= self.model(Variable(self.state))
        print("in action_train",Variable(self.state),self.o1,Variable(self.o1),self.o2)

        yt2 = self.model.getTermination2(Variable(self.state),self.o1,self.o2)
        term2 = yt2.bernoulli()
        t2 = term2.data[0][0]
        yt1 = self.model.getTermination1(Variable(self.state),self.o1)
        term1 = yt1.bernoulli()
        t1 = term1.data[0][0]
        oldo2 = self.o2
        oldo1 = self.o1

        ### check if option 2 terminates
        if t2==1.0:
            ### check if option 1 terminates
            if t1==1.0:
                ### if yes, select option 1
                probo1,logpo1,self.o1 = self.model.getPolicyO1(Variable(self.state))
                self.terms1 += 1
                if self.o1 != oldo1:
                    self.switches1 += 1
            else:
                probo1,logpo1,o1temp = self.model.getPolicyO1(Variable(self.state))
            ### select option 2 given option 1
            probo2,logpo2,self.o2 = self.model.getPolicyO2(Variable(self.state),self.o1)
            self.terms2 += 1
            if self.o2 != oldo2:
                self.switches2 += 1
        else:
            ### need to still collect logprobs about option 2 even though the policy was not used
            probo2,logpo2,o2temp = self.model.getPolicyO2(Variable(self.state),self.o1)
            probo1,logpo1,o1temp = self.model.getPolicyO1(Variable(self.state))

        ### take an action following the current options
        probA,logpA,action = self.model.getPolicyA(Variable(self.state),self.o1,self.o2)

        entropy = -(logpA * probA).sum(1)
        entropyo1 = -(logpo1 * probo1).sum(1)
        entropyo2 = -(logpo2 * probo2).sum(1)
        self.entropiesA.append(entropy)
        self.entropieso1.append(entropyo1)
        self.entropieso2.append(entropyo2)
        # action = probA.multinomial(1).data
        log_probA = logpA.gather(1, Variable(action))
        log_probo1 = logpo2.gather(1, Variable(torch.from_numpy(np.array([[self.o1]]))))
        log_probo2 = logpo2.gather(1, Variable(torch.from_numpy(np.array([[self.o2]]))))
        qo2 = self.model.getQo2(Variable(self.state),self.o1)
        try:
            state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy())
        except:
            state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy()[0][0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()

        # reward clipping for bounded gradients
        # self.reward = max(min(self.reward, 1), -1)
        value = qo1.max(-1)[0]
        self.qs1.append(qo1[0][self.o1])
        self.qs2.append(qo2[0][self.o2])
        self.values.append(value)
        self.log_probsa.append(log_probA)
        self.log_probso1.append(log_probo1)
        self.log_probso2.append(log_probo2)
        self.rewards.append(self.reward)
        self.termprobs1.append(yt1)
        self.termprobs2.append(yt2)
        return self



    def action_test(self):
        with torch.no_grad():
            if self.done:
                # selecting self.o1, self.o2
                probo1,logpo1,self.o1 = self.model.getPolicyO1(Variable(self.state))
                probo2,logpo2,self.o2 = self.model.getPolicyO2(Variable(self.state),self.o1)
            ### go through step

            ###new to logit
            qo1 = self.model(Variable(self.state))

            yt2 = self.model.getTermination2(Variable(self.state),self.o1,self.o2)
            term2 = yt2.bernoulli()
            t2 = term2.data[0][0]
            yt1 = self.model.getTermination1(Variable(self.state),self.o1)
            term1 = yt1.bernoulli()
            t1 = term1.data[0][0]
            oldo1 = self.o1
            oldo2 = self.o2
            ### check if option 2 terminates
            if t2==1.0:
                ### check if option 1 terminates
                if t1==1.0:
                    ### if yes, select option 1
                    probo1,logpo1,self.o1 = self.model.getPolicyO1(Variable(self.state))
                    self.terms1 += 1
                    if self.o1 != oldo1:
                        self.switches1 += 1
                else:
                    probo1,logpo1,o1temp = self.model.getPolicyO1(Variable(self.state))
                ### select option 2 given option 1
                probo2,logpo2,self.o2 = self.model.getPolicyO2(Variable(self.state),self.o1)
                self.terms2 += 1
                if self.o2 != oldo2:
                    self.switches2 += 1
            else:
                ### need to still collect logprobs about option 2 even though the policy was not used
                probo2,logpo2,o2temp = self.model.getPolicyO2(Variable(self.state),self.o1)
                probo1,logpo1,o1temp = self.model.getPolicyO1(Variable(self.state))

            ### take an action following the current options
            probA,logpA,actTemp = self.model.getPolicyA(Variable(self.state),self.o1,self.o2)


        ### collect basics
        action = probA.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()


        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()

        self.eps_len += 1
        return self

    def clearTermsAndSwitches(self):
        self.terms1 = 0
        self.terms2 = 0
        self.switches1 = 0
        self.switches2 = 0

    def clear_actions(self):
        self.values = []
        self.log_probsa = []
        self.log_probso2 = []
        self.log_probso1 = []
        self.rewards = []
        self.entropiesA = []
        self.entropieso1 = []
        self.entropieso2 = []
        self.qs1 = []
        self.termprobs1 = []
        self.qs2 = []
        self.termprobs2 = []
        return self
