# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from setproctitle import setproctitle as ptitle
from torch import optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

from environment import *
from model.OCPGLinear import OCPGModel
from agent.OCPG import OCPGAgent

from utils import ensure_shared_grads
import gym

def trainocpg(rank, args, shared_model, optimizer, env_conf):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = OC_env(args.env)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = OCPGAgent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = OCPGModel(player.env.observation_space.shape[0],
                           player.env.action_space,args.options,args.width)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2
    threshold = 0
    EnvNumSteps=0
    reward_mean=0.
    while True:
        if EnvNumSteps > threshold:
            threshold += 5000
            print("thread:",rank,"steps:",EnvNumSteps)

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            ### add in option selection part
            q, logito = player.model(Variable(player.state))
            probo = F.softmax(logito, dim=1)
            player.otensor = probo.multinomial(1).data
            player.o = player.otensor.numpy()[0][0]

        else:
            player.o = player.o

        for step in range(args.num_steps):
            EnvNumSteps+=1
            player.action_train()
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        # if not player.done:
        q, logito = player.model(Variable(player.state))
        v = q.max(-1)[0]
        R = v.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = torch.zeros(1,1)
        value_loss = torch.zeros(1,1)
        phi_loss = torch.zeros(1,1)
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        thesize = len(player.rewards)
        reward_sum=sum(player.rewards)
        reward_mean = reward_mean + (reward_sum - thesize*reward_mean)/ EnvNumSteps
        JPi = Variable(torch.tensor(reward_mean))
        for i in reversed(range(len(player.rewards))):
            before = R
            R = args.gamma * R + player.rewards[i] - JPi
            difference = R - player.qs[i]
            if i+1 < thesize:
                difference2 = before - player.values[i+1]

            else:
                NextQ, NextLogito = player.model(Variable(player.state))
                NextTerm = player.model.getTermination(Variable(player.state), player.o)
                NextProbso = F.softmax(NextLogito, dim=1)
                ### select new option
                otensor = NextProbso.multinomial(1).data
                NextLog_probso = F.log_softmax(NextLogito, dim=1)

                NextValue = NextQ.max(-1)[0]
                NextEntropyso = -(NextLog_probso * NextProbso).sum(1)
                NextLog_probso = NextLog_probso.gather(1, Variable(otensor))
                difference2 = before - NextValue


            value_loss = value_loss + 0.5 * difference.pow(2)


            policy_loss = policy_loss - player.log_probs[i]*Variable(difference.data) - 0.1*player.entropies[i]

            if i+1 < thesize:
                beta = player.termprobs[i+1].data

                policy_loss = policy_loss - args.gamma*beta*player.log_probso[i+1]*Variable(difference2.data) - 0.1*player.entropieso[i+1]

                ###!!!!! termination update
                advantage = player.qs[i+1].data-player.values[i+1].data+args.delib
                phi_loss = phi_loss + args.gamma*player.termprobs[i+1]*Variable(advantage, requires_grad=False)

            else:
                beta = NextTerm.data

                policy_loss = policy_loss - args.gamma*beta*NextLog_probso*Variable(difference2.data) - 0.1*NextEntropyso

                ###!!!!! termination update
                advantage = NextQ.data-NextValue.data+args.delib
                phi_loss = phi_loss + args.gamma*NextTerm*Variable(advantage, requires_grad=False)



        player.model.zero_grad()
        (phi_loss.sum() + policy_loss.sum() + 0.5 * value_loss.sum()).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()
        if str(rank)=="1":
            fullname = args.save_model_dir+args.env+str(rank)+".torch"
            tmpname = args.save_model_dir+args.env+str(rank)+".tmp"
            torch.save(optimizer.state_dict(), tmpname) #optimizer.state_dict()
            os.rename(tmpname, fullname)
