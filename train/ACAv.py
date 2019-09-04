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
from model.ACLinear import ACModel
from agent.AC import ACAgent

from utils import ensure_shared_grads
import gym

def trainac(rank, args, shared_model, optimizer, env_conf):
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
    player = ACAgent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = ACModel(player.env.observation_space.shape[0],
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
        q, logit = player.model(Variable(player.state))
        v = q.max(-1)[0]
        R = v.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = torch.zeros(1,1)
        value_loss = torch.zeros(1,1)
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
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * difference.pow(2)


            policy_loss = policy_loss - player.log_probs[i]*Variable(advantage.data) - 0.1*player.entropies[i]



        player.model.zero_grad()
        (policy_loss.sum() + 0.5 * value_loss.sum()).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()
        if str(rank)=="1":
            fullname = args.save_model_dir+args.env+str(rank)+".torch"
            tmpname = args.save_model_dir+args.env+str(rank)+".tmp"
            torch.save(optimizer.state_dict(), tmpname) #optimizer.state_dict()
            os.rename(tmpname, fullname)
