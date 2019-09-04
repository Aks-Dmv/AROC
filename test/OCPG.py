# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time

import torch
from setproctitle import setproctitle as ptitle


from model.OCPGLinear import OCPGModel

from environment import *
from agent.OCPG import OCPGAgent
from utils import setup_logger
import gym


def testocpg(args, shared_model, env_conf):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = OC_env(args.env)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    num_frames = 0
    reward_mean =0
    player = OCPGAgent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = OCPGModel(player.env.observation_space.shape[0],
                           player.env.action_space,args.options, args.width)

    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = 0
    EpisodeLength=0
    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

        player.action_test()
        player.env.render()
        EpisodeLength+=1
        reward_sum += player.reward

        if player.done or EpisodeLength>args.num_steps:
            flag = True
            num_tests += 1
            reward_mean = reward_mean + (reward_sum - player.eps_len*reward_mean) / num_tests
            num_frames += player.eps_len
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}, terminations {4}, switches {5}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean,player.terms,player.switches))

            #if args.save_max and reward_sum >= max_score:
            max_score = reward_sum
            datname = '{0}{1}.dat'.format(args.save_model_dir, args.env)
            tmpname = '{0}{1}.tmp'.format(args.save_model_dir, args.env)

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, tmpname)
            else:
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, tmpname)

            os.rename(tmpname, datname)

            EpisodeLength=0
            reward_sum = 0
            player.eps_len = 0
            player.terms = 0
            player.switches = 0

            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
