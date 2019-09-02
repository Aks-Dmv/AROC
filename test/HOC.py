import logging
import os
import time

import torch
from setproctitle import setproctitle as ptitle

from model.HOCLinear import HOCModel
from environment import *
from agent.HOC import HOCAgent
from utils import setup_logger


def testhoc(args, shared_model, env_conf):
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
    reward_mean = 0
    player = HOCAgent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = HOCModel(player.env.observation_space.shape[0],
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
        EpisodeLength+=1
        reward_sum += player.reward

        if player.done or EpisodeLength>args.num_steps:
            flag = True
            num_tests += 1
            reward_mean = reward_mean + (reward_sum - player.eps_len*reward_mean) / num_tests
            num_frames += player.eps_len
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}, terminations o1 {4}, switches o1 {5}, terminations o2 {6}, switches o2 {7}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean,player.terms1,player.switches1,player.terms2,player.switches2))

            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}.dat'.format(
                            args.save_model_dir, args.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(
                        args.save_model_dir, args.env))

            EpisodeLength=0
            reward_sum = 0
            player.eps_len = 0
            player.terms1 = 0
            player.switches1 = 0
            player.terms2 = 0
            player.switches2 = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
