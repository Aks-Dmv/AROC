import torch
from setproctitle import setproctitle as ptitle
from torch import optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

from environment import *
from model.HOCLinear import HOCModel
from agent.HOC import HOCAgent
from utils import ensure_shared_grads


def trainhoc(rank, args, shared_model, optimizer, env_conf):
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
    player = HOCAgent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = HOCModel(player.env.observation_space.shape[0],
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
            probo1,logpo1,player.o1 = player.model.getPolicyO1(Variable(player.state))
            probo2,logpo2,player.o2 = player.model.getPolicyO2(Variable(player.state),player.o1)

        else:
            player.o1 = player.o1
            player.o2 = player.o2

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
        if not player.done:
            q = player.model(Variable(player.state))

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
        for i in reversed(range(len(player.rewards))):
            ### update discounted reward
            before = R
            R = args.gamma * R + player.rewards[i]

            ### update value function
            difference1 = R - player.qs1[i]
            value_loss = value_loss + 0.5 * difference1.pow(2)
            difference2 = R - player.qs2[i]
            value_loss = value_loss + 0.5 * difference2.pow(2)
            if i+1 < thesize:
                difference3 = before - player.values[i+1]
                difference4 = before - player.qs1[i+1]

            ### update policy
            # adv1 = R - player.qs1[i]
            delta2 = R - player.qs2[i]


            policy_loss = policy_loss - \
                player.log_probsa[i] * \
                Variable(delta2) - 0.1 * player.entropiesA[i]

            if i+1 < thesize:
                beta1 = player.termprobs1[i+1].data
                beta2 = player.termprobs2[i+1].data

                policy_loss = policy_loss - \
                    args.gamma * player.log_probso1[i+1] * \
                    Variable(beta1 * beta2 * difference3.data) - 0.1 * player.entropieso1[i+1]

                policy_loss = policy_loss - \
                    args.gamma * player.log_probso2[i+1] * \
                    Variable(beta2 * difference4.data) - 0.1 * player.entropieso2[i+1]

                advantage1 = player.qs1[i+1].data - player.values[i+1].data + args.delib
                phi_loss = phi_loss + \
                    args.gamma * player.termprobs1[i+1] * \
                    Variable(advantage1 * beta2, requires_grad=False)

                advantage2 = player.qs2[i+1].data-(1-beta1)*player.qs1[i+1].data-(beta1*player.values[i+1].data)+args.delib
                phi_loss = phi_loss + \
                    args.gamma * player.termprobs2[i+1] * \
                    Variable(advantage2, requires_grad=False)


        player.model.zero_grad()
        (phi_loss.sum() + policy_loss.sum() + 0.5 * value_loss.sum()).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()
