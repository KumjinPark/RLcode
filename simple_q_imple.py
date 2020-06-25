import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from collections import deque
import random
import math
import time


device = torch.device('cpu')
# torch.cuda.set_device(device)


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    n = len(sizes)
    for i in range(n-1):
        if i < n-2:
            act = activation
        else:
            act = output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, e):
        self.buffer.append(e)

    def sample(self, batch_size):
        return random.sample(self.buffer, k=batch_size)

    def __len__(self):
        return len(self.buffer)


steps_done = 0


def train(env_name='CartPole-v1', hidden_sizes=[128, 128], buffer_sizes=50000, lr=0.0005,
          gamma=0.98, eps_start=0.08, eps_end=0.01, eps_decay=200, batch_size=32,
          epochs=50, epis_epoch=30, target_update=10, render=False):

    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous observation space"
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action space"

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    q_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts]).to(device=device)
    q_target = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts]).to(device=device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_sizes)

    def get_action(obs):
        global steps_done
        p = random.random()
        eps = eps_end + (eps_start - eps_end)*math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        if p > eps:
            return q_net(obs).argmax().view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_acts)]], device=device, dtype=torch.long)

    def train_step():
        if len(buffer) < batch_size:
            return
        batch = buffer.sample(batch_size)
        batch_o, batch_a, batch_r, batch_t, done_mask = zip(*batch)
        obs = torch.cat(batch_o)
        act = torch.cat(batch_a)
        rew = torch.cat(batch_r)
        obs_p = torch.cat(batch_t)
        done_mask = torch.cat(done_mask)
        q_hat = q_net(obs).gather(1, act)
        next_act = q_net(obs_p).max(1, keepdim=True)[1]
        q_next = q_target(obs_p).gather(1, next_act).detach()
        target = rew + gamma*q_next*done_mask

        loss = F.smooth_l1_loss(q_hat, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_epi = 0
    for epoch in range(epochs):
        start = time.time()
        obs = env.reset()
        obs = torch.tensor([obs], device=device, dtype=torch.float)
        done = False
        returns = []
        epi_lens = []
        finished_rendering_this_epoch = False

        epi_len = 0
        ret = 0
        while True:

            if (not finished_rendering_this_epoch) and render:
                env.render()

            act = get_action(obs[0])
            next_obs, rew, done, _ = env.step(act.item())
            epi_len += 1
            ret += rew
            rew = torch.tensor([[rew]], device=device, dtype=torch.float)
            next_obs = torch.tensor([next_obs], device=device, dtype=torch.float)
            if done:
                done_mask = torch.tensor([[0.0]], device=device)
            else:
                done_mask = torch.tensor([[1.0]], device=device)
            transition = (obs, act, rew, next_obs, done_mask)
            buffer.push(transition)
            obs = next_obs
            train_step()

            if done:
                epi_lens.append(epi_len)
                total_epi += 1
                if total_epi % target_update == 0:
                    q_target.load_state_dict(q_net.state_dict())
                returns.append(ret)
                obs, done, epi_len, ret = env.reset(), False, 0, 0
                obs = torch.tensor([obs], device=device, dtype=torch.float)
                finished_rendering_this_epoch = True
                if total_epi % epis_epoch == 0:
                    break

        mean_return, mean_len = np.mean(returns), np.mean(epi_lens)
        print('epoch: %3d \t time: %3f \t return: %.3f \t ep_len: %.3f' %
              (epoch+1, time.time()-start, mean_return, mean_len))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()
    print('\nUsing simplest formulation of q learning.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
