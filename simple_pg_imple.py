import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from copy import deepcopy


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    n = len(sizes)
    for i in range(n-1):
        act = activation if i < n-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]

    return nn.Sequential(*layers)


class simplePG:
    def __init__(self, env, hidden_sizes=[32], activation=nn.Tanh, output_activation=nn.Identity):
        self.obs_dim = env.observation_space.shape[0]
        self.n_acts = env.action_space.n
        sizes = [self.obs_dim] + hidden_sizes + [self.n_acts]
        self.logit_net = mlp(sizes, activation, output_activation)

    def get_policy(self, obs):
        logits = self.logit_net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs):
        pi = self.get_policy(obs)
        return pi.sample().item()

    def compute_loss(self, obs_seq, act_seq, weights):
        log_pi = self.get_policy(obs_seq).log_prob(act_seq)
        loss = -(log_pi*weights).mean()
        return loss


# for training policy
def train_one_epoch(trajectories, agent, optimizer):
    obs_seq = []
    act_seq = []
    weights = []
    lens_episode = []
    returns = []
    n = len(trajectories)

    for episode in range(n):
        total_reward = 0
        trajectory = trajectories[episode]
        T = len(trajectory)
        lens_episode.append(T)

        for t in range(T):
            obs, act, reward = trajectory[t]
            obs_seq.append(obs)
            act_seq.append(act)
            total_reward += reward

        returns.append(total_reward)
        weights += [total_reward] * T

    optimizer.zero_grad()
    loss = agent.compute_loss(torch.as_tensor(obs_seq, dtype=torch.float32),
                              torch.as_tensor(act_seq, dtype=torch.float32),
                              torch.as_tensor(weights, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    return loss, returns, lens_episode


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    env = gym.make(env_name)

    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    agent = simplePG(env, hidden_sizes=hidden_sizes)
    optimizer = Adam(agent.logit_net.parameters(), lr=lr)
    obs = env.reset()
    done = False
    for epoch in range(epochs):
        trajectories = []
        finished_rendering_this_epoch = False
        t = 0
        trajectory = []
        while True:
            t += 1

            if (not finished_rendering_this_epoch) and render:
                env.render()

            act = agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, reward, done, _ = env.step(act)
            trajectory.append((obs.copy(), act, reward))
            obs = next_obs

            if done:
                obs = env.reset()
                done = False
                trajectories.append(deepcopy(trajectory))
                trajectory.clear()
                finished_rendering_this_epoch = True

                if t > batch_size:
                    break

        loss, returns, lens_episode = train_one_epoch(trajectories, agent, optimizer)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (epoch, loss, np.mean(returns), np.mean(lens_episode)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
