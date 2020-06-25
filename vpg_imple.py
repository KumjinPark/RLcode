import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import gym
from gym.spaces import Box, Discrete
import time

device = torch.device('cuda')
def combined_shape(length, dims=None):
    if dims is None:
        return (length, )
    elif np.isscalar(dims):
        return (length, dims)
    else:
        return (length, *dims)


class VPGBuffer:
    def __init__(self, obs_dim, n_acts, n_steps, gamma=0.99):
        self.obs_buf = np.zeros(combined_shape(n_steps, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(n_steps, n_acts), dtype=np.float32)
        self.rew_buf = np.zeros(n_steps, dtype=np.float32)
        self.adv_buf = np.zeros(n_steps, dtype=np.float32)
        self.ret_buf = np.zeros(n_steps, dtype=np.float32)
        self.val_buf = np.zeros(n_steps, dtype=np.float32)
        self.logp_buf = np.zeros(n_steps, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, n_steps
    
    def store(self, obs, act, rew, val, logp):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        self.adv_buf[path_slice] = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        gammas = np.ones(len(rews))
        gammas[1:-1].fill(self.gamma)
        gammas = np.cumprod(gammas)
        self.ret_buf[path_slice] = np.flipud(np.cumsum(np.flipud(rews*gammas)))[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return torch.as_tensor(self.obs_buf, dtype=torch.float32), \
            torch.as_tensor(self.act_buf, dtype=torch.float32), \
            torch.as_tensor(self.ret_buf, dtype=torch.float32), \
            torch.as_tensor(self.val_buf, dtype=torch.float32), \
            torch.as_tensor(self.adv_buf, dtype=torch.float32), \
            torch.as_tensor(self.logp_buf, dtype=torch.float32)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    N = len(sizes)
    for i in range(N-1):
        if i < N-2:
            act = activation
        else:
            act = output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def _distribution(self, x):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, x, act=None):
        pi = self._distribution(x)
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
            return pi, logp_a
        else:
            return pi


class GaussianActor(Actor):
    def __init__(self, f_net, h_dim, sizes, n_acts, activation=nn.ReLU):
        super(GaussianActor, self).__init__()
        self.f_net = f_net
        self.fc_mu = mlp(sizes=[h_dim]+sizes+[n_acts], activation=activation)
        log_std = -0.5 * np.ones(n_acts, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std, device=device))

    def _distribution(self, x):
        x = F.relu(self.f_net(x))
        mu = self.fc_mu(x)
        std = torch.exp(self.log_std)

        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class CategoricalActor(Actor):
    def __init__(self, f_net, h_dim, sizes, n_acts, activation=nn.ReLU):
        super(CategoricalActor, self).__init__()
        self.f_net = f_net
        self.fc_logits = mlp(sizes=[h_dim]+sizes+[n_acts], activation=activation)

    def _distribution(self, x):
        x = F.relu(self.f_net(x))
        logits = self.fc_logits(x)

        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class Critic(nn.Module):
    def __init__(self, f_net, h_dim, sizes, activation):
        super(Critic, self).__init__()
        self.f_net = f_net
        self.fc_v = mlp(sizes=[h_dim]+sizes+[1], activation=activation)
    
    def forward(self, x):
        x = F.relu(self.f_net(x))
        x = self.fc_v(x)

        return x.squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, feat_hidden_sizes, h_dim,
                 actor_hidden_sizes, critic_hidden_sizes, activation=nn.ReLU):
        super(ActorCritic, self).__init__()
        obs_dim = observation_space.shape[0]
        self.f_net = mlp(sizes=[obs_dim]+feat_hidden_sizes, activation=activation)
        if isinstance(action_space, Box):
            n_acts = action_space.shape[0]
            self.actor = GaussianActor(self.f_net, h_dim, actor_hidden_sizes,
                                       n_acts, activation=activation)

        elif isinstance(action_space, Discrete):
            n_acts = action_space.n
            self.actor = CategoricalActor(self.f_net, h_dim, actor_hidden_sizes,
                                          n_acts, activation=activation)

        self.critic = Critic(self.f_net, h_dim, critic_hidden_sizes, activation=activation)

    #################################################
    # Calculate hidden feature                      #
    # Input     : observation                       #
    # Output    : hidden feature                    #
    #################################################

    def pi(self, obs, act=None):
        return self.actor(obs, act)

    def v(self, obs):
        return self.critic(obs)
    
    def step(self, obs):
        pi = self.pi(obs)
        a = pi.sample()
        v = self.v(obs)
        logp_a = self.actor._log_prob_from_distribution(pi, a)

        return a.item(), v.item(), logp_a.item()
    
    def act(self, obs):
        return self.step(obs)[0]


def train(env_name='CartPole-v1', hidden_sizes=[256, 256], lr=7e-4, gamma=0.98,
          epochs=50, epis_epoch=30, steps_epi=1000, render=False):

    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous observation space"
    obs_dim = env.observation_space.shape
    if isinstance(env.action_space, Box):
        act_dim = env.action_space.shape
    elif isinstance(env.action_space, Discrete):
        act_dim = None

    h_len = len(hidden_sizes)
    feat_hidden_sizes = hidden_sizes[:-1]
    h_dim = hidden_sizes[-2]
    actor_hidden_sizes = hidden_sizes[-1:]
    critic_hidden_sizes = hidden_sizes[-1:]
    agent = ActorCritic(env.observation_space, env.action_space, feat_hidden_sizes, h_dim,
                        actor_hidden_sizes, critic_hidden_sizes, activation=nn.ReLU)
    buffer = VPGBuffer(obs_dim, act_dim, steps_epi, gamma)
    optimizer = Adam(agent.parameters(), lr=lr)

    def compute_loss_pi(obss, acts, advs):
        pi, logps = agent.pi(obss, acts)
        return -(logps*advs).mean()

    def compute_loss_v(obss, rets):
        vals = agent.v(obss)
        return ((rets - vals)**2).mean()

    def train_one_epoch():
        ret_epoch = []
        len_epoch = []
        for epi in range(epis_epoch):
            ret_epi = 0
            len_epi = 0
            obs = env.reset()
            done = False
            for t in range(steps_epi):
                act, val, logp = agent.step(torch.as_tensor(obs, dtype=torch.float32))
                next_obs, rew, done, _ = env.step(act)
                buffer.store(obs, act, rew, val, logp)
                obs = next_obs
                len_epi += 1
                ret_epi += rew
                if done:
                    ret_epoch.append(ret_epi)
                    len_epoch.append(len_epi)
                    ret_epi = 0
                    len_epi = 0
                    buffer.finish_path()
                    obs = env.reset()
                    done = False

            if buffer.path_start_idx != buffer.ptr:
                vT = agent.v(torch.as_tensor(obs, dtype=torch.float32)).item()
                buffer.finish_path(vT)

            obs_buf, act_buf, ret_buf, val_buf, adv_buf, logp_buf = buffer.get()

            optimizer.zero_grad()
            loss = compute_loss_pi(obs_buf, act_buf, adv_buf) + compute_loss_v(obs_buf, ret_buf)
            loss.backward()
            optimizer.step()

        return np.mean(len_epoch), np.mean(ret_epoch)

    for epoch in range(epochs):
        start = time.time()
        lens, rets = train_one_epoch()
        print('epoch: %3d \t time: %3f \t return: %.3f \t ep_len: %.3f' %
              (epoch+1, time.time()-start, rets, lens))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    # parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=0.0002)
    args = parser.parse_args()
    print('\nUsing vanila actor critic.\n')
    train(env_name=args.env_name, lr=args.lr)

