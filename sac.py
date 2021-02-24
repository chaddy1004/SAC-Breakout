import argparse
import os
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import tensorflow as tf
from torch.nn import Module, Linear, ReLU, Sequential, Softmax, Parameter, Conv2d, BatchNorm2d, MaxPool2d, Flatten, \
    AvgPool2d
from torch.optim import Adam

from utils import frame_to_state

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

mse_loss_function = torch.nn.MSELoss()


# cce_loss_function = torch.nn.CrossEntropyLoss()


def conv3x3(in_channels, out_channels, stride=1):
    return Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)


class BasicBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=True):
        super(BasicBlock, self).__init__()
        self.conv = conv3x3(in_channels, out_channels, stride)
        self.bn = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample:
            self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.downsample:
            out = self.maxpool(out)
        out = self.relu(out)
        return out


class Actor(Module):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.layer1 = BasicBlock(in_channels=1, out_channels=16, stride=1, downsample=True)  # 42 42 16
        self.layer2 = BasicBlock(in_channels=16, out_channels=32, stride=1, downsample=True)  # 21 21 8
        self.layer3 = BasicBlock(in_channels=32, out_channels=64, stride=1, downsample=True)  # 10 10 64
        self.layer4 = BasicBlock(in_channels=64, out_channels=128, stride=1, downsample=True)  # 5  5 128

        self.gap = AvgPool2d(5)
        # The softmax has to be there to make it a probability distribution (pi)
        self.final_lin = Sequential(Linear(in_features=128, out_features=n_actions), Softmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.gap(x)
        output = self.lin(features)
        return output


class Critic(Module):
    def __init__(self, n_actions):
        super(Critic, self).__init__()
        self.layer1 = BasicBlock(in_channels=1, out_channels=16, stride=1, downsample=True)  # 42 42 16
        self.layer2 = BasicBlock(in_channels=16, out_channels=32, stride=1, downsample=True)  # 21 21 8
        self.layer3 = BasicBlock(in_channels=32, out_channels=64, stride=1, downsample=True)  # 10 10 64
        self.layer4 = BasicBlock(in_channels=64, out_channels=128, stride=1, downsample=True)  # 5  5 128

        self.gap = AvgPool2d(5)
        self.lin = Linear(in_features=128, out_features=n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.gap(x)
        output = self.lin(features)
        return output


class SAC_discrete:
    def __init__(self, n_states, n_actions):
        self.replay_size = 1024
        self.experience_replay = deque(maxlen=self.replay_size)
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.0003
        self.batch_size = 64
        self.gamma = 0.99
        self.actor = Actor(n_actions=n_actions)
        self.critic = Critic(n_actions=n_actions)
        self.target_critic = Critic(n_actions=n_actions)
        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr)
        self.H = 0.98 * (-np.log(1 / self.n_actions))
        self.Tau = 0.5
        self.alpha = Parameter(torch.tensor(0.5))
        self.optim_alpha = Adam(params=[self.alpha], lr=self.lr)

    def get_action(self, state, test=False):
        action_probs = self.actor(state.float())
        action_probs_np = action_probs.detach().cpu().numpy().squeeze()
        if not test:
            return int(np.random.choice(self.n_actions, 1, p=action_probs_np)), action_probs
        else:
            return int(np.argmax(action_probs_np)), action_probs

    def get_v(self, state_batch):
        action_probs = self.actor(state_batch).detach().unsqueeze(-1)  # (batch, 4, 1)
        action_probs_transpose = torch.transpose(action_probs, 1, -1)  # (batch, 1, 4)
        q_values = self.target_critic(state_batch).unsqueeze(-1)  # (batch, 4, 1)
        log_action_probs = torch.log(action_probs)
        value = q_values - self.alpha * log_action_probs
        return torch.matmul(action_probs_transpose, value).squeeze(-1)  # (batch, 1)

    def train_critic(self, s_currs, a_currs, r, s_nexts, dones):

        predicts = self.critic(s_currs)  # (batch, actions)

        a_indices = a_currs.long()

        v_vector = self.get_v(s_nexts)

        predicts_per_action = torch.gather(predicts, 1, a_indices)  # (batch, 1)

        target = r + self.gamma * v_vector  # (batch, 1)
        done_indices = np.argwhere(dones)
        if done_indices.shape[0] > 0:
            done_indices = torch.squeeze(dones.nonzero())
            target[done_indices, 0] = torch.squeeze(r[done_indices])

        self.optim_critic.zero_grad()
        loss = mse_loss_function(predicts_per_action, target)
        loss.backward()
        self.optim_critic.step()
        return

    # actor -> policy network (improve policy network)
    def train_actor(self, s_currs, a_currs, r, s_nexts, dones):
        self.optim_actor.zero_grad()

        action_prob = self.actor(s_currs).unsqueeze(-1)  # (batch, actions, 1)

        log_action_prob = torch.log(action_prob)

        action_prob = torch.transpose(action_prob, dim0=1, dim1=-1)
        q_values = self.critic(s_currs).detach().unsqueeze(-1)
        loss = torch.matmul(action_prob, (self.alpha * log_action_prob - q_values))
        loss = torch.mean(loss)
        loss.backward()
        self.optim_actor.step()

    def train_alpha(self, s_currs, a_currs, r, s_nexts, dones):
        self.optim_alpha.zero_grad()
        action_prob = self.actor(s_currs).unsqueeze(-1)  # (batch, actions, 1)

        log_action_prob = torch.log(action_prob)

        action_prob = torch.transpose(action_prob, dim0=1, dim1=-1)

        loss = torch.matmul(action_prob, (-1 * self.alpha * (log_action_prob + self.H)))
        loss = torch.mean(loss)
        loss.backward()
        self.optim_alpha.step()

    def process_batch(self, x_batch):
        s_currs = torch.zeros((self.batch_size, self.n_states))
        a_currs = torch.zeros((self.batch_size, 1))
        r = torch.zeros((self.batch_size, 1))
        s_nexts = torch.zeros((self.batch_size, self.n_states))
        dones = torch.zeros((self.batch_size,))

        for batch in range(self.batch_size):
            s_currs[batch] = x_batch[batch].s_curr
            a_currs[batch] = x_batch[batch].a_curr
            r[batch] = x_batch[batch].reward
            s_nexts[batch] = x_batch[batch].s_next
            dones[batch] = x_batch[batch].done

        return s_currs, a_currs, r, s_nexts, dones

    def train(self, x_batch):
        s_currs, a_currs, r, s_nexts, dones = self.process_batch(x_batch=x_batch)
        self.train_critic(s_currs, a_currs, r, s_nexts, dones)
        self.train_actor(s_currs, a_currs, r, s_nexts, dones)
        self.train_alpha(s_currs, a_currs, r, s_nexts, dones)
        self.update_weights()
        return

    def update_weights(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)


def main(episodes, exp_name):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    # no frame skip as DQFD did not have this
    # v4 always performs the chosen action
    env = gym.make('BreakoutNoFrameskip-v4')
    states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.n
    agent = SAC_discrete(n_states=states, n_actions=n_actions)
    warmup_ep = 0
    # import time
    # env.reset()
    # _, r, done, info = env.step(1)
    # while True:
    #     _, r, done, info = env.step(3)
    #     print(r,done, info)
    #     env.render()
    #     time.sleep(1/10)

    for ep in range(episodes):
        frame_curr = env.reset()
        # pre-processes the original raw rgb frame to resized, grayscale PyTorch tensor
        s_curr_tensor = frame_to_state(frame_curr)
        done = False
        dead = False
        score = 0
        agent.update_weights()  # update weight every time an episode ends
        n_lives = 5
        while not done:
            a_curr, action_prob = agent.get_action(s_curr_tensor)
            frame_next, r, done, info = env.step(a_curr)

            if n_lives > info['ale.lives']:
                dead = True
                n_lives = info['ale_lives']

            r = np.clip(r, -1.0, 1.0)

            s_next_tensor = frame_to_state(frame_next)

            score += r
            r = np.clip(r, -1, 1)
            sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])

            # must re-make training dataloader since the dataset is now updated with aggregation of new data
            sample.s_curr = s_curr_tensor
            sample.a_curr = a_curr
            sample.reward = r
            sample.s_next = s_next_tensor
            sample.done = done

            if len(agent.experience_replay) < agent.replay_size:
                agent.experience_replay.append(sample)
                s_curr_tensor = s_next_tensor
                continue
            else:
                agent.experience_replay.append(sample)
                x_batch = random.sample(agent.experience_replay, agent.batch_size)
                agent.train(x_batch)
                s_curr_tensor = s_next_tensor

            # the next state is not the current state after each step

            if done:
                score = score if score == 500 else score + 100
                print(f"ep:{ep - warmup_ep}:################Goal Reached###################", score)
                # print(agent.alpha)
                with writer.as_default():
                    tf.summary.scalar("reward", r, ep)
                    tf.summary.scalar("score", score, ep)
    return agent


def env_with_render(agent):
    done = False

    # no frame skip as DQFD did not have this
    # v4 always performs the chosen action
    env = gym.make('BreakoutNoFrameskip-v4')
    score = 0
    states = env.observation_space.shape[0]  # shape returns a tuple
    s_curr = np.reshape(env.reset(), (1, states))
    while True:
        if done:
            print(score)
            score = 0
            s_curr = np.reshape(env.reset(), (1, states))
        env.render()
        s_curr_tensor = torch.from_numpy(s_curr)
        a_curr, _ = agent.get_action(s_curr_tensor, test=True)
        s_next, r, done, _ = env.step(a_curr)
        s_next = np.reshape(s_next, (1, states))
        s_curr = s_next
        score += r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="PG_REINFORCE_pt", help="exp_name")
    ap.add_argument("--episodes", type=int, default=460, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    env_with_render(agent=trained_agent)
