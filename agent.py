import random
from collections import deque

import numpy as np
import torch
from torch.nn import Module, Linear, ReLU, Sequential, Softmax, Parameter, Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d
from torch.optim import Adam

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

mse_loss_function = torch.nn.MSELoss()

if torch.cuda.device_count() > 0:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


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
        self.lin = Sequential(Linear(in_features=128, out_features=n_actions), Softmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.gap(x)
        # "flattens" the feature vector so that it can be processed with linear layer
        features = features.view(-1, 128)
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
        # "flattens" the feature vector so that it can be processed with linear layer
        features = features.view(-1, 128)
        output = self.lin(features)
        return output


class SAC_discrete:
    def __init__(self, n_states, n_actions):
        self.replay_size = 120000
        self.experience_replay = deque(maxlen=self.replay_size)
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.0003
        self.batch_size = 64
        self.gamma = 0.99
        self.actor = Actor(n_actions=n_actions).to(DEVICE)
        self.critic = Critic(n_actions=n_actions).to(DEVICE)
        self.target_critic = Critic(n_actions=n_actions).to(DEVICE)
        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr)
        self.H = 0.98 * (-np.log(1 / self.n_actions))
        self.Tau = 0.5
        self.alpha = Parameter(torch.tensor(0.5))
        self.optim_alpha = Adam(params=[self.alpha], lr=self.lr)

    def get_action(self, state, test=False):
        action_probs = self.actor(state.float().to(DEVICE))
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

    def train_critic(self, s_currs, a_currs, r, s_nexts, deads):

        predicts = self.critic(s_currs)  # (batch, actions)

        a_indices = a_currs.long()

        v_vector = self.get_v(s_nexts)

        predicts_per_action = torch.gather(predicts, 1, a_indices)  # (batch, 1)

        target = r + self.gamma * v_vector  # (batch, 1)
        done_indices = np.argwhere(deads.cpu().numpy())
        if done_indices.shape[0] > 0:
            done_indices = torch.squeeze(torch.nonzero(deads)).to(DEVICE)
            target[done_indices, 0] = torch.squeeze(r[done_indices])

        self.optim_critic.zero_grad()
        loss = mse_loss_function(predicts_per_action, target)
        loss.backward()
        self.optim_critic.step()
        return

    # actor -> policy network (improve policy network)
    def train_actor(self, s_currs):
        self.optim_actor.zero_grad()

        action_prob = self.actor(s_currs).unsqueeze(-1)  # (batch, actions, 1)

        log_action_prob = torch.log(action_prob)

        action_prob = torch.transpose(action_prob, dim0=1, dim1=-1)
        q_values = self.critic(s_currs).detach().unsqueeze(-1)
        loss = torch.matmul(action_prob, (self.alpha * log_action_prob - q_values))
        loss = torch.mean(loss)
        loss.backward()
        self.optim_actor.step()

    def train_alpha(self, s_currs):
        self.optim_alpha.zero_grad()
        action_prob = self.actor(s_currs).unsqueeze(-1)  # (batch, actions, 1)

        log_action_prob = torch.log(action_prob)

        action_prob = torch.transpose(action_prob, dim0=1, dim1=-1)

        loss = torch.matmul(action_prob, (-1 * self.alpha * (log_action_prob + self.H)))
        loss = torch.mean(loss)
        loss.backward()
        self.optim_alpha.step()

    def process_batch(self, x_batch):
        img_shape = x_batch[0].s_curr.shape[-1]
        ch_shape = x_batch[0].s_curr.shape[1]

        s_currs = torch.zeros((self.batch_size, ch_shape, img_shape, img_shape))
        a_currs = torch.zeros((self.batch_size, 1))
        r = torch.zeros((self.batch_size, 1))
        s_nexts = torch.zeros((self.batch_size, ch_shape, img_shape, img_shape))
        deads = torch.zeros((self.batch_size, 1))

        for batch in range(self.batch_size):
            s_currs[batch] = x_batch[batch].s_curr
            a_currs[batch] = x_batch[batch].a_curr
            r[batch] = x_batch[batch].reward
            s_nexts[batch] = x_batch[batch].s_next
            deads[batch] = x_batch[batch].dead

        return s_currs.to(DEVICE), a_currs.to(DEVICE), r.to(DEVICE), s_nexts.to(DEVICE), deads.to(DEVICE)

    def train(self, x_batch):
        s_currs, a_currs, r, s_nexts, deads = self.process_batch(x_batch=x_batch)
        self.train_critic(s_currs, a_currs, r, s_nexts, deads)
        self.train_actor(s_currs)
        self.train_alpha(s_currs)
        self.update_weights()
        return

    def update_weights(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)
