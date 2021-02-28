import argparse
import os
import random
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf
import torch

from agent import SAC_discrete
from utils import frame_to_tensor

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

if torch.cuda.device_count() > 0:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

mse_loss_function = torch.nn.MSELoss()


# cce_loss_function = torch.nn.CrossEntropyLoss()


def main(episodes, exp_name):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    # no frame skip as DQFD did not have this
    # v4 always performs the chosen action
    # env = gym.make('BreakoutNoFrameskip-v4')
    env = gym.make('BreakoutDeterministic-v4')
    states = env.observation_space.shape[0]  # shape returns a tuple
    # n_actions = env.action_space.n
    n_actions = 3
    agent = SAC_discrete(n_states=states, n_actions=n_actions)
    warmup_ep = 0
    # import time
    # env.reset()
    # # _, r, done, info = env.step(1)
    # while True:
    #     observe, r, done, info = env.step(1)
    #     # print(observe)
    #     print(r,done, info)
    #     # env.render()
    #     time.sleep(1/10)

    for ep in range(episodes):
        env.reset()
        # pre-processes the original raw rgb frame to resized, grayscale PyTorch tensor
        done = False
        dead = False
        score = 0
        agent.update_weights()  # update weight every time an episode ends
        n_lives = 5
        frame_next = None
        for _ in range(0, random.randint(1, 30)):
            frame_next, *_ = env.step(1)

        frame_curr = frame_next
        frame_curr_tensor = frame_to_tensor(frame_curr)
        step = 0

        # state will now refer to the concatenated frames
        s_curr_tensor = torch.cat([frame_curr_tensor, frame_curr_tensor, frame_curr_tensor, frame_curr_tensor], dim=1)
        while not done:
            a_curr, action_prob = agent.get_action(s_curr_tensor)
            frame_next, r, done, info = env.step(a_curr+1)
            # print(r, done, info)
            if n_lives > info['ale.lives']:
                dead = True
                n_lives = info['ale.lives']

            r = np.clip(r, -1.0, 1.0)

            frame_next_tensor = frame_to_tensor(frame_next)

            s_next_tensor = torch.cat([frame_next_tensor, s_curr_tensor[:, 0:3, ...]], dim=1)

            score += r
            r = np.clip(r, -1, 1)
            sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'dead'])

            # must re-make training dataloader since the dataset is now updated with aggregation of new data
            sample.s_curr = s_curr_tensor
            sample.a_curr = a_curr
            sample.reward = r
            sample.s_next = s_next_tensor
            sample.dead = dead

            if len(agent.experience_replay) < (agent.replay_size//2):
                agent.experience_replay.append(sample)
                s_curr_tensor = s_next_tensor
            else:
                agent.experience_replay.append(sample)
                x_batch = random.sample(agent.experience_replay, agent.batch_size)
                agent.train(x_batch)
                print(f"score: {score}, lives: {n_lives}")
                print(f"Episode: {ep}, Step : {step}, train on batch complete")
                s_curr_tensor = s_next_tensor

            # reset the dead-ness to play next episode
            if dead:
                dead = False
            if done:
                print(f"ep:{ep - warmup_ep}:################Game Over###################", score)
                with writer.as_default():
                    tf.summary.scalar("score", score, ep)
            step += 1
        if (ep+1) % 500 == 0:
            torch.save(agent.actor.state_dict(), f"actor_{ep+1}.pt")
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
    ap.add_argument("--episodes", type=int, default=50000, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"])
    env_with_render(agent=trained_agent)
