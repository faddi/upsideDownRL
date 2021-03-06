# Naive implementation of "Training Agents usingUpside-Down Reinforcement Learning"
# Paper link: https://arxiv.org/pdf/1912.02877.pdf
import numpy as np
import gym
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pdb
from collections import deque
from sortedcontainers import SortedListWithKey
import random
import os
import time
import sys
from typing import NamedTuple, List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trajectory(NamedTuple):
    total_reward: int
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[int]

class DelayRewardsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self._total_reward = 0.0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self._total_reward += reward
        # modify ...
        r = 0
        if done:
          r = self._total_reward
          self._total_reward = 0.0
        return next_state, r, done, info

    def reset(self, *args):
        self._total_reward = 0.0
        return self.env.reset(*args)


# Behavior func: If an agent is in a given state and desires a given return over a given horizon,
# which action should it take next?
# Input state s_t and command c_t=(dr_t, dh_t)
# where dr_t is the desired return and dh_t is desired time horizon


class BehaviorFunc(nn.Module):
    def __init__(self, state_size, action_size, args):
        super(BehaviorFunc, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(state_size, self.args.hidden_size)
        self.fc2 = nn.Linear(2, self.args.hidden_size)
        self.fc3 = nn.Linear(self.args.hidden_size, action_size)
        self.command_scale = args.command_scale

    def forward(self, state, desired_return, desired_horizon):
        x = torch.sigmoid(self.fc1(state))
        concat_command = torch.cat(
            (desired_return, desired_horizon), 1)*self.command_scale
        y = torch.sigmoid(self.fc2(concat_command))
        x = x * y
        return self.fc3(x)


class UpsideDownRL(object):
    def __init__(self, env, args, max_reward = 250):
        super(UpsideDownRL, self).__init__()
        self.env = env
        self.args = args
        self.nb_actions = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        # Used to clip rewards so that B does not get unrealistic expected reward inputs.
        self.max_reward = max_reward

        # Use sorted dict to store experiences gathered.
        # This helps in fetching highest reward trajectories during exploratory stage.
        # self.experience = SortedDict()
        self.experience: SortedListWithKey[Trajectory] = SortedListWithKey([], key=lambda entry: -entry.total_reward)
        self.B = BehaviorFunc(self.state_space, self.nb_actions, args).to(device)
        self.optimizer = optim.Adam(self.B.parameters(), lr=self.args.lr)
        self.use_random_actions = True  # True for the first training epoch.

    # Generate an episode using given command inputs to the B function.
    def gen_episode(self, dr, dh):
        state = self.env.reset().astype(np.float32)
        episode_data = []
        states = []
        rewards = []
        actions = []
        total_reward = 0
        while True:
            action = self.select_action(state, dr, dh)
            next_state, reward, is_terminal, _ = self.env.step(action)
            next_state = next_state.astype(np.float32)
            if self.args.render:
                self.env.render()
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            dr = min(dr - reward, self.max_reward)
            dh = max(dh - 1, 1)
            if is_terminal:
                break

        return total_reward, states, actions, rewards

    # Fetch the desired return and horizon from the best trajectories in the current replay buffer
    # to sample more trajectories using the latest behavior function.
    def fill_replay_buffer(self):
        dr, dh = self.get_desired_return_and_horizon()
        # self.experience.clear()
        for i in range(self.args.replay_buffer_capacity):
            total_reward, states, actions, rewards = self.gen_episode(dr, dh)
            # self.experience.__setitem__(
            #     total_reward, (states, actions, rewards))
            self.experience.add(Trajectory(total_reward=total_reward, states=states, actions=actions, rewards=rewards))

            if len(self.experience) > self.args.replay_buffer_capacity:
                self.experience.pop()

        if self.args.verbose:
            if self.use_random_actions:
                print("Filled replay buffer with random actions")
            else:
                print("Filled replay buffer using BehaviorFunc")
        self.use_random_actions = False

    def select_action(self, state, desired_return=None, desired_horizon=None):
        if self.use_random_actions:
            action = np.random.randint(self.nb_actions)
        else:
            action_prob = self.B(torch.from_numpy(state).to(device),
                                 torch.from_numpy(
                                     np.array(desired_return, dtype=np.float32)).reshape(-1, 1).to(device),
                                 torch.from_numpy(
                                     np.array(desired_horizon, dtype=np.float32).reshape(-1, 1)).to(device)
                                 )
            action_prob = F.softmax(action_prob, dim=1)
            # create a categorical distribution over action probabilities
            dist = Categorical(action_prob)
            action = dist.sample().item()
        return action

    # Todo: don't popitem from the experience buffer since these best-performing trajectories can have huge impact on learning of B
    def get_desired_return_and_horizon(self):
        if (self.use_random_actions):
            return 0, 0

        h = []
        r = []
        for i in range(self.args.explore_buffer_len):
            # episode = self.experience.popitem()  # will return in sorted order
            episode = self.experience[i]  # will return in sorted order
            # episode = self.experience.pop(0)  # will return in sorted order
            h.append(len(episode.actions))
            r.append(episode.total_reward)

        mean_horizon_len = np.mean(h)
        mean_reward = np.random.uniform(
            low=np.mean(r), high=np.mean(r)+np.std(r))
        return mean_reward, mean_horizon_len

    def trainBehaviorFunc(self):
        # experience_dict = dict(self.experience)
        # experience_values = list(experience_dict.values())
        losses = []
        experience_values = self.experience
        for i in range(self.args.train_iter):
            state = []
            dr = []
            dh = []
            target = []
            indices = np.random.choice(len(experience_values), self.args.batch_size, replace=True)
            train_episodes = [experience_values[i] for i in indices]
            t1 = [np.random.choice(len(e.states)-2, 1) for e in train_episodes]

            # for pair in zip(t1, train_episodes):
            for index_list, trajectory in zip(t1, train_episodes):
                i = index_list[0]
                state.append(trajectory.states[i])
                dr.append(np.sum(trajectory.rewards[i:]))
                dh.append(len(trajectory.actions)-i)
                target.append(trajectory.actions[i])

            self.optimizer.zero_grad()
            state = torch.from_numpy(np.array(state)).to(device)
            dr = torch.from_numpy(
                np.array(dr, dtype=np.float32).reshape(-1, 1)).to(device)
            dh = torch.from_numpy(
                np.array(dh, dtype=np.float32).reshape(-1, 1)).to(device)
            target = torch.from_numpy(np.array(target)).long().to(device)
            action_logits = self.B(state, dr, dh)
            loss = nn.CrossEntropyLoss()
            output = loss(action_logits, target).mean()
            losses.append(output.item())
            output.backward()
            self.optimizer.step()

        print(f"loss {np.mean(losses)}")

    # Evaluate the agent using the initial command input from the best topK performing trajectories.
    def evaluate(self):
        testing_rewards = []
        testing_steps = []
        dr, dh = self.get_desired_return_and_horizon()
        for i in range(self.args.evaluate_trials):
            total_reward, states, actions, rewards = self.gen_episode(dr, dh)
            testing_rewards.append(total_reward)
            testing_steps.append(len(rewards))

        print("Mean reward achieved : {}".format(np.mean(testing_rewards)))
        return np.mean(testing_rewards)

    def train(self):
        # Fill replay buffer with random actions for the first time.
        self.fill_replay_buffer()
        iterations = 0
        test_returns = []
        while True:
            self.evaluate()
            # Train behavior function with trajectories stored in the replay buffer.
            self.trainBehaviorFunc()
            self.fill_replay_buffer()

            if iterations % self.args.eval_every_k_epoch == 0 and self.args.do_save:
                test_returns.append(self.evaluate())
                torch.save(self.B.state_dict(), os.path.join(
                    self.args.save_path, "model.pkl"))
                np.save(os.path.join(self.args.save_path,
                                     "testing_rewards"), test_returns)
            iterations += 1


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameters for UpsideDown RL")
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--command_scale", type=float, default=0.01)
    parser.add_argument("--replay_buffer_capacity", type=int, default=300)
    parser.add_argument("--explore_buffer_len", type=int, default=20)
    parser.add_argument("--eval_every_k_epoch", type=int, default=5)
    parser.add_argument("--evaluate_trials", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--train_iter", type=int, default=200)
    parser.add_argument("--save_path", type=str, default="DefaultParams/")
    parser.add_argument("--do_save", default=False, action='store_true')

    args = parser.parse_args()
    if args.do_save:
      if not os.path.exists(args.save_path):
          os.mkdir(args.save_path)
      else:
          sys.exit("Directory already exists.")

    env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v1")
    env = DelayRewardsWrapper(env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = UpsideDownRL(env, args)
    agent.train()
    env.close()


if __name__ == "__main__":
    main()
