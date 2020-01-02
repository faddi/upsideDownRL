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
from torch.autograd import Variable
import json


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trajectory(NamedTuple):
    total_reward: int
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[int]
    prev_actions_logits: List[np.ndarray]

class DelayRewardsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self._total_reward = 0.0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self._total_reward += reward
        # modify ...
        r = reward
        # r = 0
        if done:
          r += self._total_reward
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
        self.action_size = action_size
        self.args = args
        self.fc1 = nn.Linear(state_size + 2 + action_size, self.args.hidden_size)
        self.fc2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        # self.lstm = nn.LSTM(self.args.hidden_size, self.args.hidden_size, num_layers=2)

        self.fc3 = nn.Linear(self.args.hidden_size, action_size)
        self.command_scale = args.command_scale

    def forward(self, state, prev_action, desired_return, desired_horizon, hidden_state, cell_state):

        cmd = torch.cat((desired_return * self.command_scale, desired_horizon * self.command_scale, prev_action), 1)
        # cmd = cmd.repeat(B)
        
        x = torch.cat((state, cmd), 1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # concat_command = torch.cat((desired_return, desired_horizon), 1)*self.command_scale
        # y = torch.relu(self.fc2(concat_command))
        # x = x * y

        # x = x.view(state.size(0), 1, -1)
        # x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        # x = x.view(state.size(0), -1)

        return self.fc3(x), hidden_state, cell_state
    
    def init_states(self):
        hidden_state = torch.zeros(2, 1, self.args.hidden_size).to(device)
        cell_state = torch.zeros(2, 1, self.args.hidden_size).to(device)
        return hidden_state, cell_state

    def reset_states(self, hidden_state, cell_state):
        hidden_state[:, :, :] = 0
        cell_state[:, :, :] = 0
        return hidden_state.detach(), cell_state.detach()


class UpsideDownRL(object):
    def __init__(self, env, args, max_reward = 250):
        super(UpsideDownRL, self).__init__()
        self.env = env
        self.args = args
        self.nb_actions = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        # Used to clip rewards so that B does not get unrealistic expected reward inputs.
        self.max_reward = max_reward
        self.episode_counter = 0
        self.train_counter = 0
        self.episode_counter = 0
        self.step_counter = 0

        # Use sorted dict to store experiences gathered.
        # This helps in fetching highest reward trajectories during exploratory stage.
        # self.experience = SortedDict()
        self.experience: SortedListWithKey = SortedListWithKey([], key=lambda entry: -entry.total_reward)
        self.B = BehaviorFunc(self.state_space, self.nb_actions, args).to(device)
        # self.optimizer = optim.Adam(self.B.parameters(), lr=self.args.lr)
        self.optimizer = optim.RMSprop(self.B.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=500, verbose=True, mode='min', threshold_mode='abs', threshold=1e-10)
        # self.optimizer = optim.Adadelta(self.B.parameters())
        self.use_random_actions = True  # True for the first training epoch.

        self.hidden_state, self.cell_state = self.B.init_states()

    # Generate an episode using given command inputs to the B function.
    def gen_episode(self, dr, dh):
        self.episode_counter += 1
        state = self.env.reset().astype(np.float32)
        self.hidden_state, self.cell_state = self.B.reset_states(self.hidden_state, self.cell_state)
        prev_action_logits = np.zeros(self.B.action_size)
        states = []
        rewards = []
        actions = []
        action_logits = []
        total_reward = 0
        while True:
            action, current_action_logits = self.select_action(state, prev_action_logits, dr, dh)
            next_state, reward, is_terminal, _ = self.env.step(action)
            next_state = next_state.astype(np.float32)
            if self.args.render:
                self.env.render()
            states.append(state)
            actions.append(action)
            action_logits.append(prev_action_logits)
            prev_action_logits = current_action_logits
            rewards.append(reward)
            total_reward += reward
            state = next_state
            dr = min(dr - reward, self.max_reward)
            dh = max(dh - 1, 1)
            self.step_counter += 1
            if is_terminal:
                break

        return total_reward, states, actions, rewards, action_logits

    # Fetch the desired return and horizon from the best trajectories in the current replay buffer
    # to sample more trajectories using the latest behavior function.
    def fill_replay_buffer(self):
        all_rewards = []
        all_dr = []
        all_dh = []
        dr, dh = self.get_desired_return_and_horizon(print_means=True)

        if len(self.experience) == self.args.replay_buffer_capacity:
            for i in range(self.args.episodes_per_iter):
                if len(self.experience) > 0:
                    self.experience.pop()

        while len(self.experience) < self.args.replay_buffer_capacity:
            # dr, dh = self.get_desired_return_and_horizon(print_means=True)
            all_dr.append(dr)
            all_dh.append(dh)
            total_reward, states, actions, rewards, prev_actions_logits = self.gen_episode(dr, dh)
            self.experience.add(Trajectory(total_reward=total_reward, states=states, actions=actions, rewards=rewards, prev_actions_logits=prev_actions_logits))
            all_rewards.append(total_reward)


        if self.args.verbose:
            if self.use_random_actions:
                print("Filled replay buffer with random actions")
            else:
                print("Filled replay buffer using BehaviorFunc")
        self.use_random_actions = False

        writer.add_scalar('Train/DesiredReturn', np.mean(all_dr), self.episode_counter)
        writer.add_scalar('Train/DesiredHorizon', np.mean(all_dh), self.episode_counter)
        writer.add_scalar('Train/FillMeanTotalReward', np.mean(all_rewards), self.episode_counter)


    def select_action(self, state, prev_action, desired_return=None, desired_horizon=None):
        if self.use_random_actions:
            action = np.random.randint(self.nb_actions)
            action_prob = np.zeros(self.B.action_size, dtype=np.float32)
        else:
            state = torch.from_numpy(state).to(device)
            dr = torch.from_numpy( np.array(desired_return, dtype=np.float32)).reshape(-1, 1).to(device)
            dh = torch.from_numpy( np.array(desired_horizon, dtype=np.float32).reshape(-1, 1)).to(device)
            prev_action = torch.tensor(np.array(prev_action, dtype=np.float32)).reshape(-1, self.B.action_size).to(device)
            action_prob_logits, self.hidden_state, self.cell_state = self.B(state.unsqueeze(0), prev_action, dr, dh, self.hidden_state, self.cell_state)

            action_prob = F.softmax(action_prob_logits, dim=1)
            # create a categorical distribution over action probabilities
            dist = Categorical(action_prob)
            action = dist.sample().item()
            action_prob = action_prob.detach().cpu().numpy()[0]
        return action, action_prob

    def get_desired_return_and_horizon(self, print_means=False):
        if (self.use_random_actions):
            return 0, 0

        h = []
        r = []
        # for i in range(min(self.args.explore_buffer_len, len(self.experience))):
        #     # episode = self.experience.popitem()  # will return in sorted order
        #     episode = self.experience[i]  # will return in sorted order
        #     # episode = self.experience.pop(0)  # will return in sorted order
        #     h.append(len(episode.actions))
        #     r.append(episode.total_reward)

        for i in range(min(self.args.explore_buffer_len, len(self.experience))):
            # episode = self.experience.popitem()  # will return in sorted order
            episode = self.experience[-(i + 1)]  # will return in sorted order
            # episode = self.experience.pop(0)  # will return in sorted order
            h.append(len(episode.actions))
            r.append(episode.total_reward)

        # for i in range(len(self.experience)):
        #     # episode = self.experience.popitem()  # will return in sorted order
        #     episode = self.experience[-(i + 1)]  # will return in sorted order
        #     # episode = self.experience.pop(0)  # will return in sorted order
        #     h.append(len(episode.actions))
        #     r.append(episode.total_reward)

        # episode_indexes = np.random.choice(len(self.experience), size=min(self.args.explore_buffer_len, len(self.experience)), replace=False)
        # for index in episode_indexes:
        #     episode = self.experience[index]
        #     h.append(len(episode.actions))
        #     r.append(episode.total_reward)

        mean_horizon_len = np.mean(h)
        mean_reward = np.random.uniform(low=np.mean(r), high=np.mean(r)+np.std(r))
        # mean_reward = np.random.uniform(low=np.mean(r), high=np.max(r) * 1.05)
        return mean_reward, mean_horizon_len

    def trainBehaviorFunc(self):
        # experience_dict = dict(self.experience)
        # experience_values = list(experience_dict.values())
        losses = []
        experience_values = self.experience
        for i in range(self.args.train_iter):
            self.hidden_state, self.cell_state = self.B.reset_states(self.hidden_state, self.cell_state)
            state = []
            dr = []
            dh = []
            target = []
            prev_actions_logits = []
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
                prev_actions_logits.append(trajectory.prev_actions_logits[i])

            self.optimizer.zero_grad()
            state = torch.from_numpy(np.array(state)).to(device)
            dr = torch.from_numpy(np.array(dr, dtype=np.float32).reshape(-1, 1)).to(device)
            dh = torch.from_numpy(np.array(dh, dtype=np.float32).reshape(-1, 1)).to(device)
            prev_actions_logits = torch.from_numpy(np.array(prev_actions_logits, dtype=np.float32).reshape(-1, self.B.action_size)).to(device)
            target = torch.from_numpy(np.array(target)).long().to(device)
            action_logits, _, _ = self.B(state, prev_actions_logits, dr, dh, self.hidden_state, self.cell_state)
            loss = nn.CrossEntropyLoss()
            output = loss(action_logits, target).mean()
            losses.append(output.item())
            output.backward()
            self.optimizer.step()
            # print(f"loss {output.item()}")

        print(f"loss {np.mean(losses)}")
        writer.add_scalar('Train/Loss', np.mean(losses), self.episode_counter)
        self.train_counter += 1
        self.scheduler.step(self.train_counter)

    # Evaluate the agent using the initial command input from the best topK performing trajectories.
    def evaluate(self):
        testing_rewards = []
        testing_steps = []
        dr, dh = self.get_desired_return_and_horizon(print_means=False)
        for i in range(self.args.evaluate_trials):
            total_reward, states, actions, rewards, prev_action_logits = self.gen_episode(dr, dh)
            testing_rewards.append(total_reward)
            testing_steps.append(len(rewards))

        print("Mean reward achieved : {}".format(np.mean(testing_rewards)))
        writer.add_scalar('Train/MeanReward', np.mean(testing_rewards), self.episode_counter)
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
                print("save")
                test_returns.append(self.evaluate())
                torch.save(self.B.state_dict(), os.path.join(
                    self.args.save_path, "model.pkl"))
                np.save(os.path.join(self.args.save_path,
                                     "testing_rewards"), test_returns)
            iterations += 1
            writer.add_scalar('Train/iterations', iterations , self.episode_counter)
            writer.add_scalar('Train/episodes', self.episode_counter , self.episode_counter)
            writer.add_scalar('Train/steps', self.step_counter, self.episode_counter)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameters for UpsideDown RL")
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--command_scale", type=float, default=0.01)
    parser.add_argument("--episodes_per_iter", type=int, default=30)
    parser.add_argument("--replay_buffer_capacity", type=int, default=500)
    parser.add_argument("--explore_buffer_len", type=int, default=30) # decides exploration pase, lower values -> more aggressive
    parser.add_argument("--eval_every_k_epoch", type=int, default=10)
    parser.add_argument("--evaluate_trials", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_iter", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="./models/")
    parser.add_argument("--do_save", default=True, action='store_true')
    parser.add_argument("--env_name", type=str, default='LunarLander-v2')
    # parser.add_argument("--env_name", type=str, default='CartPole-v0')

    args = parser.parse_args()
    if args.do_save:
      if not os.path.exists(args.save_path):
          os.mkdir(args.save_path)
      # else:
      #     sys.exit("Directory already exists.")

    writer.add_hparams(hparam_dict=vars(args), metric_dict=dict())
    env = gym.make(args.env_name)
    # env = gym.make("CartPole-v1")
    env = DelayRewardsWrapper(env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = UpsideDownRL(env, args, max_reward=400)
    agent.train()
    env.close()


if __name__ == "__main__":
    main()
