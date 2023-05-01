import copy
import pickle
import random
import gymnasium as gym
import torch
from collections import deque, namedtuple
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *


# Class for training an RL agent with Actor-Critic
class ACTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = ACAgent(env=self.env, params=self.params)
        self.actor_net = ActorNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.critic_net = CriticNet(input_size=self.env.observation_space.shape[0], output_size=1, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.actor_optimizer = Adam(params=self.actor_net.parameters(), lr=self.params['actor_lr'])
        self.critic_optimizer = Adam(params=self.critic_net.parameters(), lr=self.params['critic_lr'])
        self.trajectory = None

    def run_training_loop(self):
        list_ro_reward = list()
        for ro_idx in range(self.params['n_rollout']):
            self.trajectory = self.agent.collect_trajectory(policy=self.actor_net)
            self.update_critic_net()
            self.estimate_advantage()
            self.update_actor_net()
            # TODO: Calculate avg reward for this rollout
            # HINT: Add all the rewards from each trajectory. There should be "ntr" trajectories within a single rollout.
            avg_ro_reward = sum([sum(traj['reward']) for traj in self.trajectory])/self.params['n_trajectory_per_rollout'] #summing all trajectories and storing average reward  
            print(f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')#print average rollout reward
            # Append the average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def update_critic_net(self):
        for critic_iter_idx in range(self.params['n_critic_iter']):
            self.update_target_value()
            for critic_epoch_idx in range(self.params['n_critic_epoch']):
                critic_loss = self.estimate_critic_loss_function()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

    def update_target_value(self, gamma=0.99):
        # TODO: Update target values
        # HINT: Use definition of target-estimate from equation 7 of teh assignment PDF
        states = torch.tensor(self.trajectory['state_value'], dtype=torch.float32, device=get_device())# creating a tensor and passing dtype argument
        with torch.no_grad():
            target_values = self.critic_net(states)#obtain predicted q values for each state
            target_values = target_values.squeeze(dim=1).tolist()# and convert it to list

        self.trajectory['state_value'] = target_values[:-1]#updated predicted Q values expect for last state
        self.trajectory['target_value'] = target_values[1:]#updated predicted Q values expect for first state



    def estimate_advantage(self, gamma=0.99):
        # TODO: Estimate advantage
        # HINT: Use definition of advantage-estimate from equation 6 of teh assignment PDF
        advantage = []#  an empty list
        for i  in range(self.params['n_trajectory_per_rollout']):
            adv = list()
            for j in range(len(self.trajectory['state_value'][i])):
                # Calculate the advantage 
                result = self.trajectory['reward'][i][j] + gamma * self.trajectory['state_value'][i][j + 1] * (1 - self.trajectory['terminated'][i][j]) - self.trajectory['state_value'][i][j]
                adv.append(result)#append to list
        advantage.append(adv)
        self.trajectory['advantage'] = torch.tensor(advantage, dtype=torch.float32, device=get_device())

    def update_actor_net(self):
        actor_loss = self.estimate_actor_loss_function()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

    def estimate_critic_loss_function(self):
        # TODO: Compute critic loss function
        # HINT: Use definition of critic-loss from equation 7 of teh assignment PDF. It is the MSE between target-values and state-values.
        states = torch.tensor(self.trajectory['state_value'], dtype=torch.float32, device=get_device())
        target_values = torch.tensor(self.trajectory['target_value'], dtype=torch.float32, device=get_device()).unsqueeze(1)
        state_values = self.critic_net(states)
        critic_loss = nn.MSELoss()(target_values, state_values)#MSE between target and state values
        return critic_loss

    def estimate_actor_loss_function(self):
        actor_loss = list()
        for t_idx in range(self.params['n_trajectory_per_rollout']):
            advantage = apply_discount(self.trajectory['advantage'][t_idx])
            # TODO: Compute actor loss function
            log_probs = self.trajectory['log_prob'][t_idx]
            actor_loss.append(torch.mean(-log_probs * advantage))#calculating actor loss
        actor_loss = torch.stack(actor_loss).mean()
        return actor_loss

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action_idx, log_prob = self.actor_net(torch.tensor(obs, dtype=torch.float32, device=get_device()))
            obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


# CLass for actor-net
class ActorNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(ActorNet, self).__init__()
        # TODO: Define the actor net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Softmax(dim=-1))

    def forward(self, obs):
        # TODO: Forward pass of actor net
        # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)
        x = self.ff_net(obs)# Pass obs through ff net
        dist = Categorical(logits=x)# Use Categorical  to draw samples and log-prob from model output
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        return action_index, log_prob# Return action index and  log prob


# CLass for actor-net
class CriticNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(CriticNet, self).__init__()
        # TODO: Define the critic net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        # TODO: Forward pass of critic net
        # HINT: (get state value from the network using the current observation)
            state_value = self.ff_net(obs)#pass obs to ff_net neural network
            return state_value


# Class for agent
class ACAgent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
            while True:
                obs = torch.tensor(obs, dtype=torch.float32, device=get_device())
                # Save observation
                trajectory_buffer['obs'].append(obs)
                action_idx, log_prob = policy(obs)
                obs, reward, terminated, truncated, info = self.env.step(self.action_space[action_idx.item()])
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['obs'].append(torch.stack(trajectory_buffer['obs']))
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer


class DQNTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.q_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = self.params['init_epsilon']
        self.optimizer = Adam(params=self.q_net.parameters(), lr=self.params['lr'])
        self.replay_memory = ReplayMemory(capacity=self.params['rm_cap'])

    def run_training_loop(self):
        list_ep_reward = list()
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        for idx_episode in range(self.params['n_episode']):
            ep_len = 0
            while True:
                ep_len += 1
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    self.epsilon = max(self.epsilon*self.params['epsilon_decay'], self.params['min_epsilon'])
                    next_obs = None
                    self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                    list_ep_reward.append(ep_len)
                    print(f'End of episode {idx_episode} with epsilon = {self.epsilon: 0.2f} and reward = {ep_len}, memory = {len(self.replay_memory.buffer)}')
                    obs, _ = self.env.reset()
                    break
                self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                obs = copy.deepcopy(next_obs)
                self.update_q_net()
                self.update_target_net()
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ep_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def get_action(self, obs):
        # TODO: Implement the epsilon-greedy behavior
        # HINT: The agent will will choose action based on maximum Q-value with
        # '1-ε' probability, and a random action with 'ε' probability.
        if torch.rand(1) < self.epsilon: # if random number is less than epsilon
            return self.env.action_space.sample()# # take random action from action space
        else:
            with torch.no_grad():
                # convert observation to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(get_device())
                q_values = self.q_net(obs_tensor)# pass tensor through Q-network to get Q-values
                action = q_values.argmax().item() # select action with highest Q-value
        return action#return action which is selected

    def update_q_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        # TODO: Update Q-net
        # HINT: You should draw a batch of random samples from the replay buffer
        # and train your Q-net with that sampled batch.

        # Draw a batch of random samples from the replay buffer
        transitions = self.replay_memory.sample(self.params['batch_size'])
        transitions = [t for t in transitions if t is not None]
        transitions = list(filter(lambda x: x is not None, transitions))
    
        if len(transitions) == 0:#if no valid transitions in sample then return
            return
        #seperating components of transition
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        mask_batch = []

        for t in transitions:
            obs_batch.append(t[0])
            action_batch.append(t[1])
            reward_batch.append(t[2])
            next_obs_batch.append(t[3])
            mask_batch.append(t[4])
        #converting batched data into the tensors
        obs_batch = torch.tensor(obs_batch).float().to(get_device())
        action_batch = torch.tensor(action_batch).long().to(get_device())
        reward_batch = torch.tensor(reward_batch).float().to(get_device())
        next_obs_batch = torch.tensor(next_obs_batch).float().to(get_device())
        mask_batch = torch.tensor(mask_batch).float().to(get_device())
        predicted_state_value = self.q_net(obs_batch).gather(1, action_batch)

    # Compute targets for Q-learning update
        with torch.no_grad():
            next_q_values = self.target_net(next_obs_batch).max(1)[0].unsqueeze(1)
            target_value = reward_batch + self.params['gamma'] * mask_batch * next_q_values


        # Compute predicted state values and target values
        criterion = nn.SmoothL1Loss()
        q_loss = criterion(predicted_state_value, target_value.unsqueeze(1))
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        q_net_state_dict = self.q_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = self.params['tau']*q_net_state_dict[key] + (1 - self.params['tau'])*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        self.epsilon = 0.0
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action = self.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


class ReplayMemory:
    # TODO: Implement replay buffer
    # HINT: You can use python data structure deque to construct a replay buffer
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)#buffer max capacity

    def push(self, *args):
        self.buffer.append(args)#transition is added to the buffer
    def sample(self, n_samples):
        transitions = []#sample transitions from buffer
        while len(transitions) < n_samples:
            sample = random.choice(self.buffer)
            if sample is not None:
                transitions.append(sample)
        return transitions



class QNet(nn.Module):
    # TODO: Define Q-net
    # This is identical to policy network from HW1
    def __init__(self, input_size, output_size, hidden_dim):
        super(QNet, self).__init__()
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        #
        self.ff_net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),# input layer
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),#hidden layer
            nn.ReLU(),#second layer activation function
            nn.Linear(self.hidden_dim, self.output_size)#output layer
        )

    def forward(self, obs):
        return self.ff_net(obs)

