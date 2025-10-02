import torch
import torch.nn as nn
from torchinfo import summary

from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
import pickle
import os

from .replay_memory import ReplayMemory


class DQN:
  def __init__(self, memory_size, number_of_observations, number_of_actions, exploration_max, exploration_decay, exploration_min, gamma, batch_size, 
                hidden_layers, device, learning_rate, run_name, seed=1234):
      # Initialize variables and create neural model
      self.memory_size = memory_size
      self.number_of_observations = number_of_observations
      self.number_of_actions = number_of_actions
      self.exploration_rate = exploration_max
      self.exploration_decay = exploration_decay
      self.exploration_min = exploration_min
      self.gamma = gamma
      self.batch_size = batch_size
      self.device = device
      self.seed = seed

      # Save metrics
      self.episode_return = []
      self.mean_episode_return = []
      self.episode_length = []
      self.exploration_rate_episode = []
      
      self.memory = ReplayMemory(number_of_observations, memory_size, self.device)

      self.hidden_layers = hidden_layers

      self.model = self.create_model().to(self.device)
      self.target_model = self.create_model().to(self.device)
      
      #self.model = torch.compile(self.model)
      #self.target_model = torch.compile(self.target_model)

      self.update_target_network()

      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
      self.criterion = nn.MSELoss()

      # Initialize Tensorboard
      self.log_dir = os.path.join("logs_tensorflow", "dqn", run_name)
      self.writer = SummaryWriter(self.log_dir)
  
      if device.type == "cuda":
        self.scaler = GradScaler("cuda")
      else:
        self.scaler = GradScaler("cpu")

  def create_model(self):
      layers = []

      # Input layer
      layer = nn.Linear(self.number_of_observations, self.hidden_layers[0])
      # Initialize weights and bias
      nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
      nn.init.zeros_(layer.bias)
      layers.append(layer)
      layers.append(nn.ReLU())

      # Hidden layers
      for in_size, out_size in zip(self.hidden_layers, self.hidden_layers[1:]):
        layer = nn.Linear(in_size, out_size)
        # Initialize weights and bias
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        layers.append(nn.ReLU())

      # Outpur layer
      layer = nn.Linear(self.hidden_layers[-1], int(self.number_of_actions))
      # Initialize weights and bias
      nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
      nn.init.zeros_(layer.bias)
      layers.append(layer)

      model = nn.Sequential(*layers)
      return model

  def update_target_network(self):
      self.target_model.load_state_dict(self.model.state_dict())

  def remember(self, state, action, reward, next_state, terminal_state):
      # Store a tuple (s, a, r, s') for experience replay
      self.memory.store_transition(state, action, reward, next_state, terminal_state)

  def select(self, state):
      # Generate an action for a given state using epsilon-greedy policy
      if np.random.rand() < self.exploration_rate:
          return random.randrange(self.number_of_actions)
      else:
          state = state.reshape(1, self.number_of_observations).to(self.device)
          self.model.eval()
          with torch.no_grad():
            q_values = self.model(state)
          self.model.train()
          return q_values.argmax(dim=1).item()

  def select_greedy_policy(self, state):
      # Generate an action for a given state using greedy policy
      state = state.reshape(1, self.number_of_observations).to(self.device)
      self.model.eval()
      with torch.no_grad():
        q_values = self.model(state)
      self.model.train()
      return q_values.argmax(dim=1).item()

  def learn(self):
      # Learn the value Q using a sample of examples from the replay memory
      if self.memory.pos < self.batch_size: return

      states, actions, rewards, next_states, terminal_states = self.memory.sample_memory(self.batch_size)
      # Negation of the boolean array terminal_states, True if state not terminated
      not_terminal_states = ~terminal_states

      self.model.eval()
      self.target_model.eval()

      with autocast(device_type=self.device.type):
        q_pred = self.model(states)

        with torch.no_grad():
          q_targets = self.model(states)
          q_next_states = self.target_model(next_states)

        self.model.train()
        self.target_model.train()

        # Apply the Bellman equation to update target Q values for terminal and not_terminal states
        q_targets[terminal_states, actions[terminal_states]] = rewards[terminal_states].type(q_targets.dtype) 
        q_targets[not_terminal_states, actions[not_terminal_states]] = rewards[not_terminal_states].type(q_targets.dtype) + self.gamma * torch.max(q_next_states[not_terminal_states], dim=1).values

        # Compute the training       
        loss = self.criterion(q_pred, q_targets)

      self.optimizer.zero_grad()
      self.scaler.scale(loss).backward()
      self.scaler.step(self.optimizer)
      self.scaler.update()

      # Decrease exploration rate
      #print("exploration_rate BEFORE: ", self.exploration_rate)
      self.exploration_rate *= self.exploration_decay
      #print("exploration_rate AFTER: ", self.exploration_rate)
      self.exploration_rate = max(self.exploration_min, self.exploration_rate)

  def train(self, environment, num_episodes_training, num_episodes_update_target, verbose=False):
    self.model.train()
    total_steps = 0
    start_time = time.perf_counter()

    #print("\nTraining start...")
    for episode in range(num_episodes_training):
      score = 0
      episode_length = 0
        
      state, _ = environment.reset(seed=self.seed)
      # Transform the first state from numpy array to tensor from Pytorch
      state = torch.tensor(state, dtype=torch.float32, device=self.device)

      end_episode = False
      while not(end_episode):
        total_steps += 1
        episode_length += 1
        # Select an action for the current state
        action = self.select(state)

        # Execute the action on the environment
        state_next, reward, terminal_state, truncated, _ = environment.step(action)
        # Transfrom state_next, reward, terminal_state to tensor 
        state_next = torch.tensor(state_next, dtype=torch.float32, device=self.device)
        terminal_state = torch.tensor(terminal_state, dtype=torch.bool, device=self.device)

        # Store in memory the transition (s,a,r,s')
        self.remember(state, action, reward, state_next, terminal_state)

        score += reward

        # Learn using a batch of experience stored in memory
        self.learn()

        # Detect end of episode
        if terminal_state or truncated:
            self.add_score(score, score/episode_length, episode_length, self.exploration_rate)
            #print("Episode {0:>3}: ".format(episode), end = '')
            #print("score {0:>3} ".format(math.trunc(score)), end = '')
            #print("(exploration rate: %.2f, " % self.exploration_rate, end = '')
            #print("transitions: " + str(self.memory.pos) + ")")

            self.writer.add_scalar("metrics/exploration_rate", self.exploration_rate, episode)
            self.writer.add_scalar("charts/episodic_return", score, episode)
            self.writer.add_scalar("charts/mean_episodic_return", score/episode_length, episode)
            self.writer.add_scalar("charts/episodic_length", episode_length, episode)
            end_episode = True
        else:
            state = state_next

        # Update the target network every "x" steps
        if total_steps % num_episodes_update_target == 0:
            self.update_target_network()

    #print("\nTime for training:", round((time.perf_counter() - start_time)/60), "minutes")
    #print("Score (max):", max(self.episodic_return))

    #average_score = np.mean(self.episodic_return[max(0,(len(self.episodic_return)-10)):(len(self.episodic_return))])
    #print("Score (average last 10 episodes):", average_score)

    self.writer.close()

  def test(self, environment, num_episodes_testing=1):
    self.model.eval()
    start_time = time.perf_counter()
    self.delete_scores()
    
    for episode in range(num_episodes_testing):
        score = 0
        episode_length = 0

        state, _ = environment.reset(seed=self.seed)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        end_episode = False
        while not(end_episode):
            episode_length += 1
            # Select an action for the current state
            action = self.select_greedy_policy(state)

            # Execute the action in the environment
            state_next, reward, terminal_state, truncated, info = environment.step(action)
            state_next = torch.tensor(state_next, dtype=torch.float32, device=self.device)
            terminal_state = torch.tensor(terminal_state, dtype=torch.bool, device=self.device)

            score += reward

            # Detect end of episode and print
            if terminal_state or truncated:
                self.add_score(score, score/episode_length, episode_length, self.exploration_rate)
                print("Episode {0:>3}: ".format(episode), end = '')
                print("score {0:>3} \n".format(math.trunc(score)), end = '')
                end_episode = True
            else:
                state = state_next

    print("Time for testing:", round((time.perf_counter() - start_time)/60), "minutes")
    print("Score (average):", np.mean(self.episode_return))
    print("Score (max):", max(self.episode_return))

  def print_summay(self):
    print("\nModel summary:")
    print(summary(self.model, input_size=(self.batch_size, self.number_of_observations)))
    print("\nTarget model summary:")
    print(summary(self.target_model, input_size=(self.batch_size, self.number_of_observations)))
    print("\nMemory summary:")
    print(f"Estimated Total Size (MB): {self.memory.get_memory_usage()/1024**2:.2f}\n")


  
  def save_agent(self):
    with open('/content/drive/My Drive/04_Proyectos/agent.plk', 'wb') as f:
      pickle.dump(self, f)

  @classmethod
  def load_agent(cls):
    with open('/content/drive/My Drive/04_Proyectos/agent.plk', 'rb') as f:
        return pickle.load(f)

  def add_score(self, episode_return, mean_episode_return, episode_length, exploration_rate):
    # Add the obtained score to a list to be presented later
    self.episode_return.append(episode_return)
    self.mean_episode_return.append(mean_episode_return)
    self.episode_length.append(episode_length)
    self.exploration_rate_episode.append(exploration_rate)

  def delete_scores(self):
    # Delete the scores
    self.episode_return = []
    self.mean_episode_return = []
    self.episode_length = []
    self.exploration_rate_episode = []

  def display_scores_graphically(self):
    # Display the obtained scores graphically
    plt.plot(self.episode_return)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")