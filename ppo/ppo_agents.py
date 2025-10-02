import numpy as np

import torch
import torch.nn as nn
from torchinfo import summary
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


class DiscreteAgent(nn.Module):
  def __init__(self, envs, hidden_layers_critic, hidden_layers_actor, device, summary_batch_size):
    super(DiscreteAgent, self).__init__()
    # Get the imput and output dimensions for the critic and actor neural networks
    self.input_dim = int(np.array(envs.single_observation_space.shape).prod())
    self.action_dim = envs.single_action_space.n

    self.hidden_layers_critic = hidden_layers_critic
    self.hidden_layers_actor = hidden_layers_actor
    self.device = device

    self.summary_batch_size = summary_batch_size

    self.critic = self.create_critic_model().to(device)

    self.actor = self.create_actor_model().to(device)


  def create_critic_model(self):
    layers = []

    # Input layer
    layer = nn.Linear(self.input_dim, self.hidden_layers_critic[0])
    # Initialize weights and bias
    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    nn.init.constant_(layer.bias, val=0.0)
    layers.append(layer)
    layers.append(nn.Tanh())

    # Hidden layers
    for in_size, out_size in zip(self.hidden_layers_critic, self.hidden_layers_critic[1:]):
      layer = nn.Linear(in_size, out_size)
      # Initialize weights and bias
      nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
      nn.init.constant_(layer.bias, val=0.0)
      layers.append(layer)
      layers.append(nn.Tanh())

    # Outpur layer
    layer = nn.Linear(self.hidden_layers_critic[-1], 1)
    # Initialize weights and bias
    nn.init.orthogonal_(layer.weight, gain=1.0)
    nn.init.constant_(layer.bias, val=0.0)
    layers.append(layer)

    critic_model = nn.Sequential(*layers)

    return critic_model

  def create_actor_model(self):
    layers = []

    # Input layer
    layer = nn.Linear(self.input_dim, self.hidden_layers_actor[0])
    # Initialize weights and bias
    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    nn.init.constant_(layer.bias, val=0.0)
    layers.append(layer)
    layers.append(nn.Tanh())

    # Hidden layers
    for in_size, out_size in zip(self.hidden_layers_actor, self.hidden_layers_actor[1:]):
      layer = nn.Linear(in_size, out_size)
      # Initialize weights and bias
      nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
      nn.init.constant_(layer.bias, val=0.0)
      layers.append(layer)
      layers.append(nn.Tanh())

    # Outpur layer
    layer = nn.Linear(self.hidden_layers_actor[-1], self.action_dim)
    # Initialize weights and bias
    nn.init.orthogonal_(layer.weight, gain=0.01)
    nn.init.constant_(layer.bias, val=0.0)
    layers.append(layer)

    actor_model = nn.Sequential(*layers)

    return actor_model

  def get_value(self, x):
    return self.critic(x)

  def get_action_normal_distribution(self, x):
    # Get the mean for the Normal distribution of the action
    logits = self.actor(x)
    # Get the Categorical distribution of the action
    return Categorical(logits=logits)

  def sample_action_and_probability(self, x, action=None):
    action_dist = self.get_action_normal_distribution(x)
    # Sample the action and their distribution
    if action is None:
      action = action_dist.sample()
    log_prob = action_dist.log_prob(action)

    return action, log_prob

  def get_entropy(self, x):
    action_dist = self.get_action_normal_distribution(x)
    return action_dist.entropy()

  def print_summary(self):
    print("\033[1m" + "Critic model summary:" + "\033[0;0m")
    print(summary(self.critic, input_size=(self.summary_batch_size, self.input_dim)))
    print("\033[1m" + "\nActor(mean) model summary:" + "\033[0;0m")
    print(summary(self.actor, input_size=(self.summary_batch_size, self.input_dim)))

  def predict_test(self, x):
    action_dist = self.get_action_normal_distribution(x)
    action = action_dist.sample()
    return action



class ContinuousAgent(nn.Module):
    def __init__(self, envs, hidden_layers_critic, hidden_layers_actor, device, summary_batch_size):
      super(ContinuousAgent, self).__init__()
      # Get the imput and output dimensions for the critic and actor neural networks
      self.input_dim = int(np.array(envs.single_observation_space.shape).prod())
      self.action_dim = int(np.prod(envs.single_action_space.shape))

      self.hidden_layers_critic = hidden_layers_critic
      self.hidden_layers_actor = hidden_layers_actor
      self.device = device

      self.summary_batch_size = summary_batch_size

      self.critic = self.create_critic_model().to(device)

      self.actor_mean = self.create_actor_model().to(device)
      self.actor_log_std = nn.Parameter(torch.zeros(1, self.action_dim)).to(device)


    def create_critic_model(self):
      layers = []

      # Input layer
      layer = nn.Linear(self.input_dim, self.hidden_layers_critic[0])
      # Initialize weights and bias
      nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
      nn.init.constant_(layer.bias, val=0.0)
      layers.append(layer)
      layers.append(nn.Tanh())

      # Hidden layers
      for in_size, out_size in zip(self.hidden_layers_critic, self.hidden_layers_critic[1:]):
        layer = nn.Linear(in_size, out_size)
        # Initialize weights and bias
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        nn.init.constant_(layer.bias, val=0.0)
        layers.append(layer)
        layers.append(nn.Tanh())

      # Outpur layer
      layer = nn.Linear(self.hidden_layers_critic[-1], 1)
      # Initialize weights and bias
      nn.init.orthogonal_(layer.weight, gain=1.0)
      nn.init.constant_(layer.bias, val=0.0)
      layers.append(layer)

      critic_model = nn.Sequential(*layers)

      return critic_model

    def create_actor_model(self):
      layers = []

      # Input layer
      layer = nn.Linear(self.input_dim, self.hidden_layers_actor[0])
      # Initialize weights and bias
      nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
      nn.init.constant_(layer.bias, val=0.0)
      layers.append(layer)
      layers.append(nn.Tanh())

      # Hidden layers
      for in_size, out_size in zip(self.hidden_layers_actor, self.hidden_layers_actor[1:]):
        layer = nn.Linear(in_size, out_size)
        # Initialize weights and bias
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        nn.init.constant_(layer.bias, val=0.0)
        layers.append(layer)
        layers.append(nn.Tanh())

      # Outpur layer
      layer = nn.Linear(self.hidden_layers_actor[-1], self.action_dim)
      # Initialize weights and bias
      nn.init.orthogonal_(layer.weight, gain=0.01)
      nn.init.constant_(layer.bias, val=0.0)
      layers.append(layer)

      actor_model = nn.Sequential(*layers)

      return actor_model

    def get_value(self, x):
      return self.critic(x)

    def get_action_normal_distribution(self, x):
      # Get the mean for the Normal distribution of the action
      action_mean = self.actor_mean(x)
      # Get the standard deviation for the Normal distribution of the action
      action_log_std = self.actor_log_std.expand_as(action_mean)
      action_std = torch.exp(action_log_std)

      action_dist = Normal(loc=action_mean, scale=action_std)
      return action_dist

    def sample_action_and_probability(self, x, action=None):
      action_dist = self.get_action_normal_distribution(x)
      # Sample the action and their distribution
      if action is None:
        action = action_dist.sample()
      log_prob = action_dist.log_prob(action).sum(axis=1)

      return action, log_prob

    def get_entropy(self, x):
      action_dist = self.get_action_normal_distribution(x)
      return action_dist.entropy().sum(axis=1)

    def print_summary(self):
      print("\033[1m" + "Critic model summary:" + "\033[0;0m")
      print(summary(self.critic, input_size=(self.summary_batch_size, self.input_dim)))
      print("\033[1m" + "\nActor(mean) model summary:" + "\033[0;0m")
      print(summary(self.actor_mean, input_size=(self.summary_batch_size, self.input_dim)))
      print("\033[1m" + "\nSummary of Actor (log std) model:" + "\033[0;0m"),
      print(f"Shape: {self.actor_log_std.shape}. Total params: {self.actor_log_std.numel()} ({self.actor_log_std.detach().cpu().numpy().nbytes / 1024:.2f} KB)")

    def predict_test(self, x):
      action_dist = self.get_action_normal_distribution(x)
      action = action_dist.sample()
      return action

