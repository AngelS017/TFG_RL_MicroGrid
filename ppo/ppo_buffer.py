import torch

class PPORolloutBuffer:
  def __init__(self, num_steps, num_envs, observation_space, action_space, device):
    self.num_steps = num_steps
    self.num_envs = num_envs
    self.observation_space = observation_space
    self.action_space = action_space
    self.device = device
    self.pos = 0

    self.observations = torch.zeros((num_steps, num_envs) + observation_space.shape, device=device)
    self.actions = torch.zeros((num_steps, num_envs) + action_space.shape, device=device)
    self.log_probs = torch.zeros((num_steps, num_envs), device=device)
    self.rewards = torch.zeros((num_steps, num_envs), device=device)
    self.dones = torch.zeros((num_steps, num_envs), device=device)
    self.values = torch.zeros((num_steps, num_envs), device=device)

  def store(self, observation, action, log_prob, reward, done, value):
    idx = self.pos

    self.observations[idx] = observation
    self.actions[idx] = action
    self.log_probs[idx] = log_prob
    self.rewards[idx] = torch.tensor(reward, device=self.device)
    self.dones[idx] = done
    self.values[idx] = value

    self.pos = (self.pos + 1) % self.num_steps

  def plot_buffer(self):
    torch.print("Observations:\n", self.observations)
    torch.print("Actions:\n", self.actions)
    torch.print("Log Probs:\n", self.log_probs)
    torch.print("Rewards:\n", self.rewards)
    torch.print("Dones:\n", self.dones)
    torch.print("Values:\n", self.values)