import torch


class ReplayMemory:
    def __init__(self, number_of_observations, memory_size, device):
      self.number_of_observations = number_of_observations
      self.memory_size = memory_size
      self.device = device
      self.pos = 0

      self.states = torch.zeros((memory_size, number_of_observations), dtype=torch.float32, device=device)
      self.states_next = torch.zeros((memory_size, number_of_observations), dtype=torch.float32, device=device)
      self.actions = torch.zeros(memory_size, dtype=torch.int32, device=device)
      self.rewards = torch.zeros(memory_size, dtype=torch.float32, device=device)
      self.terminal_states = torch.zeros(memory_size, dtype=torch.bool, device=device)

    def store_transition(self, state, action, reward, state_next, terminal_state):
      idx = self.pos

      self.states[idx] = state
      self.actions[idx] = action
      self.rewards[idx] = reward
      self.states_next[idx] = state_next
      self.terminal_states[idx] = terminal_state

      self.pos = (self.pos + 1) % self.memory_size

    def sample_memory(self, batch_size):
      batch = torch.randint(0, self.pos, (batch_size,), device=self.device)

      states = self.states[batch]
      actions = self.actions[batch]
      rewards = self.rewards[batch]
      states_next = self.states_next[batch]
      terminal_states = self.terminal_states[batch]

      return states, actions, rewards, states_next, terminal_states

    def get_memory_usage(self):
      total_bytes = 0

      tensors_to_check = {
            "states": self.states,
            "states_next": self.states_next,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminal_states": self.terminal_states
      }

      for _, tensor in tensors_to_check.items():
          tensor_bytes = tensor.element_size() * tensor.nelement()
          total_bytes += tensor_bytes

      return total_bytes