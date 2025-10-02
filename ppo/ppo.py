import gymnasium as gym

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.amp import autocast

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

from .ppo_buffer import PPORolloutBuffer


def make_env_continuous(base_env, seed, gamma):
  def thunk():
    env = base_env()
    env = gym.wrappers.RecordEpisodeStatistics(env)
    #env = gym.wrappers.ClipAction(env)
    #env = gym.wrappers.NormalizeObservation(env)
    #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
    #env = gym.wrappers.NormalizeReward(env, gamma=gamma, epsilon=1e-8)
    #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    # Set the seed properly in Gymnasium
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env

  return thunk

def make_env_discrete(base_env, seed, max_episode_steps):
  def thunk():
    env = base_env()
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    
    # Set the seed properly in Gymnasium
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env

  return thunk



class PPO:
  def __init__(
      self,
      run_name,
      agent,
      vectorized_environments,
      learning_rate = 3e-4,
      max_grad_norm = 0.5,
      num_rollout_steps = 2048,
      num_epochs = 10,
      minibatch_size = 32,
      gamma = 0.99,
      gae_lambda = 0.95,
      normalize_advantages = True,
      clip_threshold = 0.2,
      entropy_loss_coefficient = 0.0,
      value_loss_coefficient = 0.5,
      target_kl_earlystop = None,
      device = None,
      seed=1234,
      verbose=False
    ):

    self.agent = agent
    self.vectorized_environments = vectorized_environments
    self.seed = seed
    self.device = device
    self.verbose = verbose
    # Save optimizer and learning rate
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=learning_rate)
    # Save training params
    self.num_envs = vectorized_environments.num_envs

    self.num_rollout_steps = num_rollout_steps
    self.num_epochs = num_epochs
    #self.num_minibatch = num_minibatch

    self.batch_size = self.num_rollout_steps * self.num_envs
    #self.minibatch_size = self.batch_size // self.num_minibatch
    self.minibatch_size = minibatch_size
    # Save PPO params
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.normalize_advantages = normalize_advantages
    self.clip_threshold = clip_threshold
    self.entropy_loss_coefficient = entropy_loss_coefficient
    self.value_loss_coefficient = value_loss_coefficient
    self.target_kl_earlystop = target_kl_earlystop

    # Initializete PPO Buffer Rollout
    self.ppo_buffer = PPORolloutBuffer(num_rollout_steps, self.num_envs, vectorized_environments.observation_space, 
                                       vectorized_environments.action_space, device)

    self.global_timestep = 0
    """
    self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    self.finished_episodes_stats = []
    """

    # Initialize Tensorboard
    self.log_dir = os.path.join("logs_tensorflow", "ppo", run_name)
    self.writer = SummaryWriter(self.log_dir)

    if device.type == "cuda":
      self.scaler = torch.amp.GradScaler("cuda")
    else:
      self.scaler = torch.amp.GradScaler("cpu")

  def compute_advantages_gae(self, rewards, values, dones, next_observation_done, next_value):
    #print(rewards.shape, values.shape, dones.shape, next_observation_done.shape, next_value.shape)
    advantages = advantages = torch.zeros_like(rewards, device=self.device)
    lastgaelam = 0.0

    # The revers for it is use to make easier to compute the advantages, due to to compute the A_{t} you needed
    # the next advantage A_{t+1}.
    # It is easier to do it backwards, because for the computation of the A_{t} you already compute de A_{t+1} previusly (it was the firts one computed)
    for t in reversed(range(self.num_rollout_steps)):
      if t == self.num_rollout_steps - 1:
          nextnonterminal = 1.0 - next_observation_done
          nextvalues = next_value
      else:
          nextnonterminal = 1.0 - dones[t + 1]
          nextvalues = values[t + 1]

      # Compute: TD error --> δt
      # delta = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
      delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
      # Compute: Generalized Advantage Estimation --> At​
      # A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
      lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
      advantages[t] = lastgaelam

    # Finally the advantage using GAE is
    returns = advantages + values

    return advantages, returns

  def compute_advantages_gae_optim(self, rewards, values, dones, next_observation_done, next_value):
    dones_t_1 = torch.cat([dones, next_observation_done.unsqueeze(0)], dim=0)
    values_t_1 = torch.cat([values, next_value[0].unsqueeze(0)], dim=0)

    nextnonterminal = 1.0 - dones_t_1[1:]
    nextvalues = values_t_1[1:]

    delta = rewards + 0.99 * nextvalues * nextnonterminal - values

    advantages = torch.zeros_like(rewards)
    advantages[-1] = delta[-1]

    for i in reversed(range(self.num_rollout_steps-1)):
      advantages[i] = delta[i] + 0.99 * 0.95 * nextnonterminal[i] * advantages[i+1]

    returns = advantages + values

    return advantages, returns

  def rollout_data(self, next_observation, next_observation_done):
    for rollout_step in range(self.num_rollout_steps):
      # Get the action, probability and value of the next_observation
      with torch.no_grad():
        action, log_prob = self.agent.sample_action_and_probability(next_observation)
        value = self.agent.get_value(next_observation)

      # Compute the action into the vectorized_environments to get the next observation, reward, etc
      next_obs, reward, terminated, truncated, info = self.vectorized_environments.step(action.cpu().numpy())

      self.global_timestep += self.num_envs

      done = np.logical_or(terminated, truncated)

      # Save the observation, action, log_prob, reward, done and value into the rollout buffer
      self.ppo_buffer.store(next_observation, action, log_prob, reward, next_observation_done, value.reshape(-1,))

      # Get the next observation and check if its terminated
      next_observation = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
      next_observation_done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

      """
      # Save the reward and the episode length
      self.episode_returns += reward
      self.episode_lengths += 1
      
      if num_dones:
        finished_episode_rewards = np.where(done, self.episode_returns, 0.0)
        finished_episode_lengths = np.where(done, self.episode_lengths, 0)

        self.finished_episodes_stats.append({
                   'rewards': finished_episode_rewards,
                   'lenghts': finished_episode_lengths,
                   'global_step': self.global_timestep
        })

        print(f"Global Step {self.global_timestep} --> Episodic Return mean(reward): {np.mean(finished_episode_rewards)}")

        self.writer.add_scalar("charts/episodic_return", np.mean(finished_episode_rewards), self.global_timestep)
        self.writer.add_scalar("charts/episodic_length", np.mean(finished_episode_lengths), self.global_timestep)

        self.episode_returns[done] = 0
        self.episode_lengths[done] = 0
      """
      if "episode" in info.keys():
        #print(f"Global Step {self.global_timestep} --> Episodic Return mean(reward): {info['episode']['r'].mean()}")
        #print(info)
        #self.writer.add_scalar("charts/total_episodic_return", info["episode"]["r"], self.global_timestep)
        self.writer.add_scalar("charts/mean_episodic_return", info["episode"]["r"].mean(), self.global_timestep)
        self.writer.add_scalar("charts/episodic_length", info["episode"]["l"].mean(), self.global_timestep)
      
    with torch.no_grad():
      # Get the value of the next observation
      next_value = self.agent.get_value(next_observation).reshape(1, -1)

      # Compute the advantages using using Generalized Advantage Estimation (GAE)
      advantages_gae, returns_gae = self.compute_advantages_gae(self.ppo_buffer.rewards, self.ppo_buffer.values, self.ppo_buffer.dones, next_observation_done, next_value)

    # Flatten the data from this batch for later use in update_policy
    batch_observations = self.ppo_buffer.observations.reshape((-1,) + self.vectorized_environments.single_observation_space.shape)
    batch_log_probs = self.ppo_buffer.log_probs.reshape(-1,)
    batch_actions = self.ppo_buffer.actions.reshape((-1,) + self.vectorized_environments.single_action_space.shape)
    batch_advantages = advantages_gae.reshape(-1,)
    batch_returns = returns_gae.reshape(-1,)
    batch_values = self.ppo_buffer.values.reshape(-1,)

    return batch_observations, batch_actions, batch_log_probs, batch_advantages, batch_returns, batch_values


  def calculate_policy_gradient_loss(self, minibatch_advantages, ratio):
    # L^PG(θ) = r_t(θ) * A_t
    unclipped_part = -minibatch_advantages * ratio

    # L^CLIP(θ) = clip(r_t(θ), 1-ε, 1+ε) * A_t
    clipped_part = -minibatch_advantages * torch.clamp(ratio, 1.0 - self.clip_threshold, 1.0 + self.clip_threshold)

    # L^CLIP(θ) = -min(L^PG(θ), L^CLIP(θ))
    # We need to set the loss negative due to be able to compute Gradient Ascent to train the PPO algorithm
    # but as most of the gradient differenciation performs Gradient Descent (minimize the error/loss)
    # So Gradient Ascent == negative Gradient Descent
    policy_gradient_loss = torch.max(unclipped_part, clipped_part).mean()

    return policy_gradient_loss


  def calculate_value_loss(self, newvalues, minibatch_returns, minibatch_values):
    newvalues = newvalues.reshape(-1,)
    if self.clip_threshold:
      value_loss_unclipped = (newvalues - minibatch_returns) ** 2

      value_clipped = minibatch_values + torch.clamp(newvalues - minibatch_values, -self.clip_threshold, self.clip_threshold)
      value_loss_clipped = (value_clipped - minibatch_returns) ** 2

      value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)

      value_loss = 0.5 * (value_loss_max).mean()
    else:
      value_loss = 0.5 * ((newvalues - minibatch_returns) ** 2).mean()

    return value_loss

  def calculate_explained_variance(self, y_true, y_pred):
    var_y_true = torch.var(y_true, correction=0)

    return np.nan if var_y_true == 0 else 1 - torch.var(y_true - y_pred, correction=0) / var_y_true

  def update_policy(self, batch_observations, bacth_actions, batch_log_probs, batch_advantages, batch_returns, batch_values):
    batch_indices = np.arange(self.batch_size)

    for epoch in range(self.num_epochs):
      np.random.shuffle(batch_indices)

      for start in range(0, self.batch_size, self.minibatch_size):
        minibatch_indices = batch_indices[start : start + self.minibatch_size]
        # Obtain the observations, log_probs, actions, advantages, returns and values of the current mini batch
        minibatch_observations = batch_observations[minibatch_indices]
        minibatch_log_probs = batch_log_probs[minibatch_indices]
        minibatch_actions = bacth_actions[minibatch_indices]
        minibatch_advantages = batch_advantages[minibatch_indices]
        minibatch_returns = batch_returns[minibatch_indices]
        minibatch_values = batch_values[minibatch_indices]

        with autocast(device_type=self.device.type):
          # Get the new log_probs of the minibatch actions, the entropy and values of the minibatch observations
          _, new_log_probs = self.agent.sample_action_and_probability(minibatch_observations, minibatch_actions)
          entropy = self.agent.get_entropy(minibatch_observations)
          newvalues = self.agent.get_value(minibatch_observations)

          # Ratio between old and new policy
          # π_θ(at | st) / π_θold (at | st)
          log_ratio = new_log_probs - minibatch_log_probs
          ratio = log_ratio.exp()

          # Normalize advantages to rdeuce variance in the update of the policy
          if self.normalize_advantages:
            minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

          # Compute the clip loss
          # L^CLIP(θ) = -E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
          policy_gradient_loss = self.calculate_policy_gradient_loss(minibatch_advantages, ratio)
          # Compute the value loss
          # L^VF(θ) = E[(V_θ(s_t) - R_t)^2]
          value_loss = self.calculate_value_loss(newvalues, minibatch_returns, minibatch_values)
          # Compute entropy loss
          entropy_loss = entropy.mean()

          # Compute PPO loss
          # In the original PPO parer: L^CLIP+VF+S(θ) = L^CLIP(θ) - c1 * L^VF(θ) + c2 * entropy_loss
          # The idea is to minimize the policy_gradient_loss and the entropy_loss but maximize the value_loss
          # due to we need to apply Gradient Ascent
          ppo_loss = policy_gradient_loss + self.value_loss_coefficient * value_loss - self.entropy_loss_coefficient * entropy_loss

        # Performe the update/optimization of the policy
        self.optimizer.zero_grad()
        self.scaler.scale(ppo_loss).backward()
        # Its necesary to unscale first the the params of the optimizer due to we are going to perform some calculations o them
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

      # Compute KL-divergence
      with torch.no_grad():
        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()

      # Check if KL early stop applies
      if self.target_kl_earlystop is not None:
        if approx_kl > self.target_kl_earlystop:
          break

    # Calculate the explained varaince which measures how
    explained_variance = self.calculate_explained_variance(batch_returns, batch_values)

    # Record results for saving info
    update_results = {
        "ppo_loss": ppo_loss,
        "policy_loss": policy_gradient_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "learning_rate": self.learning_rate,
        "old_approx_kl": old_approx_kl,
        "approx_kl": approx_kl,
        "explained_variance": explained_variance,
    }

    return update_results


  def learn(self, total_timesteeps):
    start_time = time.perf_counter()

    self.total_timesteeps = total_timesteeps
    self.num_updates = self.total_timesteeps // self.batch_size

    # Initialize the firts observation and check if it is terminated
    next_observation, _ = self.vectorized_environments.reset(seed=self.seed)
    next_observation = torch.tensor(next_observation, dtype=torch.float32, device=self.device)
    next_observation_done = torch.zeros(self.num_envs, device=self.device)

    for update in range(self.num_updates):
      # Perform the rollout data step
      batch_observations, bacth_actions, batch_log_probs, batch_advantages, batch_returns, batch_values = self.rollout_data(next_observation, next_observation_done)

      # Update the policy (perform the trainig into the neural networks of the actor and critic)
      update_results = self.update_policy(batch_observations, bacth_actions, batch_log_probs, batch_advantages, batch_returns, batch_values)

      # Save all info into Tensorboard
      for key, value in update_results.items():
          self.writer.add_scalar("metrics/"+key, value, self.global_timestep)

      if self.verbose:
        print(
          f"Global Step {self.global_timestep} --> "
          f"PPO Loss: {update_results['ppo_loss']:.4f} | "
          f"Policy Loss: {update_results['policy_loss']:.4f} | "
          f"Value Loss: {update_results['value_loss']:.4f} | "
          f"Entropy: {update_results['entropy_loss']:.4f} | "
          f"KL: {update_results['approx_kl']:.4f} | "
          f"Explained Var: {update_results['explained_variance']:.3f}"
        )
    self.vectorized_environments.close()
    self.writer.close()

    #print(f"\nTraining completed. Total steps: {self.global_timestep}")
    #print("\nTime for training:", round((time.perf_counter() - start_time)/60), "minutes")


  def test(self, environment, num_episodes_testing=1):
    start_time = time.perf_counter()
    episode_returns = []
    mean_return = []
    
    for episode in range(num_episodes_testing):
        score = 0
        episode_length = 0

        # Initialize the firts observation and check if it is terminated
        state, _ = environment.reset(seed=self.seed)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        state_done = torch.zeros(self.num_envs, device=self.device)
        
        end_episode = False
        while not(end_episode):
            episode_length += 1
            # Select an action for the current state
            action = self.agent.predict_test(state)

            # Execute the action in the environment
            state_next, reward, terminal_state, truncated, info = environment.step(action.cpu().numpy())
            state_next = torch.tensor(state_next, dtype=torch.float32, device=self.device)
            terminal_state = torch.tensor(terminal_state, dtype=torch.bool, device=self.device)

            score += reward

            # Detect end of episode and print
            if terminal_state or truncated:
                #self.add_score(score, score/episode_length, episode_length, self.exploration_rate)
                episode_returns.append(score)
                mean_return.append(score/episode_length)
                #print("Episode {0:>3}: ".format(episode), end = '')
                #print("score {0:>3} \n".format(math.trunc(score)), end = '')
                end_episode = True
            else:
                state = state_next

    #print("Time for testing:", round((time.perf_counter() - start_time)/60), "minutes")
    #print("Score (average):", np.mean(episode_returns))
    #print("Score (max):", max(episode_returns))

    return episode_returns[0], mean_return[0]