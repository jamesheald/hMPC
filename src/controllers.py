from jax import random, lax, vmap
import jax.numpy as np
from flax import linen as nn

class MPPI:

    def __init__(self, env, args):

        self.horizon = args.horizon
        self.n_actions = env.action_space.shape[0]
        self.actions_lower_bounds = env.action_space.low
        self.actions_upper_bounds = env.action_space.high
        self.n_sequences = args.n_sequences
        self.reward_weighting_factor = args.reward_weighting_factor
        self.noise_std_MPPI = args.noise_std_MPPI

    def sample_candidate_action_sequences(self, action_sequence_mean, key):

        # use the same mean for each sequence
        action_sequence_mean = np.repeat(action_sequence_mean[:, :, None], self.n_sequences, axis = 2)

        # # sample noise from a truncated normal distribution
        # lower = (self.actions_lower_bounds[None, :, None] - action_sequence_mean) / self.noise_std
        # upper = (self.actions_upper_bounds[None, :, None] - action_sequence_mean) / self.noise_std
        # eps = random.truncated_normal(key, lower = lower, upper = upper, shape = (self.horizon, self.n_actions, self.n_sequences)) * self.noise_std

        # sample noise from a normal distribution
        eps = random.normal(key, shape = (self.horizon, self.n_actions, self.n_sequences)) * self.noise_std_MPPI

        # add noise to action sequence mean
        action_sequences = action_sequence_mean + eps

        # clip actions
        action_sequences = np.clip(action_sequences, a_min = self.actions_lower_bounds[None, :, None], a_max = self.actions_upper_bounds[None, :, None])

        return action_sequences

    def estimate_cumulative_reward(self, state, observation, action_sequences):

        _, _, estimated_cumulative_reward = state.apply_fn(state.params, observation, action_sequences)

        return estimated_cumulative_reward

    batch_estimate_cumulative_reward = vmap(estimate_cumulative_reward, in_axes = (None, None, None, 2))

    def update_action_sequence_mean(self, estimated_cumulative_reward, action_sequences):

        # assign a weight to each action sequence based on its cumulative reward
        weights = nn.activation.softmax(estimated_cumulative_reward * self.reward_weighting_factor)

        # update the mean of the action sequence distribution
        action_sequence_mean = np.sum(action_sequences * weights[None, None, :], 2)

        # the mean of the optimised action sequence distribution (at the current time step) is the action to take in the environment
        action = np.copy(action_sequence_mean[0, :])

        # warmstarting (reuse the plan from the previous planning step)
        # amortise the optimisation process across multiple planning steps
        # shift the mean of the action sequence distribution by one time point
        # this implicitly copies/repeats the last time point (alternatively it could be set to 0)
        action_sequence_mean = action_sequence_mean.at[:-1, :].set(action_sequence_mean[1:, :])

        return action, action_sequence_mean

    def get_action(self, action_sequence_mean, observation, state, key):

        action_sequences = self.sample_candidate_action_sequences(action_sequence_mean, key)

        estimated_cumulative_reward = self.batch_estimate_cumulative_reward(state, observation, action_sequences)

        action, action_sequence_mean = self.update_action_sequence_mean(estimated_cumulative_reward, action_sequences)

        return action, action_sequence_mean

class CEM:

    def __init__(self, env, args):

        self.horizon = args.horizon
        self.n_actions = env.action_space.shape[0]
        self.actions_lower_bounds = env.action_space.low
        self.actions_upper_bounds = env.action_space.high
        self.n_sequences = args.n_sequences
        self.n_elite = args.n_elite
        self.CEM_iterations = args.CEM_iterations
        self.noise_std_CEM = args.noise_std_CEM

    def sample_candidate_action_sequences(self, action_sequence_mean, action_sequence_variance, key):

        # use the same mean for each sequence
        action_sequence_mean = np.repeat(action_sequence_mean[:, :, None], self.n_sequences, axis = 2)

        # sample noise from a normal distribution
        eps = random.normal(key, shape = (self.horizon, self.n_actions, self.n_sequences)) * np.sqrt(action_sequence_variance[:, :, None])

        # add noise to action sequence mean
        action_sequences = action_sequence_mean + eps

        # clip actions
        action_sequences = np.clip(action_sequences, a_min = self.actions_lower_bounds[None, :, None], a_max = self.actions_upper_bounds[None, :, None])

        return action_sequences

    def estimate_cumulative_reward(self, state, observation, action_sequences):

        _, _, estimated_cumulative_reward = state.apply_fn(state.params, observation, action_sequences)

        return estimated_cumulative_reward

    batch_estimate_cumulative_reward = vmap(estimate_cumulative_reward, in_axes = (None, None, None, 2))

    def update_action_sequence_mean(self, estimated_cumulative_reward, action_sequences):

        # find the elite action sequences (i.e. those associated with the highest reward)
        elite_indices = np.argsort(estimated_cumulative_reward)[-self.n_elite:]
        
        # use the actions of the elite action sequences to update the mean and variance of the action sequence distribution
        action_sequence_mean = np.mean(action_sequences[:, :, elite_indices], 2)
        action_sequence_variance = np.var(action_sequences[:, :, elite_indices], 2)

        return action_sequence_mean, action_sequence_variance

    def CEM_iteration(self, carry, key):

        action_sequence_mean, action_sequence_variance, observation, state = carry

        action_sequences = self.sample_candidate_action_sequences(action_sequence_mean, action_sequence_variance, key)

        estimated_cumulative_reward = self.batch_estimate_cumulative_reward(state, observation, action_sequences)

        action_sequence_mean, action_sequence_variance = self.update_action_sequence_mean(estimated_cumulative_reward, action_sequences)

        carry = action_sequence_mean, action_sequence_variance, observation, state

        return carry, None

    def get_action(self, action_sequence_mean, observation, state, key):

        action_sequence_variance = np.ones((self.horizon, self.n_actions)) * self.noise_std_CEM ** 2
        carry = action_sequence_mean, action_sequence_variance, observation, state

        subkeys = random.split(key, self.CEM_iterations)
        (action_sequence_mean, action_sequence_variance, _, _), _ = lax.scan(self.CEM_iteration, carry, subkeys)

        action = np.copy(action_sequence_mean[0, :])

        # warmstarting (reuse the plan from the previous planning step)
        # amortise the optimisation process across multiple planning steps
        # shift the mean of the action sequence distribution by one time point
        # this implicitly copies/repeats the last time point (alternatively it could be set to 0)
        action_sequence_mean = action_sequence_mean.at[:-1, :].set(action_sequence_mean[1:, :])

        return action, action_sequence_mean