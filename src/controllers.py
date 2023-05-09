from jax import vmap, random, lax
import jax.numpy as np
from functools import partial
from reward import reward_function
from flax import linen as nn

class MPPI:

    def __init__(self, env, args):

        self.horizon = args.horizon
        self.n_actions = env.action_size
        self.n_sequences = args.n_sequences
        self.reward_weighting_factor = args.reward_weighting_factor
        self.noise_std = args.noise_std

    def ground_truth_dynamics_one_step_prediction(self, env, state, carry, action):

        observation, env_state = carry

        env_state = env.step(env_state, action)
        
        # reward = env_state.reward
        reward = reward_function([], action, env_state.obs)

        carry = observation, env_state

        return carry, reward

    def learned_dynamics_one_step_prediction(self, env, state, carry, action):

        observation, env_state = carry

        # concatenate the current observation and action to form the input to the dynamics model
        inputs = np.concatenate((observation, action))

        # predict the next observation using the learned dynamics model
        next_observation = state.apply_fn(state.params, inputs)

        # calculate the reward based on the current observation, the action and the next observation
        reward = reward_function(observation, action, next_observation)

        next_observation, env_state = carry

        return carry, reward

    def learned_dynamics_one_step_prediction_GRU(self, env, state, carry, action):

        observation, env_state = carry

        # concatenate the current observation and action to form the input to the dynamics model
        inputs = np.concatenate((observation, action))

        # predict the next observation using the learned dynamics model
        next_observation = state.apply_fn(state.params, inputs)

        # calculate the reward based on the current observation, the action and the next observation
        reward = reward_function(observation, action, next_observation)

        next_observation, env_state = carry

        return carry, reward

    def estimate_return(self, predict, env_state, action_sequence):

        carry = env_state.obs, env_state

        # use the learned model to predict the observation sequence and reward associated with each action sequence
        _, reward = lax.scan(predict, carry, action_sequence)

        total_reward = reward.sum()

        return total_reward

    batch_estimate_return = vmap(estimate_return, in_axes = (None, None, None, 2))

    def update_action_sequence(self, predict, env_state, action_sequence_mean, key):

        # # sample noise from a normal distribution
        # eps = random.normal(key, (self.horizon, self.n_actions, self.n_sequences)) * self.noise_std

        # # add the sampled noise to the mean of the action sequence distribution
        # action_sequences = action_sequence_mean[:, :, None] + eps

        # # clip the actions
        # action_sequences = np.clip(action_sequences, -1.0, 1.0)

        # use the same mean for each sequence
        action_sequence_mean = np.repeat(action_sequence_mean[:, :, None], self.n_sequences, axis = 2)

        # sample noise from a truncated normal distribution
        lower = (-1 - action_sequence_mean) / self.noise_std
        upper = (1 - action_sequence_mean) / self.noise_std
        eps = random.truncated_normal(key, lower = lower, upper = upper, shape = (self.horizon, self.n_actions, self.n_sequences)) * self.noise_std

        action_sequences = action_sequence_mean + eps

        # use the learned model to estimate the cumulative reward associated with each action sequence
        total_reward = self.batch_estimate_return(predict, env_state, action_sequences)

        # assign a weight to each action sequence based on its cumulative reward
        weights = nn.activation.softmax(total_reward * self.reward_weighting_factor)

        # update the mean of the action sequence distribution
        action_sequence_mean = np.sum(action_sequences * weights[None, None, :], 2)

        return action_sequence_mean

    def get_action(self, env, env_state, state, action_sequence_mean, key):

        # # find the elite action sequences (i.e. those associated with the highest reward)
        # elite_indices = np.argsort(total_reward)[-self.n_elite:]
        
        # # use the actions of the elite action sequences to update the mean and variance of the action sequence distribution
        # action_mean = self.alpha * np.mean(action_sequences[:, :, elite_indices], 2) + (1 - self.alpha) * action_mean
        # action_variance = self.alpha * np.var(action_sequences[:, :, elite_indices], 2) + (1 - self.alpha) * action_variance

        # if np.max(action_variance) < self.min_variance:

        #     break

        predict = partial(self.ground_truth_dynamics_one_step_prediction, env, state)
        # predict = partial(self.learned_dynamics_one_step_prediction, env, state)
        # predict = partial(self.learned_dynamics_one_step_prediction_GRU, env, state)

        # update the mean of the action sequence distribution
        action_sequence_mean = self.update_action_sequence(predict, env_state, action_sequence_mean, key)

        # the mean of the optimised action sequence distribution (at the current time step) is the action to take in the environment
        action = np.copy(action_sequence_mean[0, :])

        # warmstarting (reuse the plan from the previous planning step)
        # amortise the optimisation process across multiple planning steps
        # shift the mean of the action sequence distribution by one time point
        # this implicitly copies/repeats the last time point (alternatively it could be set to 0)
        action_sequence_mean = action_sequence_mean.at[:-1, :].set(action_sequence_mean[1:, :])

        return action, action_sequence_mean