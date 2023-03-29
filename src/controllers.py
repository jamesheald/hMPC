from jax import vmap, random, lax
import jax.numpy as np
from functools import partial
from reward import reward_function
from flax import linen as nn

class MPPI:

    def __init__(self, env, args):

        self.n_sequences = args.n_sequences
        self.n_actions = env.action_size
        self.horizon = args.horizon
        self.n_elite = args.n_elite
        self.alpha = args.alpha
        self.min_variance = args.min_variance
        self.reward_weighting_factor = args.reward_weighting_factor

    def predict_one_step(self, state, observation, action):

        # concatenate the current observation and action to form the input to the dynamics model
        inputs = np.concatenate((observation, action))

        # predict the next observation using the learned dynamics model
        next_observation = state.apply_fn(state.params, inputs)

        # calculate the reward based on the current observation and action (for some environments, the reward may be a function of the next observation also)
        reward = reward_function(observation, action)

        return next_observation, reward

    def estimate_return(self, predict, current_observation, action_sequence):

        # use the learned model to predict the observation sequence and reward associated with each action sequence
        _, reward = lax.scan(predict, current_observation, action_sequence)

        total_reward = reward.sum()

        return total_reward

    batch_estimate_return = vmap(estimate_return, in_axes = (None, None, None, 2))

    def update_action_sequence(self, predict, current_observation, action_sequence_mean, key):

        noise_variance = 0.1

        # sample noise to add to the mean of the action sequence distribution
        eps = noise_variance * random.normal(key, (self.horizon, self.n_actions, self.n_sequences))

        # add noise to the mean of the action sequence distribution
        action_sequences = np.expand_dims(action_sequence_mean, 2) + eps

        # clip actions
        action_sequences = np.clip(action_sequences, -1.0, 1.0)

        # use the learned model to estimate the cumulative reward associated with each action sequence
        total_reward = self.batch_estimate_return(predict, current_observation, action_sequences)

        # assign a weight to each action sequence based on its total reward
        weights = np.expand_dims(nn.activation.softmax(total_reward * self.reward_weighting_factor), (0, 1))

        # update the mean of the action sequence distribution
        action_sequence_mean = np.sum(action_sequences * weights, 2)

        return action_sequence_mean

    def get_action(self, current_observation, state, action_sequence_mean, key):

        # # find the elite action sequences (i.e. those associated with the highest reward)
        # elite_indices = np.argsort(total_reward)[-self.n_elite:]
        
        # # use the actions of the elite action sequences to update the mean and variance of the action sequence distribution
        # action_mean = self.alpha * np.mean(action_sequences[:, :, elite_indices], 2) + (1 - self.alpha) * action_mean
        # action_variance = self.alpha * np.var(action_sequences[:, :, elite_indices], 2) + (1 - self.alpha) * action_variance

        # if np.max(action_variance) < self.min_variance:

        #     break

        predict = partial(self.predict_one_step, state)

        # update the mean of the action sequence distribution
        action_sequence_mean = self.update_action_sequence(predict, current_observation, action_sequence_mean, key)

        # the mean of the optimised action sequence distribution (at the current time step) is the action to take in the environment
        action = action_sequence_mean[0, :]

        # shift the mean of the action sequence distribution by one time point
        # this implicitly copies/repeats the last time point (alternatively it could be set to 0)
        action_sequence_mean = action_sequence_mean.at[:-1, :].set(action_sequence_mean[1:, :])

        return action, action_sequence_mean