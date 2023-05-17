from jax import random
import jax.numpy as np
from flax import linen as nn

class MPPI:

    def __init__(self, env, args):

        self.horizon = args.horizon
        self.n_actions = env.action_space.shape[0]
        self.n_sequences = args.n_sequences
        self.reward_weighting_factor = args.reward_weighting_factor
        self.noise_std = args.noise_std

    def sample_candidate_action_sequences(self, action_sequence_mean, key):

        # use the same mean for each sequence
        action_sequence_mean = np.repeat(action_sequence_mean[:, :, None], self.n_sequences, axis = 2)

        # sample noise from a truncated normal distribution
        lower = (-1 - action_sequence_mean) / self.noise_std
        upper = (1 - action_sequence_mean) / self.noise_std
        eps = random.truncated_normal(key, lower = lower, upper = upper, shape = (self.horizon, self.n_actions, self.n_sequences)) * self.noise_std

        action_sequences = action_sequence_mean + eps

        return action_sequences

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

    # def get_action_learned_model(self, env, observation, state, action_sequence_mean, key):

        # # find the elite action sequences (i.e. those associated with the highest reward)
        # elite_indices = np.argsort(total_reward)[-self.n_elite:]
        
        # # use the actions of the elite action sequences to update the mean and variance of the action sequence distribution
        # action_mean = self.alpha * np.mean(action_sequences[:, :, elite_indices], 2) + (1 - self.alpha) * action_mean
        # action_variance = self.alpha * np.var(action_sequences[:, :, elite_indices], 2) + (1 - self.alpha) * action_variance

        # if np.max(action_variance) < self.min_variance:

        #     break