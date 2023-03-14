from jax import jit, vmap, random, lax
import jax.numpy as np
from utils import keyGen
from reward import reward_function

class CEM:

    def __init__(self, env, args):

        self.n_sequences = args.n_sequences
        self.n_actions = env.action_space.shape[0]
        self.n_iterations = args.n_iterations
        self.n_elite = args.n_elite
        self.alpha = args.alpha
        self.min_variance = args.min_variance

    def get_action(self, current_observation, dynamics_model, params, horizon, key):

        def predict_one_step(observation, action):

            # concatenate the current observation and action to form the input to the learned dynamics model
            inputs = np.concatenate((observation, action))

            # predict the next observation using the learned myosuite dynamics model
            next_observation = dynamics_model(params, inputs)

            # calculate the reward based on the current observation and action (for some environments, the reward will also depend on the next observation)
            reward = reward_function(observation, action)

            return next_observation, reward

        def estimate_cumulative_reward(current_observation, action_sequence):

            # use the learned model to predict the observation sequence and reward associated with each action sequence
            _, reward = lax.scan(predict_one_step, current_observation, action_sequence)

            cumulative_reward = reward.sum()

            return cumulative_reward

        batch_estimate_cumulative_reward = vmap(estimate_cumulative_reward, in_axes = (None, 2))
        jit_batch_estimate_cumulative_reward = jit(batch_estimate_cumulative_reward)

        # initialise mean and variance of the action sequence distribution
        action_mean = np.zeros((horizon, self.n_actions))
        action_variance = 0.1 * np.ones((horizon, self.n_actions))

        key, subkeys = keyGen(key, n_subkeys = self.n_iterations)
        for iteration in range(self.n_iterations):

            # sample multiple action sequences from the action sequence distribution
            action_sequence = np.expand_dims(action_mean, 2) + np.expand_dims(np.sqrt(action_variance), 2) * random.normal(next(subkeys), (horizon, self.n_actions, self.n_sequences))

            # use the learned model to estimate the cumulative reward associated with each action sequence
            cumulative_reward = jit_batch_estimate_cumulative_reward(current_observation, action_sequence)

            # find the elite action sequences (i.e. those associated with the highest reward)
            elite_indices = np.argsort(cumulative_reward)[-self.n_elite:]
            
            # use the actions of the elite action sequences to update the mean and variance of the action sequence distribution
            action_mean = self.alpha * np.mean(action_sequence[:, :, elite_indices], 2) + (1 - self.alpha) * action_mean
            action_variance = self.alpha * np.var(action_sequence[:, :, elite_indices], 2) + (1 - self.alpha) * action_variance

            if np.max(action_variance) < self.min_variance:

                break

        # the mean of the optimised action sequence distribution (at the current time step) is the action to take in the environment
        best_action = action_mean[0, :]

        return best_action