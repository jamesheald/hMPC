from jax import random, lax, vmap
from flax import linen as nn
import jax.numpy as np
from utils import keyGen
import gym
from brax import envs
from reward import batch_expected_reward

class observation_encoder(nn.Module):
    carry_dim: int

    @nn.compact
    def __call__(self, observation):
        
        # map observation to carry of the GRU
        carry = nn.Dense(features = self.carry_dim)(observation)

        return carry

class dynamics_model(nn.Module):
    prediction_dim: int

    @nn.compact
    def __call__(self, carry, action):
        
        # predict the next state using the learned dynamics model
        carry, outputs = nn.GRUCell(kernel_init = nn.initializers.lecun_normal())(carry, action)
        
        # mean and log variances of Gaussian distribution over next state
        mu, log_var = np.split(nn.Dense(features = self.prediction_dim * 2)(outputs), 2)

        return carry, (mu, log_var)

class rollout_prediction(nn.Module):
    carry_dim: int
    prediction_dim: int
    action_dim: int

    def setup(self):
        
        self.encoder = observation_encoder(self.carry_dim)
        self.dynamics = dynamics_model(self.prediction_dim)
        self.dynamics_params = self.param('dynamics_params', self.dynamics.init, np.ones(self.carry_dim), np.ones(self.action_dim))

    def __call__(self, observation, action_sequence):

        # initialise the carry of the recurrent dynamics model by encoding the current observation
        carry = self.encoder(observation)

        # use the learned model to predict the distribution of future observations given an action sequence
        _, (mu, log_var) = lax.scan(lambda carry, action: self.dynamics.apply(self.dynamics_params, carry, action), carry, action_sequence)

        # calculate the expected cumulative reward under the predicted distribution of future observations
        # future observations are defined relative to the current observation
        estimated_cumulative_reward = batch_expected_reward(action_sequence, mu + observation[-self.prediction_dim:], log_var).sum()
        
        return mu, log_var, estimated_cumulative_reward

def initialise_model(args):

    # explicitly generate a PRNG key
    key = random.PRNGKey(args.jax_seed)

    # generate the required number of subkeys
    key, subkeys = keyGen(key, n_subkeys = 1)
    
    # define the model and initialise its parameters
    env = gym.make(args.environment_name)
    model = rollout_prediction(carry_dim = args.carry_dim, prediction_dim = 3, action_dim = env.action_space.shape[0])
    params = model.init(observation = np.ones(env.observation_space.shape[0]), action_sequence = np.ones((args.horizon, env.action_space.shape[0])), rngs = {'params': next(subkeys)})

    return model, params, args, key