from jax import random, lax, vmap
from flax import linen as nn
import jax.numpy as np
from utils import keyGen, stabilise_variance
from flax.core.frozen_dict import freeze, unfreeze
import gym
from reward import batch_expected_reward

class encode_observation(nn.Module):
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
        
        self.encoder = encode_observation(self.carry_dim)
        self.dynamics = dynamics_model(self.prediction_dim)
        self.dynamics_params = self.param('dynamics_params', self.dynamics.init, np.ones(self.carry_dim), np.ones(self.action_dim))

    def __call__(self, observation, action_sequence):

        # initialise the carry of the recurrent dynamics model by encoding the current observation
        carry = self.encoder(observation)

        # use the learned model to predict the distribution of future observations given an action sequence
        _, (mu, log_var) = lax.scan(lambda carry, action: self.dynamics.apply(self.dynamics_params, carry, action), carry, action_sequence)

        # calculate the expected cumulative reward under the predicted distribution of future observations
        # future observations are defined relative to the current observation
        estimated_cumulative_reward = batch_expected_reward(action_sequence, mu[:,-3:-1] + observation[-3:-1] - observation[-6:-4], log_var).sum()
        
        return mu, log_var, estimated_cumulative_reward

class encode_action(nn.Module):
    action_encoder_hidden_dim: int
    action_code_dim: int

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(features = self.action_encoder_hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features = self.action_code_dim * 2)(x)

        # mean and log variances of the Gaussian distribution over the latent code
        z_mean, z_log_var = np.split(x, 2)

        return {'mean': z_mean, 'log_var': z_log_var}

class sampler(nn.Module):
    
    @nn.compact
    def __call__(self, p_z, key):
        
        def sample_diag_Gaussian(mean, log_var, key):
            """
            sample from a diagonal Gaussian distribution
            """
            log_var = stabilise_variance(log_var)

            return mean + np.exp(0.5 * log_var) * random.normal(key, mean.shape)

        # sample from a diagonal Gaussian distribution
        z = sample_diag_Gaussian(p_z['mean'], p_z['log_var'], key)

        return z

class decode_action(nn.Module):
    action_decoder_hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(features = self.action_decoder_hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features = self.output_dim * 2)(x)

        # mean and log variances of the Gaussian distribution over the reconstructed action
        a_mean, a_log_var = np.split(x, 2)

        return {'mean': a_mean, 'log_var': a_log_var}

class VAE(nn.Module):
    action_encoder_hidden_dim: int
    action_code_dim: int
    action_decoder_hidden_dim: int
    output_dim: int

    def setup(self):
        
        # encoder
        self.action_encoder = encode_action(self.action_encoder_hidden_dim, self.action_code_dim)
        
        # latent code sampler
        self.sampler = sampler()

        # decoder
        self.action_decoder = decode_action(self.action_decoder_hidden_dim, self.output_dim)

    def __call__(self, action, key):

        # infer the distribution of the latent code
        p_z = self.action_encoder(action)

        # sample the latent code
        z = self.sampler(p_z, key)

        # reconstruct the action from the sampled latent code
        p_a = self.action_decoder(z)
        
        return {'p_z': p_z, 'p_a': p_a}

def initialise_model(args):

    # explicitly generate a PRNG key
    key = random.PRNGKey(args.jax_seed)

    # generate the required number of subkeys
    key, subkeys = keyGen(key, n_subkeys = 3)
    
    # define the models and initialise their parameters
    env = gym.make(args.environment_name)
    dynamics_model = rollout_prediction(carry_dim = args.carry_dim, prediction_dim = env.observation_space.shape[0], action_dim = env.action_space.shape[0])
    dynamics_params = dynamics_model.init(observation = np.ones(env.observation_space.shape[0]), action_sequence = np.ones((args.horizon, env.action_space.shape[0])), rngs = {'params': next(subkeys)})

    VAE_model = VAE(action_encoder_hidden_dim = args.action_encoder_hidden_dim, action_code_dim = args.action_code_dim,
              action_decoder_hidden_dim = args.action_decoder_hidden_dim, output_dim = env.action_space.shape[0])
    VAE_params = VAE_model.init(action = np.ones(env.action_space.shape[0]), key = next(subkeys), rngs = {'params': next(subkeys)})

    # concatenate all parameters into a single dictionary
    VAE_params = unfreeze(VAE_params)
    VAE_params = VAE_params | {'prior_z_log_var': args.prior_z_log_var}
    VAE_params = freeze(VAE_params)

    # action_encoder = encode_action(hidden_dim = args.action_encoder_hidden_dim, code_dim = args.action_code_dim)
    # action_encoder_params = action_encoder.init(x = np.ones(env.action_space.shape[0]), rngs = {'params': next(subkeys)})

    # action_decoder = decode_action(hidden_dim = args.action_decoder_hidden_dim, output_dim = env.action_space.shape[0])
    # action_decoder_params = action_decoder.init(x = np.ones(args.action_code_dim), rngs = {'params': next(subkeys)})

    models = (dynamics_model, VAE_model)
    params = (dynamics_params, VAE_params)

    return models, params, args, key