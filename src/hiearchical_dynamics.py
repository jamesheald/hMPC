import jax.numpy as np
from flax import linen as nn
from jax import random
from utils import stabilise_variance
from utils import keyGen

class encoder(nn.Module):
    hidden_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(features = self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features = self.z_dim * 2)(x)

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

        # sample the latents code
        z = sample_diag_Gaussian(p_z['mean'], p_z['log_var'], key)

        return z

class decoder(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(features = self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features = self.output_dim)(x)
        
        return x

class dynamics(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, state, action):
        
        x = np.concatenate((state, action))
        x = nn.Dense(features = self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features = self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features = self.output_dim)(x)
        
        # x = nn.Dense(features = self.x_dim * 2)(x)
        # # mean and log variances of Gaussian distribution over next state
        # x_mean, x_log_var = np.split(x, 2, axis = 1)
        # return {'x_mean': x_mean, 'x_log_var': x_log_var}
        
        return x

class hierarchical_dynamics_model(nn.Module):
    
    # state encoder
    hidden_dim_state_encoder: int
    z_dim_state_encoder: int

    # action encoder
    hidden_dim_action_encoder: int
    z_dim_action_encoder: int

    # state decoder
    hidden_dim_state_decoder: int
    output_dim_state_decoder: int

    # action decoder
    hidden_dim_action_decoder: int
    output_dim_action_decoder: int

    # dynamics model
    hidden_dim_dynamics: int # 500
    output_dim_dynamics: int

    # dynamics model in latent space
    hidden_dim_dynamics_latent: int
    output_dim_dynamics_latent: int

    def setup(self):
        
        # encoders
        self.state_encoder = encoder(self.hidden_dim_state_encoder, self.z_dim_state_encoder)
        self.action_encoder = encoder(self.hidden_dim_action_encoder, self.z_dim_action_encoder)
        
        # latent code sampler
        self.sampler = sampler()

        # decoders
        self.state_decoder = decoder(self.hidden_dim_state_decoder, self.output_dim_state_decoder)
        self.action_decoder = decoder(self.hidden_dim_action_decoder, self.output_dim_action_decoder)

        # dynamics models
        self.dynamics = dynamics(self.hidden_dim_dynamics, self.output_dim_dynamics)
        self.dynamics_latent = dynamics(self.hidden_dim_dynamics_latent, self.output_dim_dynamics_latent)

    # def state_encoder(self, state):

    #     return encoder(self.hidden_dim_state_encoder, self.z_dim_state_encoder)(state)

    def __call__(self, state, action, key):

        key, subkeys = keyGen(key, n_subkeys = 2)

        # infer latent codes
        p_z_state = self.state_encoder(state)
        p_z_action = self.action_encoder(action)

        # sample latent codes
        z_state = self.sampler(p_z_state, next(subkeys))
        z_action = self.sampler(p_z_action, next(subkeys))

        # reconstruct state and action from sampled latent codes
        state_hat = self.state_decoder(z_state)
        action_hat = self.action_decoder(z_action)

        # predict next state
        next_state_hat = self.dynamics(state, action)
        next_z_state = self.dynamics_latent(z_state, z_action)
        next_state_hat_via_z = self.state_decoder(next_z_state)
        
        return {'z_state': z_state, 'z_action': z_action,
                'state_hat': state_hat, 'action_hat': action_hat,
                'next_state': next_state_hat, 'next_state_hat_via_z': next_state_hat_via_z}

# from hiearchical_dynamics import hierarchical_dynamics_model
# from jax import numpy as np
# from jax import random
# model = hierarchical_dynamics_model(hidden_dim_state_encoder = 10, z_dim_state_encoder = 3,
#                                     hidden_dim_action_encoder = 5, z_dim_action_encoder = 3, 
#                                     hidden_dim_state_decoder = 10, output_dim_state_decoder = 15,
#                                     hidden_dim_action_decoder = 3, output_dim_action_decoder = 15,
#                                     hidden_dim_dynamics = 500, output_dim_dynamics = 10,
#                                     hidden_dim_dynamics_latent = 50, output_dim_dynamics_latent = 3)
# params = model.init(state = np.ones(2), action = np.ones(2), key = random.PRNGKey(1), rngs = {'params': random.PRNGKey(1)})
# output = model.apply(params, state = np.ones(2), action = np.ones(2), key = random.PRNGKey(1))
