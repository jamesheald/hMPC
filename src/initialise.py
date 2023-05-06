from jax import random
from flax import linen as nn
import jax.numpy as np
from utils import keyGen
from brax import envs

class dynamics_model_MLP(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(features = 500)(x)
        x = nn.relu(x)
        x = nn.Dense(features = 500)(x)
        x = nn.relu(x)
        x = nn.Dense(features = self.output_dim)(x)
        
        # x = nn.Dense(features = self.x_dim * 2)(x)
        # # mean and log variances of Gaussian distribution over next state
        # x_mean, x_log_var = np.split(x, 2, axis = 1)
        # return {'x_mean': x_mean, 'x_log_var': x_log_var}
        
        return x

class state_encoding_GRU(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(features = self.hidden_dim)(x)

        return x

class dynamics_model_GRU(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, carry, inputs):
        
        carry, outputs = nn.GRUCell(kernel_init = initializers.lecun_normal())(carry, inputs)
        
        # mean and log variances of Gaussian distribution over next state
        params = nn.Dense(features = self.output_dim * 2)(outputs)
        x_mean, x_log_var = np.split(params, 2, axis = 1)

        return carry, {'x_mean': x_mean, 'x_log_var': x_log_var}

class VAE(nn.Module):
    n_loops_top_layer: int
    x_dim: list
    image_dim: list
    T: int
    dt: float
    tau: float

    def setup(self):
        
        self.state_encoding_GRU = state_encoding_GRU(self.hidden_dim)
        self.sampler = dynamics_model_GRU(self.hidden_dim, self.output_dim)

    def __call__(self, data, params, A, gamma, state_myo, key):

        output_encoder = self.encoder(data[None,:,:,None])
        z1, z2 = self.sampler(output_encoder, params, key)
        output_decoder = self.decoder(params, A, gamma, z1, z2, state_myo)
        
        return output_encoder | output_decoder

def estimate_return(self, predict, env_state, action_sequence):

    carry = env_state.obs, env_state

    # use the learned model to predict the observation sequence and reward associated with each action sequence
    _, reward = lax.scan(predict, carry, action_sequence)

    total_reward = reward.sum()

    return total_reward

def initialise_model(args):

    # explicitly generate a PRNG key
    key = random.PRNGKey(args.jax_seed)

    # generate the required number of subkeys
    key, subkeys = keyGen(key, n_subkeys = 2)
    
    # define the model and initialise its parameters
    env = envs.create(env_name = args.environment_name)
    model = dynamics_model(output_dim = env.observation_size)
    params = model.init(x = np.ones(env.observation_size + env.action_size), rngs = {'params': next(subkeys)})

    return model, params, args, key