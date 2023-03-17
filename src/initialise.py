from jax import random
from flax import linen as nn
import jax.numpy as np
from utils import keyGen
from brax import envs

class dynamics_model(nn.Module):
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