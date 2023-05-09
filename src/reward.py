import jax.numpy as np
from jax import vmap
from utils import keyGen

# def generate_target(key, disk_radius = 0.2):
#     """
#     randomly generate a target within a disk centered at the origin
#     """
#     # sample x-coordinate of target
#     x = random.uniform(next(subkeys), shape = (1,), minval = -disk_radius, maxval = disk_radius)

#     # maximum possible value of y given x and disk radius
#     max_y = np.sqrt(disk_radius ** 2 - x ** 2)

#     # sample y-coordinate of target
#     y = random.uniform(next(subkeys), shape = (1,), minval = -max_y, maxval = max_y)

#     target_position = np.array((x,y))

#     return target_position

def reward_function(observation, action, next_observation):

    reward_dist = -np.linalg.norm(next_observation[8:]) ** 2 # original: -np.linalg.norm(next_observation[8:])

    reward_ctrl = -np.sum(action ** 2)

    reward = 100 * reward_dist + 1 * reward_ctrl # 100 works better than 10 with norm ** 2; 10 works ok with norm

    return reward

def expected_reward(action, mu, log_var):

    state_reward = -(mu[8:] @ mu[8:] + np.exp(log_var).sum()) # expectation not of norm (as in original reacher environment) but of norm ** 2

    action_reward = -np.sum(action ** 2)

    reward = 100 * state_reward + 1 * action_reward # 100 works better than 10 with norm ** 2; 10 works ok with norm

    return reward

batch_expected_reward = vmap(expected_reward, in_axes = (0, 0, 0))