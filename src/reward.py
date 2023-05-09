import jax.numpy as np
from utils import keyGen

def generate_target(key, disk_radius = 0.2):
    """
    randomly generate a target within a disk centered at the origin
    """
    # sample x-coordinate of target
    x = random.uniform(next(subkeys), shape = (1,), minval = -disk_radius, maxval = disk_radius)

    # maximum possible value of y given x and disk radius
    max_y = np.sqrt(disk_radius ** 2 - x ** 2)

    # sample y-coordinate of target
    y = random.uniform(next(subkeys), shape = (1,), minval = -max_y, maxval = max_y)

    target_position = np.array((x,y))

    return target_position

def reward_function(observation, action, next_observation):

    reward_dist = -np.linalg.norm(next_observation[8:])

    reward_ctrl = -np.sum(action ** 2)

    reward = 10 * reward_dist + 1 * reward_ctrl

    return reward

def expected_reward(action, mu, log_var):

    state_reward = -np.linalg.norm(mu[8:])

    action_reward = -np.sum(action ** 2)

    reward = 10 * state_reward + 1 * action_reward

    return reward