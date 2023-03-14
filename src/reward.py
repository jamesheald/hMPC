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

def reward_function(state, action):

    target_position = np.array([0.14, 0.14]) # state[4:6]

    fingertip_position = state[8:10]

    accuracy_reward = -np.linalg.norm(target_position - fingertip_position)

    effort_cost = np.sum(action ** 2)

    reward = accuracy_reward - effort_cost

    return reward