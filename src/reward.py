import jax.numpy as np
from jax import vmap
from utils import keyGen

def reward_function(observation, action, next_observation):

    reward_dist = -np.linalg.norm(next_observation[8:]) ** 2 # original: -np.linalg.norm(next_observation[8:])

    reward_ctrl = -np.sum(action ** 2)

    reward = 100 * reward_dist + 1 * reward_ctrl # 100 works better than 10 with norm ** 2; 10 works ok with norm

    return reward

def expected_reward(action, mu, log_var):

    state_reward = -(mu @ mu + np.exp(log_var).sum()) # expectation not of norm (as in original reacher environment) but of norm ** 2

    action_reward = -np.sum(action ** 2)

    reward = 100 * state_reward + 1 * action_reward # 100 works better than 10 with norm ** 2; 10 works ok with norm

    return reward

batch_expected_reward = vmap(expected_reward, in_axes = (0, 0, 0))

# def _get_reward(self, ee_pos, action):
#     lamb = 1e-4  # 1e-4
#     epsilon = 1e-4
#     log_weight = 1.0
#     rew_weight = 0.1

#     d = np.mean(np.square(ee_pos - self.target))
#     activ_cost = lamb * np.mean(np.square(action))
#     if self.sparse_reward:
#         return -1.0
#     return (
#         -rew_weight * (d + log_weight * np.log(d + epsilon**2))
#         - activ_cost
#         - 2
#     )