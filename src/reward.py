import jax.numpy as np
from jax import vmap

def expected_reward(action, mu, log_var, weight_state = 0.1, weight_action = 1e-4):

    reward_state = - (mu @ mu + np.exp(log_var).sum()) / mu.size

    reward_action = - np.mean(action ** 2)

    reward = weight_state * reward_state + weight_action * reward_action

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