from jax import random, jit
from train import random_action
from brax import envs
from brax.io import image
from IPython.display import Image
from render import forward_kinematics, get_camera, create_gif, merge_gifs
import imageio
from utils import keyGen
from copy import copy
import time
import os

def render_rollout(env, controller, state, time_steps, key):

    jit_env_step = jit(env.step)

    key, subkeys = keyGen(key, n_subkeys = time_steps + 1)

    env_state = env.reset(rng = next(subkeys))
    pipeline_state = copy(env_state.pipeline_state)

    rollout = []
    rollout_obs = []
    for time in range(time_steps):

        action, action_sequence_mean = random_action(controller, env, env_state, state, [], next(subkeys))

        env_state = jit_env_step(env_state, action)
        rollout.append(env_state.pipeline_state)

        pipeline_state = forward_kinematics(env, env_state.obs, pipeline_state)
        rollout_obs.append(pipeline_state)

    cameras = [get_camera()] * time_steps

    # create and save a gif of the state-based rollout
    create_gif(env, rollout, cameras, 'state.gif')

    # create and save a gif of the observation-based rollout
    create_gif(env, rollout_obs, cameras, 'obs.gif')

    # merge both gifs into a single gif (arranged side-by-side)
    gif1 = imageio.get_reader('state.gif')
    gif2 = imageio.get_reader('obs.gif')
    merge_gifs(gif1, gif2, 'merged_gif.gif', env.dt)

    # delete original gifs
    os.remove('state.gif')
    os.remove('obs.gif')

    return None

env = envs.create(env_name = 'reacher')
controller = []
state = []
time_steps = 100
key = random.PRNGKey(0)

start_time = time.time()
render_rollout(env, controller, state, time_steps, key)
duration = time.time() - start_time
print(duration)