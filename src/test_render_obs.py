from train import random_action
from brax import kinematics
from brax import envs
from jax import random, jit
import numpy as np
from utils import keyGen
from copy import copy
import math
from itertools import product

from brax.io import image
# from brax.io.image import _eye, _up #, _eye
from IPython.display import Image 
from pytinyrenderer import TinyRenderCamera as Camera

import numpy as onp

import imageio
import numpy as np    

def merge_gifs(gif1, gif2):

	merged_gif = imageio.get_writer('merged_gif.gif')

	number_of_frames = gif1.get_length()
	for frame_number in range(number_of_frames):
	    img1 = gif1.get_next_data()
	    img2 = gif2.get_next_data()
	    new_image = np.hstack((img1, img2))
	    merged_gif.append_data(new_image)

	merged_gif.close()
	gif = imageio.mimread('merged_gif.gif')
	imageio.mimsave('merged_gif.gif', gif, fps = 25)

	gif1.close()
	gif2.close()

	return None

# def get_camera(env, env_state, width, height):

#     sys = env.sys
#     qp = env_state.pipeline_state.x # env_state.qp
#     ssaa = 2
#     eye, up = _eye(sys, env_state.pipeline_state), _up(sys)
#     hfov = 58.0
#     vfov = hfov * height / width
#     target = [qp.pos[0, 0], qp.pos[0, 1], 0]
#     camera = Camera(
#         viewWidth = width * ssaa,
#         viewHeight = height * ssaa,
#         position = eye,
#         target = target,
#         up = up,
#         hfov = hfov,
#         vfov = vfov)

#     return camera

def _eye(sys, state):
	"""Determines the camera location for a Brax system."""
	
	xj = state.x.vmap().do(sys.link.joint)
	dist = onp.concatenate(xj.pos[None, ...] - xj.pos[:, None, ...])
	dist = onp.linalg.norm(dist, axis=1).max()
	off = [5 * dist, -5 * dist, 5 * dist] # off = [2 * dist, -2 * dist, dist]
	
	return list(state.x.pos[0, :] + onp.array(off))

def _up(unused_sys):
	"""Determines the up orientation of the camera."""
	
	return [0, 0, 1]

def get_camera(sys, state, width: int = 320, height: int = 240, ssaa: int = 2):
      """Gets camera object."""
      eye, up = _eye(sys, state), _up(sys)
      hfov = 58.0 # 58.0
      vfov = hfov * height / width
      target = [state.x.pos[0, 0], state.x.pos[0, 1], 0]
      camera = Camera(
          viewWidth=width * ssaa,
          viewHeight=height * ssaa,
          position=eye,
          target=target,
          up=up,
          hfov=hfov,
          vfov=vfov)
      return camera

def calculate_theta(obs):

	theta = []
	for joint in range(2):
		cos_theta = obs[joint]
		sin_theta = obs[joint + 2]
		options = [math.acos(cos_theta), -math.acos(cos_theta), math.asin(sin_theta)]
		if math.asin(sin_theta) < 0:
			options.append(-math.pi - math.asin(sin_theta))
		else:
			options.append(math.pi - math.asin(sin_theta))
		for i, j in product(range(2), range(2)):
			if math.isclose(options[i], options[j + 2], abs_tol = 1e-3):
				theta.append(options[i])
				break

	return theta

def render_rollout(env, controller, state, time_steps, key):

    jit_env_step = jit(env.step)
    jit_kinematics_forward = jit(kinematics.forward)

    key, subkeys = keyGen(key, n_subkeys = time_steps + 1)

    env_state = env.reset(rng = next(subkeys))
    pipeline_state = copy(env_state.pipeline_state) # env_state

    cameras = []
    width = 320
    height = 240

    # perform rollout
    rollout = []
    rollout_obs = []
  
    for time in range(time_steps):

        action, action_sequence_mean = random_action(controller, env, env_state, state, [], next(subkeys))

        env_state = jit_env_step(env_state, action)
        rollout.append(env_state.pipeline_state) # env_state
        # cameras.append(get_camera(env, env_state, width, height))
        cameras.append(get_camera(env.sys, env_state.pipeline_state))

        q = np.concatenate([calculate_theta(env_state.obs), env_state.obs[4:6]])
        x, _ = jit_kinematics_forward(env.sys, q, np.zeros(4,))
        pipeline_state = pipeline_state.replace(x = x) # replace(qp = x)
        rollout_obs.append(pipeline_state)

    # create and save a gif of the rollout
    gif = Image(image.render(env.sys, [s for s in rollout], cameras = cameras, width = width, height = height, fmt = 'gif')) # s.qp not s
    open('state.gif', 'wb').write(gif.data)

    gif = Image(image.render(env.sys, [s for s in rollout_obs], cameras = cameras, width = width, height = height, fmt = 'gif')) # s.qp not s
    open('obs.gif', 'wb').write(gif.data)

    gif1 = imageio.get_reader('state.gif')
    gif2 = imageio.get_reader('obs.gif')

    merge_gifs(gif1, gif2)

    return None

env = envs.create(env_name = 'reacher')
controller = []
state = []
time_steps = 100
key = random.PRNGKey(0)
render_rollout(env, controller, state, time_steps, key)

# https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb#scrollTo=kUrAlZTod7t_

#@markdown ## ⚠️ PLEASE NOTE:
#@markdown This colab runs best using a GPU runtime.  From the Colab menu, choose Runtime > Change Runtime Type, then select **'GPU'** in the dropdown.

# import functools
# import jax
# import os

# from datetime import datetime
# from jax import numpy as jp
# import matplotlib.pyplot as plt

# from IPython.display import HTML, clear_output

# try:
#   import brax
# except ImportError:
#   !pip install git+https://github.com/google/brax.git@main
#   clear_output()
#   import brax

# import flax
# from brax import envs
# from brax.io import model
# from brax.io import json
# from brax.io import html
# from brax.training.agents.ppo import train as ppo
# from brax.training.agents.sac import train as sac

# if 'COLAB_TPU_ADDR' in os.environ:
#   from jax.tools import colab_tpu
#   colab_tpu.setup_tpu()



# #@title Load Env { run: "auto" }

# env_name = 'reacher'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
# backend = 'positional'  # @param ['generalized', 'positional', 'spring']

# env = envs.get_environment(env_name=env_name,
#                            backend=backend)
# state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

# HTML(html.render(env.sys, [state.pipeline_state]))


# # create an env with auto-reset
# env = envs.create(env_name=env_name, backend=backend)

# jit_env_reset = jax.jit(env.reset)
# jit_env_step = jax.jit(env.step)

# from brax import kinematics
# from copy import copy
# import math
# import numpy as np

# from itertools import product

# def calculate_theta(obs):

# 	theta = []
# 	for joint in range(2):
# 		cos_theta = obs[joint]
# 		sin_theta = obs[joint + 2]
# 		options = [math.acos(cos_theta), -math.acos(cos_theta), math.asin(sin_theta)]
# 		if math.asin(sin_theta) < 0:
# 			options.append(-math.pi - math.asin(sin_theta))
# 		else:
# 			options.append(math.pi - math.asin(sin_theta))
# 		for i, j in product(range(2), range(2)):
# 			if math.isclose(options[i], options[j + 2], abs_tol = 1e-3):
# 				theta.append(options[i])
# 				break

# 	return theta

# rollout = []
# rollout_obs = []
# rng = jax.random.PRNGKey(seed=2)
# state = jit_env_reset(rng=rng)
# pipeline_state = copy(state.pipeline_state)
# for _ in range(100):
#   rollout.append(state.pipeline_state)

#   q = np.concatenate([calculate_theta(state.obs), state.obs[4:6]])
#   x, _ = kinematics.forward(env.sys, q, np.zeros(4,))
#   pipeline_state = pipeline_state.replace(x = x)
#   rollout_obs.append(pipeline_state)

#   act_rng, rng = jax.random.split(rng)
#   act = jax.random.uniform(act_rng, shape = (env.action_size,), minval = -1.0, maxval = 1.0)
#   state = jit_env_step(state, act)

# HTML(html.render(env.sys.replace(dt=env.dt), rollout))
# HTML(html.render(env.sys.replace(dt=env.dt), rollout_obs))