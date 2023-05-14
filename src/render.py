from jax import numpy as np  
import numpy as onp  
from itertools import product
from jax import jit, vmap
from utils import calculate_theta
from brax.kinematics import forward
from pytinyrenderer import TinyRenderCamera as Camera
from brax.io import image
from IPython.display import Image
import imageio

def get_camera(width: int = 340, height: int = 240, ssaa: int = 2):
      """Gets camera object."""
      
      eye, up = list(np.array([0.01, 0, 1])), [0, 0, -1] # camera location and up orientation
      hfov = 50.0
      vfov = hfov * height / width
      target = [0, 0, 0]
      camera = Camera(viewWidth = width * ssaa, viewHeight = height * ssaa, position = eye, target = target, up = up, hfov = hfov, vfov = vfov)
      
      return camera

jit_forward = jit(forward)

def forward_kinematics(env, observations, state):

    q = np.concatenate([calculate_theta(observations), observations[4:6]])
    x, _ = jit_forward(env.sys, q, np.zeros(4,))
    state = state.replace(x = x)

    return state

def create_gif(env, rollout, cameras, filename, width = 320, height = 240):

    # create and save a gif of the state-based rollout
    gif = Image(image.render(env.sys, [s for s in rollout], cameras = cameras, width = width, height = height, fmt = 'gif'))
    open(filename, 'wb').write(gif.data)

    return None

def merge_gifs(gif1, gif2, filename, dt):

    merged_gif = imageio.get_writer(filename)

    number_of_frames = gif1.get_length()
    for frame_number in range(number_of_frames):
        
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        new_image = onp.hstack((img1, img2))
        merged_gif.append_data(new_image)

    merged_gif.close()
    gif = imageio.mimread(filename)
    imageio.mimsave(filename, gif, fps = 1 / dt)

    gif1.close()
    gif2.close()

    return None