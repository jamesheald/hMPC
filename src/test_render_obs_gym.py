from matplotlib import animation
import matplotlib.pyplot as plt

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


from mujoco_py import GlfwContext
GlfwContext(offscreen = True)

import gym
import warmup
from jax import jit
import numpy as np
# env = gym.make("humanreacher-v0")
env = gym.make("muscle_arm-v0")
frames = []
qpos = []
qvel = []
env.seed(6)
state = env.reset()
ep_steps = 0
while ep_steps < 50:
    next_state, reward, done, info = env.step(env.action_space.sample())
    qpos.append(np.copy(next_state[:2]))
    qvel.append(np.copy(next_state[2:4]))
    frames.append(env.render(mode = "rgb_array"))
    ep_steps += 1

frames2 = []
env.seed(6)
state = env.reset()
ep_steps = 0
while ep_steps < 50:
    env.set_state(np.concatenate((qpos[ep_steps], env.target[:2])), np.concatenate((qvel[ep_steps], np.zeros(2,))))
    frames2.append(env.render(mode = "rgb_array"))
    ep_steps += 1

save_frames_as_gif(frames,filename='gym_animation.gif')
save_frames_as_gif(frames2,filename='gym_animation2.gif')