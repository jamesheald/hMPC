import gym
import numpy as np
import mujoco
from copy import deepcopy, copy
import glfw
import pickle
from numpy.random import multivariate_normal as mvn

mode = 1 # 1 for MPC, 2 for replay of MPC solution found in mode 1
episode_length = 10
max_horizon = episode_length
n_rollouts = 10
CEM_iterations = 2
CEM_alpha = 0.9
n_elite = 5

def sample_action(mean,covariance):
    action = mvn(mean,covariance)
    return action

def random_shooting(t,n_rollouts,episode_length,max_horizon,n_actions):
    first_action = []
    horizon = min(episode_length-t,max_horizon)
    action_sequences = np.zeros((n_rollouts,horizon,n_actions))
    horizon_return = [0 for i in range(n_rollouts)]
    action_mean = np.zeros(n_actions)
    action_covariance = 0.1*np.identity(n_actions)
    for i in range(n_rollouts):
        for j in range(horizon):
            # action_sequences[i,j,:] = env.action_space.sample()
            action_sequences[i,j,:] = sample_action(action_mean,action_covariance)
            observation, reward, done, info = env.step(action_sequences[i,j,:])
            if j == 0:
                first_action.append(action_sequences[i,j,:])
            horizon_return[i] += reward
        # glfw.terminate() # close renderer viewer, as a new viewer opens. use env.render(close=True) for non-mujoco environments possibly (untested)
        # env = save_state
        env.data.qpos = qpos
        env.data.qvel = qvel
        # print(observation)
        # print(qpos)
    idx_optimal_action = horizon_return.index(max(horizon_return))
    optimal_action = first_action[idx_optimal_action]
    return optimal_action

def CEM_method(t,n_rollouts,episode_length,max_horizon,n_actions,CEM_iterations,CEM_alpha,n_elite):
    horizon = min(episode_length-t,max_horizon)
    for CEM_iteration in range(CEM_iterations):
        action_sequences = np.zeros((n_actions,n_rollouts,horizon))
        horizon_return = [0 for i in range(n_rollouts)]
        if CEM_iteration == 0:
            action_mean = np.zeros((n_actions,horizon))
            action_covariance = 0.1*np.identity(n_actions).repeat(horizon,axis=1).reshape((n_actions,n_actions,horizon))
        for i in range(n_rollouts):
            for j in range(horizon):
                action_sequences[:,i,j] = sample_action(action_mean[:,j],action_covariance[:,:,j])
                observation, reward, done, info = env.step(action_sequences[:,i,j])
                horizon_return[i] += reward
            # glfw.terminate() # close renderer viewer, as a new viewer opens. use env.render(close=True) for non-mujoco environments possibly (untested)
            # env = save_state
            env.data.qpos = qpos
            env.data.qvel = qvel
            # print(observation)
            # print(qpos)
        elite_indices = np.argsort(horizon_return)[-n_elite:]
        action_mean = CEM_alpha*np.mean(action_sequences[:,elite_indices,:],1) + (1-CEM_alpha)*action_mean
        new_covar = np.zeros((n_actions,n_actions,horizon))
        for j in range(horizon):
            new_covar[:,:,j] = np.diag(np.var(action_sequences[:,elite_indices,j],1))
        action_covariance = CEM_alpha*new_covar + (1-CEM_alpha)*action_covariance
    optimal_action = action_mean[:,0]
    return optimal_action

if mode == 1:

    action_sequence = []
    env = gym.make('Reacher-v4')
    n_actions = env.action_space.shape[0]
    # env = gym.make('HandManipulatePen-v0')
    # env = gym.make('Humanoid-v4')
    # env = gym.make('HandReach-v0')
    observation = env.reset()
    save_initial_state = deepcopy(env)
    initial_qpos = copy(env.data.qpos)
    initial_qvel = copy(env.data.qvel)
    for t in range(episode_length):
        # env.render()
        # print(observation)
        print(t)
        # save_state = deepcopy(env)
        qpos = copy(env.data.qpos)
        qvel = copy(env.data.qvel)
        # print(qpos)
        optimal_action = CEM_method(t,n_rollouts,episode_length,max_horizon,n_actions,CEM_iterations,CEM_alpha,n_elite)
        # optimal_action = random_shooting(t,n_rollouts,episode_length,max_horizon)
        action_sequence.append(optimal_action)
        observation, reward, done, info = env.step(optimal_action)
        done = False
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    env.data.qpos = initial_qpos
    env.data.qvel = initial_qvel
    for t in range(episode_length):
        env.render()
        observation, reward, done, info = env.step(action_sequence[t])
        done = False
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()

    save_dict = {"action_sequence":action_sequence,"initial_qpos":initial_qpos,"initial_qvel":initial_qvel}
    filename = "action_sequence.pickle"
    open_file = open(filename,"wb")
    pickle.dump(save_dict,open_file)
    open_file.close()

if mode == 2: 
    filename = "action_sequence.pickle"
    open_file = open(filename,"rb")
    loaded_dict = pickle.load(open_file)
    open_file.close()

    env = gym.make('Reacher-v4')
    observation = env.reset()

    env.data.qpos = loaded_dict['initial_qpos']
    env.data.qvel = loaded_dict['initial_qvel']
    for t in range(episode_length):
        env.render()
        observation, reward, done, info = env.step(loaded_dict['action_sequence'][t])
        done = False
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()