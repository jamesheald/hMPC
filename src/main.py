import argparse
import pickle
import os

# things that matter:
# weights on error and controls
# softmax temperature
# action noise magnitude

# things to do

# change GRU for transformer

# save collected data so you can easily train GRUs/VAEs for testing ideas!!!
# why do we need to use a random policy in first iteration (you don't need to in principal i don't think), is it just to save computational costs?

# sort out reward in dynamics_model - currently expectation not of norm (as in original reacher environment) but of norm squared
# sort of checkpoints/gif savinsg etc to monitor progress
# check youre happy with args parameters
# check clipping etc - maybe leave for now as worked fine with true dynamics model
# could calculate dynamics loss on separate unseen validation dataset to monitor progress and decide when to stop (maybe less relevant as training dataset is growing?)

# determine gradient clipping value by logging gradient norms during training and assessing

# tensorboard - save loss function, save gifs and rewards on a small number (say 3) of examples

# check lambda in predict_return is working ok (function works when parameters and arguments change)

# have a condition flag for using true vs learned dynamics
# incorporate continuation/termination flag into learned dynamics?
# only clip when passing to dynamics function, not when performing MPPI (think about whether clipped/unclipped should be passed to dyanmics fucntions and VAE)
# use scan for rollout render
# how to choose noise variance?
# smooth actions?
# have ensemble of models
# parralelise action sequence rollouts in MPC using pmap

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from initialise import initialise_model
from train import optimise_model
# from assess_model_learning import optimise_model

def main():

    parser = argparse.ArgumentParser(description = 'hyperparameters')

    # directories
    parser.add_argument('--folder_name',             default = 'to_save_model')
    parser.add_argument('--reload_state',            type = bool, default = False)
    parser.add_argument('--reload_folder_name',      default = 'saved_model')

    # model
    parser.add_argument('--jax_seed',                type = int, default = 1)
    parser.add_argument('--carry_dim',               type = int, default = 200)

    # gym environment
    # https://www.gymlibrary.dev/environments/mujoco/reacher/
    # ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    parser.add_argument('--environment_name',        default = 'muscle_arm-v0')
    parser.add_argument('--n_rollouts',              type = int, default = 5) # 30
    parser.add_argument('--time_steps',              type = int, default = 50) # 50, 1000 

    # # muscle_arm-v0 observations
    # env.sim.data.qpos[: env.nq],
    # env.sim.data.qvel[: env.nq],
    # env.muscle_length(),
    # env.muscle_velocity(),
    # env.muscle_force(),
    # env.muscle_activity(),
    # env.target,
    # env.sim.data.get_site_xpos(env.tracking_str)
    
    # model evluation
    parser.add_argument('--eval_every',              type = int, default = 10) # 10
    parser.add_argument('--n_eval_envs',             type = int, default = 2) # 50, 1000

    # MPPI
    parser.add_argument('--horizon',                 type = int, default = 50) # 7
    parser.add_argument('--n_sequences',             type = int, default = 200) # 200
    parser.add_argument('--reward_weighting_factor', type = float, default = 1.0)
    parser.add_argument('--noise_std',               type = float, default = 0.1)

    # optimisation
    parser.add_argument('--adam_b1',                 type = float, default = 0.9)
    parser.add_argument('--adam_b2',                 type = float, default = 0.999)
    parser.add_argument('--adam_eps',                type = float, default = 1e-8)
    parser.add_argument('--weight_decay',            type = float, default = 0) # 0.0001 (0 is adam not adamw)
    parser.add_argument('--max_grad_norm',           type = float, default = 10.0)
    parser.add_argument('--step_size',               type = float, default = 0.001)
    parser.add_argument('--decay_steps',             type = int, default = 1)
    parser.add_argument('--decay_factor',            type = float, default = 1) # 0.9999 (1 is constant learning rate)
    parser.add_argument('--print_every',             type = int, default = 50)
    parser.add_argument('--n_model_iterations',      type = int, default = 1000)
    parser.add_argument('--n_batches',               type = int, default = 50) # 50, 30
    parser.add_argument('--chunk_length',            type = int, default = 50) # 50, shouldn't this be equal to planning horizon?
    parser.add_argument('--n_updates',               type = int, default = 100)
    parser.add_argument('--min_delta',               type = float, default = 1e-3)
    parser.add_argument('--patience',                type = int, default = 2)

    args = parser.parse_args()

    # make sure that if you reset myosuite environment multiple times the initial state is always the same, if not you need to set it to be so
    # linear decay
    # LDS dims and number of CNN layers
    # is lambda lax scan inefficient
    # in evaluation, calculate loss using actual myosuite
    # probably don't use decaying learning rate (at least not straight away - maybe after myosuite loss has converged, though this will be convergence to local model) as loss function is non stationary so will converge too early
    # myosutie dt (0.02 is myosuite default)
    # sigmoid on muscle inputs?
    # be open minded about the choice of the scaling factor (and bias) on pen_state
    # i don't bound fingertip prediction, as should be on scale of 1 and centered on zero

    # to change an argument via the command line: python main.py --reload_folder_name 'run_1' --reload_state True

    # save the hyperparameters
    path = 'runs/' + args.folder_name + '/hyperparameters'
    os.makedirs(os.path.dirname(path))
    file = open(path, 'wb') # change 'wb' to 'rb' to load
    pickle.dump(args, file) # change to args = pickle.load(file) to load
    file.close()

    # from jax.config import config
    # config.update("jax_disable_jit", True)
    # config.update("jax_debug_nans", False)
    # type help at a breakpoint() to see available commands
    # use xeus-python kernel -- Python 3.9 (XPython) -- for debugging

    model, params, args, key = initialise_model(args)

    # import jax
    # jax.profiler.start_trace('runs/' + folder_name)
    optimise_model(model, params, args, key)
    # jax.profiler.stop_trace()

    # from train import render_rollout, get_train_state
    # from brax.v1 import envs
    # from jax import random
    # from controllers import MPPI
    # import numpy as np
    # env = envs.create(env_name = args.environment_name)
    # mppi = MPPI(env, args)
    # state, _ = get_train_state(model, params, args)
    # iteration = 1
    # n_targets = 1
    # actions = np.empty((env.action_size, args.time_steps, n_targets))
    # for target in range(n_targets):
    #     key = random.PRNGKey(args.jax_seed + target)
    #     actions[:, :, target], cumulative_reward = render_rollout(env, mppi, state, iteration, args, key)

    # from matplotlib import pyplot as plt
    # for target in range(n_targets):
    #     plt.plot(actions[0, :, target], actions[1, :, target], '.')
    # plt.show()
    # breakpoint()

if __name__ == '__main__':

    main()
