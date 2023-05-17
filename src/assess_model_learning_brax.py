from jax import value_and_grad, vmap, jit, random, lax
from functools import partial
import jax.numpy as np
import optax
from controllers import MPPI
from flax.training import checkpoints, train_state
from flax.training.early_stopping import EarlyStopping
from utils import keyGen, stabilise_variance, batch_calculate_theta, print_metrics, create_tensorboard_writer, write_metrics_to_tensorboard
import time
from copy import copy
from brax import envs

from brax.kinematics import forward
from render import forward_kinematics, get_camera, create_gif, merge_gifs
import imageio
import os

def get_train_state(model, param, args):
    
    lr_scheduler = optax.exponential_decay(args.step_size, args.decay_steps, args.decay_factor)

    optimiser = optax.chain(optax.adamw(learning_rate = lr_scheduler, b1 = args.adam_b1, b2 = args.adam_b2, eps = args.adam_eps, 
                            weight_decay = args.weight_decay), optax.clip_by_global_norm(args.max_grad_norm))
    
    state = train_state.TrainState.create(apply_fn = model.apply, params = param, tx = optimiser)

    if args.reload_state:

        state = checkpoints.restore_checkpoint(ckpt_dir = 'runs/' + args.reload_folder_name, target = state)

    return state, lr_scheduler

def render_observations(env, actions, observations, state, iteration):

    env_state = env.reset(rng = random.PRNGKey(0))
    pipeline_state = copy(env_state.pipeline_state)

    jit_forward = jit(forward)

    # use the learned dynamics model to predict the sequence of observations given the initial observation and the sequence of actions
    mu, log_var, _ = state.apply_fn(state.params, observations[0, [0, 1, 2, 3, 6, 7, 8, 9]], actions)

    true_rollout = []
    predicted_rollout = []
    time_steps = observations.shape[0]
    for time in range(time_steps):

        if time == 0:

            pipeline_state = forward_kinematics(env, observations[time,:], pipeline_state)
            true_rollout.append(pipeline_state)
            predicted_rollout.append(pipeline_state)

        else:

            pipeline_state = forward_kinematics(env, observations[time,:], pipeline_state)
            true_rollout.append(pipeline_state)

            # pipeline_state = forward_kinematics(env, mu[time - 1,:], pipeline_state)
            q = np.concatenate([mu[time - 1, :2], observations[time, 4:6]])
            x, _ = jit_forward(env.sys, q, np.zeros(4,))
            pipeline_state = pipeline_state.replace(x = x)
            predicted_rollout.append(pipeline_state)

    cameras = [get_camera()] * time_steps

    # create and save a gif of the state-based rollout
    create_gif(env, true_rollout, cameras, 'true_rollout.gif')

    # create and save a gif of the observation-based rollout
    create_gif(env, predicted_rollout, cameras, 'predicted_rollout.gif')

    # merge both gifs into one (side-by-side)
    gif1 = imageio.get_reader('true_rollout.gif')
    gif2 = imageio.get_reader('predicted_rollout.gif')
    merge_gifs(gif1, gif2, 'rollout_' + str(iteration) + '.gif', env.dt)

    # delete original gifs
    os.remove('true_rollout.gif')
    os.remove('predicted_rollout.gif')

    return None

def render_rollout(env, controller, state, iteration, args, key, seed = 0):

    # jit environment and mpc policy
    jit_env_step = jit(env.step)
    jit_mpc_action = jit(partial(mpc_action, controller, env))

    key, subkeys = keyGen(key, n_subkeys = args.time_steps + 1)

    env_state = env.reset(rng = next(subkeys))

    # initialise the mean of the action sequence distribution
    action_sequence_mean = np.zeros((args.horizon, env.action_size))

    # perform rollout
    rollout = []
    cameras = []
    width = 320
    height = 240
    actions = np.empty((env.action_size, args.time_steps))
    cumulative_reward = 0
    for time in range(args.time_steps):

        # action, action_sequence_mean = random_action(controller, env, env_state, state, action_sequence_mean, next(subkeys))
        action, action_sequence_mean = jit_mpc_action(env_state, state, action_sequence_mean, next(subkeys))

        actions = actions.at[:, time].set(action)
        env_state = jit_env_step(env_state, action)
        cumulative_reward += env_state.reward
        rollout.append(env_state)
        cameras.append(get_camera(env, env_state, width, height))

    # (_, cumulative_reward, _, _, _), (observation, action, next_observation) = lax.scan(partial(environment_step, env, controller), carry, subkeys)

    # create and save a gif of the rollout
    gif = Image(image.render(env.sys, [s.qp for s in rollout], cameras = cameras, width = width, height = height, fmt = 'gif'))
    open('runs/' + args.folder_name + '/seed' + str(seed) + '_iteration' + str(iteration) + '.gif', 'wb').write(gif.data)

    return actions, cumulative_reward

def evaluate_model(env, controller, state, iteration, args):

    cumulative_reward_list = []
    for seed in range(args.n_eval_envs):

        key = random.PRNGKey(seed)

        _, cumulative_reward = render_rollout(env, controller, state, iteration, args, key, seed)
        
        cumulative_reward_list.append(cumulative_reward)

    print("cumulative_reward_list:", cumulative_reward_list)

    return None

def log_likelihood_diagonal_Gaussian(x, mu, log_var):
    """
    Calculate the log likelihood of x under a diagonal Gaussian distribution
    var_min is added to the variances for numerical stability
    """
    log_var = stabilise_variance(log_var)
    
    return np.sum(-0.5*(log_var + np.log(2*np.pi) + (x - mu)**2/np.exp(log_var)))

def loss_fn(params, state, actions, observations):

    # use the learned dynamics model to predict the sequence of observations given the initial observation and the sequence of actions
    mu, log_var, _ = state.apply_fn(params, observations[0, [0, 1, 2, 3, 6, 7, 8, 9]], actions)

    theta = batch_calculate_theta(observations)

    # calculate the negative log probability of the sequence of observations
    loss = -log_likelihood_diagonal_Gaussian(np.concatenate((theta[1:, :], observations[1:, [6, 7, 8, 9]]), axis = 1), mu + np.concatenate((theta[0, :], observations[0, [6, 7, 8, 9]])), log_var)

    return loss

batch_loss_fn = vmap(loss_fn, in_axes = (None, None, 0, 0))

def loss(params, state, actions, observations):

    batch_loss = batch_loss_fn(params, state, actions, observations).mean()

    return batch_loss

loss_grad = value_and_grad(loss)

def sample_sequence_chunk(actions, observations, chunk_length, key):

    key, subkeys = keyGen(key, n_subkeys = 2)

    # sample an episode
    e = random.randint(next(subkeys), shape = (1,), minval = 0, maxval = observations.shape[0])

    # sample a time step
    t = random.randint(next(subkeys), shape = (1,), minval = 0, maxval = observations.shape[1] - chunk_length)

    sampled_actions = lax.dynamic_slice(actions, (e[0], t[0], 0), (1, chunk_length, actions.shape[2]))[0,:,:]
    sampled_observations = lax.dynamic_slice(observations, (e[0], t[0], 0), (1, chunk_length + 1, observations.shape[2]))[0,:,:]

    return sampled_actions, sampled_observations

batch_sample_sequence_chunk = vmap(sample_sequence_chunk, in_axes = (None, None, None, 0))

def train_step(state, actions, observations, key, n_batches, chunk_length):

    subkeys = random.split(key, n_batches)

    sampled_actions, sampled_observations = batch_sample_sequence_chunk(actions, observations, chunk_length, subkeys)
    
    loss, grads = loss_grad(state.params, state, sampled_actions, sampled_observations)

    state = state.apply_gradients(grads = grads)
    
    return state, loss

def random_action(controller, env, env_state, state, action_sequence_mean, key):

    action = random.uniform(key, shape = (env.action_size,), minval = -1.0, maxval = 1.0)

    return action, action_sequence_mean

def mpc_action(controller, env, env_state, state, action_sequence_mean, key):

    action, action_sequence_mean = controller.get_action(env, env_state, state, action_sequence_mean, key)

    return action, action_sequence_mean

def environment_step(env, controller, carry, key):

    env_state, cumulative_reward, is_random_policy, state, action_sequence_mean = carry

    action, action_sequence_mean = lax.cond(is_random_policy, partial(random_action, controller, env), partial(mpc_action, controller, env), env_state, state, action_sequence_mean, key)

    observation = np.copy(env_state.obs)

    env_state = env.step(env_state, action)

    next_observation = np.copy(env_state.obs)
    
    cumulative_reward += env_state.reward

    carry = env_state, cumulative_reward, is_random_policy, state, action_sequence_mean
    outputs = observation, action, next_observation

    return carry, outputs

def perform_rollout(is_random_policy, state, env, key, controller, time_steps, horizon):

    key, subkeys = keyGen(key, n_subkeys = 2)

    # randomly reset the environment for each new rollout
    env_state = env.reset(rng = next(subkeys))
    
    cumulative_reward = 0

    # initialise the mean of the action sequence distribution
    action_sequence_mean = np.zeros((horizon, env.action_size))
    
    carry = env_state, cumulative_reward, is_random_policy, state, action_sequence_mean

    subkeys = random.split(next(subkeys), time_steps)

    (_, cumulative_reward, _, _, _), (observation, action, next_observation) = lax.scan(partial(environment_step, env, controller), carry, subkeys)

    outputs = action, np.vstack((observation[0,:], next_observation)), cumulative_reward

    return outputs

def collect_data(env, is_random_policy, batch_perform_rollout, state, key, args):

    subkeys = random.split(key, args.n_rollouts)
    actions, observations, cumulative_rewards = batch_perform_rollout(is_random_policy, state, env, subkeys)
    
    mean_cumulative_reward = cumulative_rewards.mean()

    return actions, observations, mean_cumulative_reward

def optimise_model(model, params, args, key):

    # start optimisation timer
    optimisation_start_time = time.time()
    
    # create train state
    state, lr_scheduler = get_train_state(model, params, args)

    # instantiate a brax environment
    env = envs.create(env_name = args.environment_name)

    # set early stopping criteria
    early_stop = EarlyStopping(min_delta = args.min_delta, patience = args.patience)

    # create tensorboard writer
    writer = create_tensorboard_writer(args)

    mppi = MPPI(env, args)

    batch_perform_rollout = jit(vmap(partial(perform_rollout, controller = mppi, time_steps = args.time_steps, horizon = args.horizon), in_axes = (None, None, None, 0)), static_argnums = (2,))

    train_step_jit = jit(partial(train_step, n_batches = args.n_batches, chunk_length = args.chunk_length))

    # collect dataset
    key, subkey = keyGen(key, n_subkeys = 1)

    is_random_policy = True

    actions, observations, mean_cumulative_reward = collect_data(env, is_random_policy, batch_perform_rollout, state, next(subkey), args)

    all_actions = np.copy(actions)
    all_observations = np.copy(observations)

    for iteration in range(args.n_model_iterations):

        # periodically evaluate the model
        if iteration % args.eval_every == 0:

            render_observations(env, actions[0, :, :], observations[0, :, :], state, iteration)

        ###########################################
        ########## collect data ###################
        ###########################################

        # collect dataset
        key, subkey = keyGen(key, n_subkeys = args.n_updates)

        ###########################################
        ########## train dynamics model ###########
        ###########################################

        # initialise the losses and the timer
        training_loss = 0
        train_start_time = time.time()
        for update in range(args.n_updates):

            # perform a parameter update
            state, loss = train_step_jit(state, all_actions, all_observations, next(subkey))

            # compute the average training loss
            training_loss = (training_loss * update + loss) / (update + 1)

            if update == args.n_updates - 1:

                # calculate duration
                train_duration = time.time() - train_start_time

                # print metrics
                print_metrics("batch", train_duration, training_loss, batch_range = [update - args.print_every + 1, update], 
                              lr = lr_scheduler(state.step - 1))
            
    return
