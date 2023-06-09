from jax import value_and_grad, vmap, jit, random, lax
from functools import partial
import jax.numpy as np
import numpy as onp
import optax
from controllers import MPPI, CEM
from flax.training import train_state
from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler
from flax.training.early_stopping import EarlyStopping
from utils import keyGen, stabilise_variance, print_metrics, create_tensorboard_writer, write_metrics_to_tensorboard, save_object_using_pickle
import time
from copy import deepcopy
import gym
import warmup
from render import save_frames_as_gif
import imageio
import os

# from mujoco_py import GlfwContext
# GlfwContext(offscreen = True) # this is to avoid a GLEW initialization error when rendering using rgb_array mode (https://github.com/openai/mujoco-py/issues/390)

def get_train_states(models, params, args, ckptr):

    lr_scheduler = optax.exponential_decay(args.step_size, args.decay_steps, args.decay_factor)

    optimiser = optax.chain(optax.adamw(learning_rate = lr_scheduler, b1 = args.adam_b1, b2 = args.adam_b2, eps = args.adam_eps, 
                            weight_decay = args.weight_decay), optax.clip_by_global_norm(args.max_grad_norm))

    states = []
    for i, (model, param) in enumerate(zip(models, params)):
        
        states.append(train_state.TrainState.create(apply_fn = model.apply, params = param, tx = optimiser))

        if args.reload_state:

            if i == 0:

                states.append(ckptr.restore('runs/' + args.reload_folder_name + '/checkpoints/dynamics_model', states[0]))

            elif i == 1:

                states.append(ckptr.restore('runs/' + args.reload_folder_name + '/checkpoints/VAE', states[1]))

    return states, lr_scheduler

def get_action_CEM(controller, action_sequence_mean, observation, state, key):

    action, action_sequence_mean = controller.get_action(action_sequence_mean, observation, state, key)

    return action, action_sequence_mean

jit_get_action_CEM = jit(get_action_CEM, static_argnums = (0,))

def get_action_MPPI(controller, action_sequence_mean, observation, state, key):

    action, action_sequence_mean = controller.get_action(action_sequence_mean, observation, state, key)

    return action, action_sequence_mean

jit_get_action_MPPI = jit(get_action_MPPI, static_argnums = (0,))

def render_rollout(env, controller, state, iteration, args, key, seed, render_prediction):

    # reset the environment
    env.seed(seed)
    initial_observation = np.copy(env.reset())

    # initialise the mean of the action sequence distribution
    action_sequence_mean = np.zeros((args.horizon, env.action_space.shape[0]))

    # preallocate
    frames = []
    cumulative_reward = 0
    actions = np.empty((args.time_steps, env.action_space.shape[0]))

    key, subkeys = keyGen(key, n_subkeys = args.time_steps)
    for time in range(args.time_steps):

        if args.controller == 'MPPI':

            action, action_sequence_mean = jit_get_action_MPPI(controller, action_sequence_mean, env.env._get_obs(), state, key)

        elif args.controller == 'CEM':

            action, action_sequence_mean = jit_get_action_CEM(controller, action_sequence_mean, env.env._get_obs(), state, key)
        # action, action_sequence_mean = get_mpc_action(controller, state, env, action_sequence_mean, args, next(subkeys))
        
        _, reward, _, _ = env.step(action)

        frames.append(env.render(mode = "rgb_array"))
        cumulative_reward += reward
        actions = actions.at[time, :].set(action)

    # save a gif of the rollout
    path = 'runs/' + args.folder_name + '/gifs/'
    rollout_filename = 'seed' + str(seed) + '_iteration' + str(iteration) + '.gif'
    save_frames_as_gif(frames, path = path, filename = rollout_filename)

    if render_prediction:

        # use the learned dynamics model to predict the sequence of observations given the initial observation and the sequence of actions
        mu, log_var, _ = state.apply_fn(state.params, initial_observation, actions)

        predicted_frames = []
        for time in range(args.time_steps):

            if time == 0:

                predicted_frames.append(frames[0])

            else:

                env.set_state(np.concatenate((mu[time - 1, :2], env.target[:2])), np.zeros(4,))
                predicted_frames.append(env.render(mode = "rgb_array"))

        # save a gif of the predicted rollout
        predicted_rollout_filename = 'predicted_rollout.gif'
        save_frames_as_gif(frames, path = path, filename = predicted_rollout_filename)

        # horizontally stack the frames of the true and predicted rollouts
        stacked_frames = [onp.hstack((i, j)) for i, j in zip(frames, predicted_frames)]

        # save a gif showing the true (left) and predicted (right) rollouts side-by-side
        stacked_gif_filename = 'seed' + str(seed) + '_iteration' + str(iteration) + '_prediction' + '.gif'
        save_frames_as_gif(stacked_frames, path = path, filename = stacked_gif_filename)

        # delete the original gifs
        os.remove(path + rollout_filename)
        os.remove(path + predicted_rollout_filename)

    return actions, cumulative_reward

def evaluate_model(env, controller, state, iteration, args):

    cumulative_reward_list = []
    for seed in range(args.n_envs_render):

        key = random.PRNGKey(seed)

        _, cumulative_reward = render_rollout(env, controller, state, iteration, args, key, seed, render_prediction = True)
        
        cumulative_reward_list.append(cumulative_reward)

    print("cumulative_reward_list:", cumulative_reward_list)

def log_likelihood_diagonal_Gaussian(x_including_nans, mu, log_var):
    """
    Calculate the log likelihood of x under a diagonal Gaussian distribution
    var_min is added to the variances for numerical stability
    """
    log_var = stabilise_variance(log_var)

    # set nans to 0 to avoid nan gradients
    x = np.where(np.isnan(x_including_nans), x = 0, y = x_including_nans)

    # calculate the log likelihood
    log_likelihood = -0.5 * (log_var + np.log(2 * np.pi) + (x - mu) ** 2 / np.exp(log_var))

    # set terms associated with nans to zero
    log_likelihood = np.where(np.isnan(x_including_nans), x = 0, y = log_likelihood)
    
    return np.sum(log_likelihood)

def dynamics_loss_fn(params, state, actions, observations):

    # use the learned dynamics model to predict the sequence of observations given the initial observation and the sequence of actions
    mu, log_var, _ = state.apply_fn(params, observations[0,:], actions)

    # calculate the negative log probability of the sequence of observations
    loss = -log_likelihood_diagonal_Gaussian(observations[1:,:] - observations[0,:], mu, log_var)

    return loss

dynamics_batch_loss_fn = vmap(dynamics_loss_fn, in_axes = (None, None, 0, 0))

def dynamics_loss(params, state, actions, observations):

    batch_loss = dynamics_batch_loss_fn(params, state, actions, observations).mean()

    return batch_loss

dynamics_loss_grad = value_and_grad(dynamics_loss)

def sample_sequence_chunk(actions, observations, chunk_length, time_steps, key):

    key, subkeys = keyGen(key, n_subkeys = 2)

    # sample an episode
    e = random.randint(next(subkeys), shape = (1,), minval = 0, maxval = observations.shape[0])

    # sample a time step
    t = random.randint(next(subkeys), shape = (1,), minval = 0, maxval = time_steps) # observations.shape[1] - chunk_length)

    sampled_actions = lax.dynamic_slice(actions, (e[0], t[0], 0), (1, chunk_length, actions.shape[2]))[0,:,:]
    sampled_observations = lax.dynamic_slice(observations, (e[0], t[0], 0), (1, chunk_length + 1, observations.shape[2]))[0,:,:]

    return sampled_actions, sampled_observations

batch_sample_sequence_chunk = vmap(sample_sequence_chunk, in_axes = (None, None, None, None, 0))

def dynamics_train_step(state, actions, observations, key, n_batches, chunk_length, time_steps):

    subkeys = random.split(key, n_batches)

    sampled_actions, sampled_observations = batch_sample_sequence_chunk(actions, observations, chunk_length, time_steps, subkeys)
    
    loss, grads = dynamics_loss_grad(state.params, state, sampled_actions, sampled_observations)

    state = state.apply_gradients(grads = grads)
    
    return state, loss

def KL_diagonal_Gaussians(mu_1, log_var_1, mu_0, log_var_0):
        """
        KL(q||p), where q is posterior and p is prior
        mu_1, log_var_1 is the mean and log variances of the posterior
        mu_0, log_var_0 is the mean and log variances of the prior
        var_min is added to the variances for numerical stability
        """
        log_var_0 = stabilise_variance(log_var_0)
        log_var_1 = stabilise_variance(log_var_1)

        return np.sum(0.5 * (log_var_0 - log_var_1 + np.exp(log_var_1 - log_var_0) 
                             - 1.0 + (mu_1 - mu_0)**2 / np.exp(log_var_0)))

def VAE_loss_fn(params, state, action, key):

    # pass the action through the VAE to obtain the encoder and decoder distributions
    p = state.apply_fn({'params': params['params']}, action, key)

    # kl divergence between the approximate posterior and the prior
    mu_0 = 0
    log_var_0 = params['prior_z_log_var']
    kl = KL_diagonal_Gaussians(p['p_z']['mean'], p['p_z']['log_var'], mu_0, log_var_0)

    # calculate the negative log likelihood of the action
    nll = -log_likelihood_diagonal_Gaussian(action, p['p_a']['mean'], p['p_a']['log_var'])

    loss = kl + nll

    return loss

VAE_batch_loss_fn = vmap(VAE_loss_fn, in_axes = (None, None, 0, 0))

def VAE_loss(params, state, actions, subkeys):

    batch_loss = VAE_batch_loss_fn(params, state, actions, subkeys).mean()

    return batch_loss

VAE_loss_grad = value_and_grad(VAE_loss)

def sample_action(actions, key):

    key, subkeys = keyGen(key, n_subkeys = 2)

    # sample an episode
    e = random.randint(next(subkeys), shape = (1,), minval = 0, maxval = actions.shape[0])

    # sample a time step
    t = random.randint(next(subkeys), shape = (1,), minval = 0, maxval = actions.shape[1])

    sampled_action = lax.dynamic_slice(actions, (e[0], t[0], 0), (1, 1, actions.shape[2]))[0, 0, :]

    return sampled_action

batch_sample_action = vmap(sample_action, in_axes = (None, 0))

def VAE_train_step(state, actions, key, n_batches):

    subkeys = random.split(key, n_batches)

    sampled_actions = batch_sample_action(actions, subkeys)

    subkeys = random.split(key, n_batches)
    
    loss, grads = VAE_loss_grad(state.params, state, sampled_actions, subkeys)

    state = state.apply_gradients(grads = grads)
    
    return state, loss

# def sample_candidate_action_sequences(controller, action_sequence_mean, key):

#     action_sequences = controller.sample_candidate_action_sequences(action_sequence_mean, key)

#     return action_sequences

# jit_sample_candidate_action_sequences = jit(sample_candidate_action_sequences, static_argnums = (0,))

# def sample_candidate_action_sequences(controller, action_sequence_mean, action_sequence_variance, key):

#     action_sequences = controller.sample_candidate_action_sequences(action_sequence_mean, action_sequence_variance, key)

#     return action_sequences

# jit_sample_candidate_action_sequences = jit(sample_candidate_action_sequences, static_argnums = (0,))

# def estimate_cumulative_reward(state, observation, action_sequences):

#     _, _, estimated_cumulative_reward = state.apply_fn(state.params, observation, action_sequences)

#     return estimated_cumulative_reward

# jit_estimate_cumulative_reward = jit(vmap(estimate_cumulative_reward, in_axes = (None, None, 2)))

# def update_action_sequence_mean(controller, estimated_cumulative_reward, action_sequences):

#     action, action_sequence_mean = controller.update_action_sequence_mean(estimated_cumulative_reward, action_sequences)

#     return action, action_sequence_mean

# jit_update_action_sequence_mean = jit(update_action_sequence_mean, static_argnums = (0,))

# def get_mpc_action(controller, state, env, action_sequence_mean, args, key): # MPPI with ground truth option

#     action_sequences = jit_sample_candidate_action_sequences(controller, action_sequence_mean, key)

#     if args.ground_truth_dynamics:

#         estimated_cumulative_reward = np.zeros((action_sequences.shape[2]))
#         for rollout in range(action_sequences.shape[2]):

#             env_rollout = deepcopy(env)
#             for time in range(action_sequences.shape[0]):

#                 _, reward, _, _ = env_rollout.step(action_sequences[time, :, rollout])

#                 estimated_cumulative_reward = estimated_cumulative_reward.at[rollout].set(estimated_cumulative_reward[rollout] + reward)

#     else:

#         estimated_cumulative_reward = jit_estimate_cumulative_reward(state, env.env._get_obs(), action_sequences)

#     action, action_sequence_mean = jit_update_action_sequence_mean(controller, estimated_cumulative_reward, action_sequences)

#     return action, action_sequence_mean

# def get_mpc_action(controller, state, env, action_sequence_mean, args, key): # CEM with ground truth option

#     action_sequence_variance = np.ones((args.horizon, env.action_space.shape[0])) * 0.1 ** 2

#     for i in range(args.CEM_iterations):

#         action_sequences = jit_sample_candidate_action_sequences(controller, action_sequence_mean, action_sequence_variance, key)

#         if args.ground_truth_dynamics:

#             estimated_cumulative_reward = np.zeros((action_sequences.shape[2]))
#             for rollout in range(action_sequences.shape[2]):

#                 env_rollout = deepcopy(env)
#                 for time in range(action_sequences.shape[0]):

#                     _, reward, _, _ = env_rollout.step(action_sequences[time, :, rollout])

#                     estimated_cumulative_reward = estimated_cumulative_reward.at[rollout].set(estimated_cumulative_reward[rollout] + reward)

#         else:

#             estimated_cumulative_reward = jit_estimate_cumulative_reward(state, env.env._get_obs(), action_sequences)

#         action_sequence_mean, action_sequence_variance = jit_update_action_sequence_mean(controller, estimated_cumulative_reward, action_sequences)

#     action = np.copy(action_sequence_mean[0, :])

#     # warmstarting (reuse the plan from the previous planning step)
#     # amortise the optimisation process across multiple planning steps
#     # shift the mean of the action sequence distribution by one time point
#     # this implicitly copies/repeats the last time point (alternatively it could be set to 0)
#     action_sequence_mean = action_sequence_mean.at[:-1, :].set(action_sequence_mean[1:, :])

#     return action, action_sequence_mean

def environment_step(env, controller, state, action_sequence_mean, args, key):

    # action, action_sequence_mean = get_mpc_action(controller, state, env, action_sequence_mean, args, key)
    if args.controller == 'MPPI':

        action, action_sequence_mean = jit_get_action_MPPI(controller, action_sequence_mean, env.env._get_obs(), state, key)

    elif args.controller == 'CEM':

        action, action_sequence_mean = jit_get_action_CEM(controller, action_sequence_mean, env.env._get_obs(), state, key)

    _, reward, _, _ = env.step(action)

    next_observation = np.copy(env.env._get_obs())

    return action, action_sequence_mean, next_observation, reward

def perform_rollout(state, env, controller, args, key):

    # randomly reset the environment
    env_state = env.reset()

    # initial observation
    initial_observation = np.copy(env.env._get_obs())

    # initialise the cumulative reward
    cumulative_reward = 0

    # initialise the mean of the action sequence distribution
    action_sequence_mean = np.zeros((args.horizon, env.action_space.shape[0]))
    
    key, subkeys = keyGen(key, n_subkeys = args.time_steps)
    actions = np.empty((args.time_steps, env.action_space.shape[0]))
    observations = np.empty((args.time_steps, env.observation_space.shape[0]))
    for time in range(args.time_steps):

        action, action_sequence_mean, observation, reward = environment_step(env, controller, state, action_sequence_mean, args, next(subkeys))

        actions = actions.at[time,:].set(action)
        observations = observations.at[time,:].set(observation)
        cumulative_reward += reward

    return actions, np.vstack((initial_observation, observations)), cumulative_reward

def collect_data(env, controller, state, key, args):

    key, subkeys = keyGen(key, n_subkeys = args.n_rollouts)
    
    all_actions = np.empty((args.n_rollouts, args.time_steps, env.action_space.shape[0]))
    all_observations = np.empty((args.n_rollouts, args.time_steps + 1, env.observation_space.shape[0]))
    all_cumulative_rewards = np.empty((args.n_rollouts, args.time_steps))
    for rollout in range(args.n_rollouts):

        actions, observations, cumulative_rewards = perform_rollout(state, env, controller, args, next(subkeys))

        all_actions = all_actions.at[rollout,:,:].set(actions)
        all_observations = all_observations.at[rollout,:,:].set(observations)
        all_cumulative_rewards = all_cumulative_rewards.at[rollout,:].set(cumulative_rewards)

    mean_cumulative_reward = all_cumulative_rewards.mean()

    return all_actions, all_observations, mean_cumulative_reward

def pad_data(data, args, fill_value = np.nan):

    padded_data = np.concatenate((data, np.full((data.shape[0], args.horizon, data.shape[2]), fill_value)), axis = 1)

    return padded_data

def optimise_model(models, params, args, key):

    # start optimisation timer
    optimisation_start_time = time.time()

    # create a checkpoint object for saving and restoring checkpoints
    ckptr = Checkpointer(PyTreeCheckpointHandler())
    
    # create train state
    states, lr_scheduler = get_train_states(models, params, args, ckptr)

    # instantiate an environment
    env = gym.make(args.environment_name) # openai gym

    # set the random seed of the environment # openai gym
    key, subkeys = keyGen(key, n_subkeys = 1)
    random_seed = int(random.randint(next(subkeys), (1,), 0, 1000000))
    env.seed(random_seed)

    # set early stopping criteria
    early_stop = EarlyStopping(min_delta = args.min_delta, patience = args.patience)

    # create tensorboard writer
    writer = create_tensorboard_writer(args)

    if args.controller == 'MPPI':

        controller = MPPI(env, args)

    elif args.controller == 'CEM':

        controller = CEM(env, args)

    dynamics_train_step_jit = jit(partial(dynamics_train_step, n_batches = args.n_batches, chunk_length = args.chunk_length, time_steps = args.time_steps))
    VAE_train_step_jit = jit(partial(VAE_train_step, n_batches = args.n_batches))

    for iteration in range(args.n_model_iterations):

        # periodically evaluate the model
        if iteration % args.render_every == 0:

            evaluate_model(env, controller, states[0], iteration, args)

        ###########################################
        ########## collect data ###################
        ###########################################

        # collect dataset
        key, subkeys = keyGen(key, n_subkeys = args.n_updates * 2 + 1)

        data_start_time = time.time()
        actions, observations, mean_cumulative_reward = collect_data(env, controller, states[0], next(subkeys), args)
        print("time collecting data =",time.time() - data_start_time)

        print('iteration: {}, mean cumulative reward across rollouts: {:.4f}'.format(iteration, mean_cumulative_reward))

        if iteration == 0:

            all_actions = pad_data(actions, args, fill_value = 0)
            all_observations = pad_data(observations, args)

        else:

            all_actions = np.vstack((all_actions, pad_data(actions, args, fill_value = 0)))
            all_observations = np.vstack((all_observations, pad_data(observations, args)))

        print('iteration: {}, number of episodes in dataset: {:.4f}'.format(iteration, all_actions.shape[0]))

        ###########################################
        ########## train dynamics model ###########
        ###########################################

        # initialise the losses and the timer
        dynamics_training_loss = 0
        VAE_training_loss = 0
        train_start_time = time.time()
        for update in range(args.n_updates):

            # perform a parameter update for the dynamics model
            states[0], loss = dynamics_train_step_jit(states[0], all_actions, all_observations, next(subkeys))

            # compute the average training loss for the dynamics model
            dynamics_training_loss = (dynamics_training_loss * update + loss) / (update + 1)

            # perform a parameter update for the VAE
            states[1], loss = VAE_train_step_jit(states[1], all_actions, next(subkeys))

            # compute the average training loss for the VAE
            VAE_training_loss = (VAE_training_loss * update + loss) / (update + 1)

            if update == args.n_updates - 1:

                # calculate duration
                train_duration = time.time() - train_start_time

                # print metrics
                print_metrics("batch", train_duration, dynamics_training_loss, batch_range = [update - args.print_every + 1, update], 
                              lr = lr_scheduler(states[0].step - 1))

                # print metrics
                print_metrics("batch", train_duration, VAE_training_loss, batch_range = [update - args.print_every + 1, update], 
                              lr = lr_scheduler(states[1].step - 1))

                # # training losses (average of 'print_every' batches)
                # training_loss = training_loss + mse / args.print_every

                # if batch % args.print_every == 0:
                # if batch == inputs.shape[0] + 1 and (epoch == args.n_epochs - 1 or epoch == 0):

                #     # end batches timer
                #     batches_duratconcatenateion = time.time() - batch_start_time

                #     # print metrics
                #     print_metrics("batch", batches_duration, training_loss, batch_range = [batch - args.print_every + 1, batch], 
                #                   lr = lr_scheduler(state.step - 1))

                #     # store losses
                #     if batch == args.print_every:
                        
                #         t_loss_through_training = deepcopy(training_loss)
                        
                #     else:
                        
                #         t_loss_through_training = np.append(t_loss_through_training, training_loss)

                #     # re-initialise the losses and timer
                #     training_loss = 0
                #     batch_start_time = time.time()

        if iteration % args.save_trajectories_every == 0:

            save_object_using_pickle({'actions': all_actions, 'observations': all_observations}, path = 'runs/' + args.folder_name + '/trajectory_data/', filename = 'iteration_' + str(iteration))

        if iteration % args.checkpoint_every == 0:

            # save checkpoint
            ckptr.save('runs/' + args.folder_name + '/checkpoints/dynamics_model', states[0])
            ckptr.save('runs/' + args.folder_name + '/checkpoints/VAE', states[1])

            # if epoch % 50 == 0:

                # # calculate loss on validation data
                # _, (validation_loss, output) = eval_step_jit(state.params, state, state_myo, validate_dataset, kl_weight, next(validation_subkeys))

                # # end epoch timer
                # epoch_duration = time.time() - epoch_start_time
                
                # # print losses (mean over all batches in epoch)
                # print_metrics("epoch", epoch_duration, t_loss_through_training, validation_loss, epoch = epoch + 1)

                # # write metrics to tensorboard
                # write_metrics_to_tensorboard(writer, t_loss_through_training, validation_loss, epoch)
                
                # # save checkpoint
                # checkpoints.save_checkpoint(ckpt_dir = 'runs/' + args.folder_name, target = state, step = epoch)
            
                # # if early stopping criteria met, break
                # _, early_stop = early_stop.update(validation_loss['total'].mean())
                # if early_stop.should_stop:
                    
                #     print('Early stopping criteria met, breaking...')
                    
                #     break

    optimisation_duration = time.time() - optimisation_start_time

    print('Optimisation finished in {:.2f} hours.'.format(optimisation_duration / 60 ** 2))
            
    return
