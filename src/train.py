from jax import value_and_grad, vmap, jit, random, lax
from functools import partial
import jax.numpy as np
import optax
from controllers import MPPI
from flax.training import checkpoints, train_state
from flax.training.early_stopping import EarlyStopping
from utils import keyGen, stabilise_variance, print_metrics, create_tensorboard_writer, write_metrics_to_tensorboard
import time
from copy import copy
from brax.v1 import envs
from brax.v1.io import image
from brax.v1.io.image import _eye, _up
from IPython.display import Image 
from pytinyrenderer import TinyRenderCamera as Camera

def get_train_state(model, param, args):
    
    lr_scheduler = optax.exponential_decay(args.step_size, args.decay_steps, args.decay_factor)

    optimiser = optax.chain(optax.adamw(learning_rate = lr_scheduler, b1 = args.adam_b1, b2 = args.adam_b2, eps = args.adam_eps, 
                            weight_decay = args.weight_decay), optax.clip_by_global_norm(args.max_grad_norm))
    
    state = train_state.TrainState.create(apply_fn = model.apply, params = param, tx = optimiser)

    if args.reload_state:

        state = checkpoints.restore_checkpoint(ckpt_dir = 'runs/' + args.reload_folder_name, target = state)

    return state, lr_scheduler

def get_camera(env, env_state, width, height):

    sys = env.sys
    qp = env_state.qp
    ssaa = 2
    eye, up = _eye(sys, qp), _up(sys)
    hfov = 5.0
    vfov = hfov * height / width
    target = [qp.pos[0, 0], qp.pos[0, 1], 0]
    camera = Camera(
        viewWidth = width * ssaa,
        viewHeight = height * ssaa,
        position = eye,
        target = target,
        up = up,
        hfov = hfov,
        vfov = vfov)

    return camera

def render_rollout(env, controller, state, iteration, args, key):

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
    for time in range(args.time_steps):

        if iteration == 0:

            action, action_sequence_mean = random_action(controller, env, env_state, state, action_sequence_mean, next(subkeys))

        else:

            action, action_sequence_mean = jit_mpc_action(env_state, state, action_sequence_mean, next(subkeys))

        actions = actions.at[:, time].set(action)
        env_state = jit_env_step(env_state, action)
        rollout.append(env_state)
        cameras.append(get_camera(env, env_state, width, height))

    # (_, cumulative_reward, _, _, _), (observation, action, next_observation) = lax.scan(partial(environment_step, env, controller), carry, subkeys)

    # create and save a gif of the rollout
    gif = Image(image.render(env.sys, [s.qp for s in rollout], cameras = cameras, width = width, height = height, fmt = 'gif'))
    open('runs/' + args.folder_name + '/output_' + str(iteration) + '.gif', 'wb').write(gif.data)

    return actions

def log_likelihood_diagonal_Gaussian(x, mu, log_var):
    """
    Calculate the log likelihood of x under a diagonal Gaussian distribution
    var_min is added to the variances for numerical stability
    """
    log_var = stabilise_variance(log_var)
    
    return np.sum(-0.5*(log_var + np.log(2*np.pi) + (x - mu)**2/np.exp(log_var)), axis = (1, 2))

def loss_fn(params, state, actions, observations):

    batch_rollout_prediction = vmap(state.apply_fn, in_axes = (None, 0, 0))

    # use the learned dynamics model to predict the sequence of observations given the initial observation and the sequence of actions
    mu, log_var, _ = batch_rollout_prediction(params, observations[:, 0, :], actions)

    # log probability of the sequence of observations
    loss = log_likelihood_diagonal_Gaussian(observations[:, 1:, :], mu + observations[:, 0, None, :], log_var).mean()

    return loss

loss_grad = value_and_grad(loss_fn)

def sample_sequence_chunk(actions, observations, chunk_length, key):

    key, subkeys = keyGen(key, n_subkeys = 2)

    # sample an episode
    e = jax.random.randint(next(subkeys), shape = (1,), minval = 0, maxval = observations.shape[0])

    # sample a time step
    t = jax.random.randint(next(subkeys), shape = (1,), minval = 0, maxval = observations.shape[1] - chunk_length)

    sampled_actions = actions[e,t:t + chunk_length,:]
    sampled_observations = observations[e,t:t + chunk_length + 1,:]

    return sampled_actions, sampled_observations

batch_sample_sequence_chunk = vmap(sample_sequence_chunk, in_axes = (None, None, None, 0))

def train_step(state, actions, observations, n_batches, chunk_length, key):

    subkeys = random.split(key, n_batches)

    sampled_actions, sampled_observations = batch_sample_sequence_chunk(actions, observations, chunk_length, subkeys):
    
    loss, grads = loss_grad(state.params, state, sampled_actions, sampled_observations)

    state = state.apply_gradients(grads = grads)
    
    return state, loss

train_step_jit = jit(train_step)

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

    print('play with using pmap for running rollout in parallel?')

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

    # loop over iterations
    for iteration in range(args.n_model_iterations):

        ###########################################
        ########## collect data ###################
        ###########################################

        # collect dataset
        key, subkey = keyGen(key, n_subkeys = args.n_updates + 1)

        if iteration == 0:

            is_random_policy = True

        else:

            is_random_policy = False

        actions, observations, mean_cumulative_reward = collect_data(env, is_random_policy, batch_perform_rollout, state, next(subkey), args)

        if iteration == 0:

            all_actions = np.copy(actions)
            all_observations = np.copy(observations)

        else:

            all_actions = np.vstack((all_actions, actions))
            all_observations = np.vstack((all_observations, observations))

        print('iteration: {}, mean cumulative reward across rollouts: {:.4f}'.format(iteration, mean_cumulative_reward))

        ###########################################
        ########## train dynamics model ###########
        ###########################################

        # initialise the losses and the timer
        training_loss = 0
        train_start_time = time.time()
        for update in range(args.n_updates):

            # perform a parameter update
            state, loss = train_step_jit(state, all_actions, all_observations, args.n_batches, args.chunk_length, next(subkey))

            # compute the average training loss
            training_loss = (training_loss * update + loss) / (update + 1)

            if update == 0 or update == args.n_updates - 1:

                # calculate duration
                train_duration = time.time() - train_start_time

                # print metrics
                print_metrics("batch", train_duration, training_loss, batch_range = [update - args.print_every + 1, update], 
                              lr = lr_scheduler(state.step - 1))

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
                        
                #         t_loss_through_training = copy(training_loss)
                        
                #     else:
                        
                #         t_loss_through_training = np.append(t_loss_through_training, training_loss)

                #     # re-initialise the losses and timer
                #     training_loss = 0
                #     batch_start_time = time.time()

        if iteration % 100 == 0:

            # save checkpoint
            checkpoints.save_checkpoint(ckpt_dir = 'runs/' + args.folder_name, target = state, step = iteration)


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

        # render a rollout and save a gif
        # render_rollout(env, mppi, state, iteration, args, next(subkey))

    optimisation_duration = time.time() - optimisation_start_time

    print('Optimisation finished in {:.2f} hours.'.format(optimisation_duration / 60 ** 2))
            
    return
