from jax import value_and_grad, vmap, jit, random, lax
from functools import partial
import jax.numpy as np
import numpy as onp
import optax
from controllers import CEM
from flax.training import checkpoints, train_state
from flax.training.early_stopping import EarlyStopping
from utils import keyGen, print_metrics, write_metrics_to_tensorboard
from flax.metrics import tensorboard
import time
from copy import copy
from brax import envs

def get_train_state(model, param, args):
    
    lr_scheduler = optax.exponential_decay(args.step_size, args.decay_steps, args.decay_factor)

    optimiser = optax.chain(optax.adamw(learning_rate = lr_scheduler, b1 = args.adam_b1, b2 = args.adam_b2, eps = args.adam_eps, 
                            weight_decay = args.weight_decay), optax.clip_by_global_norm(args.max_grad_norm))
    
    state = train_state.TrainState.create(apply_fn = model.apply, params = param, tx = optimiser)

    if args.reload_state:

        state = checkpoints.restore_checkpoint(ckpt_dir = 'runs/' + args.reload_folder_name, target = state)

    return state, lr_scheduler

def create_tensorboard_writer(args):

    # create a tensorboard writer
    # to view tensorboard results, call 'tensorboard --logdir=.' in runs folder from terminal
    writer = tensorboard.SummaryWriter('runs/' + args.folder_name)

    return writer

def loss_fn(params, state, inputs, target_outputs):

    batch_dynamics_model = vmap(state.apply_fn, in_axes = (None, 0))

    # predict the next observation using the learned myosuite dynamics model
    predicted_outputs = batch_dynamics_model(params, inputs)

    # mean squared error between the predicted and actual (i.e. myosuite) fingertip position
    loss = ((target_outputs - predicted_outputs) ** 2).mean()

    return loss

loss_grad = value_and_grad(loss_fn)

def train_step(state, inputs, outputs):
    
    loss, grads = loss_grad(state.params, state, inputs, outputs)

    state = state.apply_gradients(grads = grads)
    
    return state, loss

train_step_jit = jit(train_step)

def collect_data(env, is_random_policy, state, key, args):

    def environment_step(carry, inputs):

        def get_random_policy(action_size, horizon, observation, state, key):

            action = random.uniform(key, shape = (action_size,), minval = -1.0, maxval = 1.0)

            return action

        def get_mpc_action(action_size, horizon, observation, state, key):

            action = controller.get_action(observation, state.apply_fn, state.params, horizon, key)
            # action = controller.jit_get_action(observation, state.apply_fn, state.params, horizon, key)

            return action

        env_state, cumulative_reward = carry
        key, horizon = inputs

        observation = np.copy(env_state.obs)

        # currently, no need to pass horizon into these functions below, as args.horizon not used. horizon is already traced, i think due to scan on environment_step - need to use partial on that first
        action = lax.cond(is_random_policy, partial(get_random_policy, env.action_size, horizon), partial(get_mpc_action, env.action_size, horizon), observation, state, key)

        # perform a step in the environment
        # env_state = jit_env_step(env_state, action)
        env_state = env.step(env_state, action)

        next_observation = np.copy(env_state.obs)
        
        cumulative_reward += env_state.reward

        carry = env_state, cumulative_reward
        outputs = observation, action, next_observation

        return carry, outputs

    # self.low_val = -1 * np.ones(self.env.action_space.low.shape)
    # self.high_val = np.ones(self.env.action_space.high.shape)
    # action = np.random.uniform(self.low_val, self.high_val, self.shape)
    
    # preallocate memory for inputs
    inputs_dynamics = onp.empty((args.n_rollouts, args.time_steps, env.observation_size + env.action_size))
    outputs_dynamics = onp.empty((args.n_rollouts, args.time_steps, env.observation_size))

    jit_env_step = jit(env.step)

    # counter = 0

    controller = CEM(env, args)

    print('play with using pmap for running rollout in parallel?')

    cumulative_rewards = onp.empty(args.n_rollouts)
    for rollout in range(args.n_rollouts):

        key, subkeys = keyGen(key, n_subkeys = 2)

        # randomly reset the environment for each new rollout
        env_state = env.reset(rng = next(subkeys))
        # env_state = env.reset(rng = jp.random_prngkey(seed = 0))
        cumulative_reward = 0
        carry = env_state, cumulative_reward

        subkeys = random.split(next(subkeys), args.time_steps)
        # horizon = np.ones(args.time_steps) * args.horizon # same horizon for all time steps
        horizon = np.concatenate((np.ones(args.time_steps - args.horizon + 1) * args.horizon, np.flip(np.arange(1, args.horizon)))) # truncate the horizon if it extends beyond the episode duration
        inputs = subkeys, horizon

        (_, cumulative_reward), (observation, action, next_observation) = lax.scan(environment_step, carry, inputs)
        
        inputs_dynamics[rollout,:,:] = np.concatenate((observation, action), axis = 1)
        outputs_dynamics[rollout,:,:] = next_observation

        cumulative_rewards[rollout] = cumulative_reward

    mean_cumulative_reward = cumulative_rewards.mean()

    return inputs_dynamics, outputs_dynamics, mean_cumulative_reward

def reshape_dataset(dataset, args):

    # reshape a.size / a.shape[2]
    dataset = dataset.reshape((int(dataset.size / dataset.shape[2] / args.batch_size), args.batch_size, dataset.shape[2]))

    # convert to jax numpy array
    dataset = np.array(dataset)

    return dataset

# def calculate_fingertip_position():

#     return 

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

    # loop over iterations
    for iteration in range(args.n_model_iterations):

        ###########################################
        ########## collect data ###################
        ###########################################

        # collect dataset
        key, subkey = keyGen(key, n_subkeys = 1)

        if iteration == 0:

            is_random_policy = True

        else:

            is_random_policy = False

        inputs_dynamics, outputs_dynamics, mean_cumulative_reward = collect_data(env, is_random_policy, state, next(subkey), args)

        print('iteration: {}, mean cumulative reward across rollouts: {:.4f}'.format(iteration, mean_cumulative_reward))

        # reshape dataset into batches
        inputs_dynamics = reshape_dataset(inputs_dynamics, args)
        outputs_dynamics = reshape_dataset(outputs_dynamics, args)

        ###########################################
        ########## train dynamics model ###########
        ###########################################

        # loop over epochs
        for epoch in range(args.n_epochs):
            
            # start epoch timer
            epoch_start_time = time.time()

            # generate subkeys
            # key, training_subkeys = keyGen(key, n_subkeys = args.n_batches)

            # initialise the losses and the timer
            training_loss = 0
            batch_start_time = time.time()

            # loop over batches
            for batch in range(1, inputs_dynamics.shape[0] + 1):

                # train the  dynamics model
                state, mse = train_step_jit(state, inputs_dynamics[batch,:,:], outputs_dynamics[batch,:,:])

                # average training loss
                training_loss = training_loss + mse / (inputs_dynamics.shape[0] + 1)

            if epoch == 0 or epoch == args.n_epochs - 1:

                # end batches timer
                epoch_duration = time.time() - epoch_start_time

                # print metrics
                print_metrics("batch", epoch_duration, training_loss, batch_range = [batch - args.print_every + 1, batch], 
                              lr = lr_scheduler(state.step - 1))


                # # training losses (average of 'print_every' batches)
                # training_loss = training_loss + mse / args.print_every

                # if batch % args.print_every == 0:
                # if batch == inputs.shape[0] + 1 and (epoch == args.n_epochs - 1 or epoch == 0):

                #     # end batches timer
                #     batches_duration = time.time() - batch_start_time

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



        if iteration % 10 == 0:

            # save checkpoint
            checkpoints.save_checkpoint(ckpt_dir = 'runs/' + args.folder_name, target = state, step = epoch)


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
