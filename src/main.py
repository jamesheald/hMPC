import argparse
import pickle
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from initialise import initialise_model
from train import optimise_model

def main():

    parser = argparse.ArgumentParser(description = 'hyperparameters')

    # directories
    parser.add_argument('--folder_name',             default = 'to_save_model')
    parser.add_argument('--reload_state',            type = bool, default = False)
    parser.add_argument('--reload_folder_name',      default = 'saved_model')

    # model
    parser.add_argument('--jax_seed',                type = int, default = 0)

    # gym environment
    # https://www.gymlibrary.dev/environments/mujoco/reacher/
    parser.add_argument('--environment_name',        default = 'reacher')
    parser.add_argument('--n_rollouts',              type = int, default = 30)
    parser.add_argument('--time_steps',              type = int, default = 50)

    # MPPI
    parser.add_argument('--horizon',                 type = int, default = 7)
    parser.add_argument('--n_sequences',             type = int, default = 200)
    parser.add_argument('--n_elite',                 type = int, default = 20)
    parser.add_argument('--alpha',                   type = float, default = 1.0)
    parser.add_argument('--min_variance',            type = float, default = 0.001)
    parser.add_argument('--reward_weighting_factor', type = float, default = 1.0)
    # parser.add_argument('--myosuite_dt',             type = float, default = 0.02)

    # optimisation
    parser.add_argument('--adam_b1',                 type = float, default = 0.9)
    parser.add_argument('--adam_b2',                 type = float, default = 0.999)
    parser.add_argument('--adam_eps',                type = float, default = 1e-8)
    parser.add_argument('--weight_decay',            type = float, default = 0.0001)
    parser.add_argument('--max_grad_norm',           type = float, default = 10.0)
    parser.add_argument('--step_size',               type = float, default = 0.001)
    parser.add_argument('--decay_steps',             type = int, default = 1)
    parser.add_argument('--decay_factor',            type = float, default = 0.9999)
    parser.add_argument('--print_every',             type = int, default = 50)
    parser.add_argument('--n_epochs',                type = int, default = 40)
    parser.add_argument('--n_model_iterations',      type = int, default = 500)
    parser.add_argument('--batch_size',              type = int, default = 30)
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

    # to change an argument via the command line: python main.py --folder_name 'run_1'

    # save the hyperparameters
    path = 'runs/' + args.folder_name + '/hyperparameters'
    os.makedirs(os.path.dirname(path))
    file = open(path, 'wb') # change 'wb' to 'rb' to load
    pickle.dump(args, file) # change to args = pickle.load(file) to load
    file.close()

    print('eventually remove if epoch % 50 == 0 code')

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

    # from utils import forward_pass_model
    # from jax import numpy as np
    # import tensorflow_datasets as tfds
    # train_dataset = np.array(list(tfds.as_numpy(train_dataset))[0]['image']).reshape(args.batch_size, 105, 105, 1)
    # output = forward_pass_model(models[0], params[0], train_dataset, state_myo, args, key)
    # from matplotlib import pyplot as plt
    # plt.imshow(output['output_images'])
    # plt.show()

if __name__ == '__main__':

    main()
